"""
[실험 기록 — 달러인덱스 포함 SDE 5차원 버전]

이 버전은 나스닥(3) + 달러인덱스(1) + 국채금리(1) = 5차원으로 SDE 모델을 학습한
실험입니다. 국채금리만 추가했을 때(4차원, src/sde_model.py)와 비교했을 때
성능 개선이 거의 없었고, 연산 비용만 늘어나는 것을 확인해 최종 파이프라인에서는
달러인덱스를 제외했습니다.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import roots_hermite
import copy
from tqdm import tqdm

from google.colab import files
import io

plt.rc('font', family='NanumBarunGothic')
plt.rcParams['axes.unicode_minus'] = False


class Config:
    TEST_DAYS = 30
    VAL_RATIO = 0.1
    FEATURE_DIM = 5  # 나스닥(3) + 달러인덱스(1) + 국채금리(1)
    HIDDEN_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 1000
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP_VAL = 1.0
    PATIENCE = 100
    SIMULATION_PATHS = 100
    RETURN_CLIP_MIN = -0.07
    RETURN_CLIP_MAX = 0.07


def convert_volume(value):
    if isinstance(value, str):
        value = value.replace(',', '').strip().upper()
        if value == '-' or not value: return np.nan
        if value.endswith('K'): return float(value[:-1]) * 1_000
        if value.endswith('M'): return float(value[:-1]) * 1_000_000
        if value.endswith('B'): return float(value[:-1]) * 1_000_000_000
        try: return float(value)
        except ValueError: return np.nan
    elif isinstance(value, (int, float)): return float(value)
    return np.nan


def convert_change_percent(value):
    if isinstance(value, str):
        value = value.replace('%', '').strip()
        if value == '-' or not value: return np.nan
        try: return float(value) / 100.0
        except ValueError: return np.nan
    elif isinstance(value, (int, float)): return float(value) / 100.0
    return np.nan


def get_preprocessed_data(nasdaq_content, dxy_content, treasury_content, dxy_filename, treasury_filename):
    nasdaq_stream = io.StringIO(nasdaq_content.decode('utf-8'))
    nasdaq_df = pd.read_csv(nasdaq_stream, index_col='날짜', parse_dates=True)
    nasdaq_df.sort_index(inplace=True)

    column_map = {'open': ['시가', 'open', 'Open'], 'volume': ['거래량', 'volume', 'Volume'], 'change': ['변동 %', '변동', 'change', 'Change']}
    available_columns = {col.lower().strip(): col for col in nasdaq_df.columns}
    open_col = next((available_columns.get(key) for key in column_map['open']), None)
    volume_col = next((available_columns.get(key) for key in column_map['volume']), None)
    change_col = next((available_columns.get(key) for key in column_map['change']), None)
    if not all([open_col, volume_col, change_col]): raise KeyError("나스닥 CSV 열 매칭 실패")
    nasdaq_df = nasdaq_df.rename(columns={open_col: 'open', volume_col: 'volume', change_col: 'change'})
    nasdaq_df['open'] = pd.to_numeric(nasdaq_df['open'].astype(str).str.replace(',', ''), errors='coerce')
    nasdaq_df['volume'] = nasdaq_df['volume'].apply(convert_volume)
    nasdaq_df['change'] = nasdaq_df['change'].apply(convert_change_percent)

    dxy_file_bytes = io.BytesIO(dxy_content)
    if dxy_filename.lower().endswith('.xlsx'): dxy_df = pd.read_excel(dxy_file_bytes, engine='openpyxl')
    else: dxy_df = pd.read_csv(dxy_file_bytes)
    if pd.api.types.is_numeric_dtype(dxy_df['observation_date']): dxy_df['DATE'] = pd.to_datetime(dxy_df['observation_date'], unit='D', origin='1899-12-30')
    else: dxy_df['DATE'] = pd.to_datetime(dxy_df['observation_date'])
    dxy_df = dxy_df.set_index('DATE')[['DTWEXBGS']].rename(columns={'DTWEXBGS': 'dxy'})
    dxy_df['dxy'] = pd.to_numeric(dxy_df['dxy'], errors='coerce')
    dxy_df.sort_index(inplace=True)

    treasury_file_bytes = io.BytesIO(treasury_content)
    if treasury_filename.lower().endswith('.xlsx'): treasury_df = pd.read_excel(treasury_file_bytes, engine='openpyxl')
    else: treasury_df = pd.read_csv(treasury_file_bytes)
    treasury_df = treasury_df.rename(columns={'날짜': 'DATE', '종가': 'yield'})
    treasury_df = treasury_df.set_index(pd.to_datetime(treasury_df['DATE']))[['yield']]
    treasury_df['yield'] = pd.to_numeric(treasury_df['yield'], errors='coerce')
    treasury_df.sort_index(inplace=True)

    df = pd.merge(nasdaq_df, dxy_df, left_index=True, right_index=True, how='inner')
    df = pd.merge(df, treasury_df, left_index=True, right_index=True, how='inner')
    initial_rows = len(df); df.dropna(inplace=True)
    print(f"[전처리] 총 {initial_rows - len(df)}개의 결측치 행 제거 후 최종 데이터 개수: {len(df)}")

    df['return_open'] = df['open'].pct_change().fillna(0)
    df['return_volume'] = df['volume'].pct_change(fill_method=None).fillna(0)
    df['return_dxy'] = df['dxy'].pct_change().fillna(0)
    df['return_yield'] = df['yield'].diff().fillna(0)

    features = pd.DataFrame(index=df.index)
    features['return_open_norm'] = (df['return_open'] - df['return_open'].mean()) / (df['return_open'].std() + 1e-9)
    features['return_volume_norm'] = (df['return_volume'] - df['return_volume'].mean()) / (df['return_volume'].std() + 1e-9)
    features['change_norm'] = (df['change'] - df['change'].mean()) / (df['change'].std() + 1e-9)
    features['return_dxy_norm'] = (df['return_dxy'] - df['return_dxy'].mean()) / (df['return_dxy'].std() + 1e-9)
    features['return_yield_norm'] = (df['return_yield'] - df['return_yield'].mean()) / (df['return_yield'].std() + 1e-9)
    return df, features.values


# (ScoreNetwork, train_score_network, simulate_one_step 등 학습·시뮬레이션 로직은
#  src/sde_model.py와 동일한 구조입니다. 차이는 FEATURE_DIM=5와 달러인덱스
#  전처리 코드뿐이라 생략합니다.)
