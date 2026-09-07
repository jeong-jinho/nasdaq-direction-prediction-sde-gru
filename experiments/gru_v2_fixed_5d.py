"""
[실험 기록 — 누수 수정 + 5차원(달러인덱스 포함) 중간 버전]

experiments/gru_v1_leakage.py에서 발견된 데이터 누수는 수정했지만
(정규화 통계량을 훈련 구간만으로 계산), 이 시점에는 아직 달러인덱스를
피처에 포함한 5차원 상태였습니다.

이후 달러인덱스를 추가해도 성능 개선이 거의 없다는 것을 확인하고
연산 효율성을 위해 제외해, 최종적으로 src/gru_model.py의 4차원 버전이 되었습니다.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm

from google.colab import files
import io


class Config:
    TEST_DAYS = 30
    VAL_RATIO = 0.1
    FEATURE_DIM = 5  # 나스닥(3) + 달러인덱스(1) + 국채금리(1)
    SEQUENCE_LENGTH = 20
    HIDDEN_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 300
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP_VAL = 1.0
    PATIENCE = 50
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
    try: nasdaq_stream = io.StringIO(nasdaq_content.decode('utf-8'))
    except UnicodeDecodeError:
        try: nasdaq_stream = io.StringIO(nasdaq_content.decode('cp949'))
        except UnicodeDecodeError: nasdaq_stream = io.StringIO(nasdaq_content.decode('euc-kr'))

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
    else:
        try: dxy_df = pd.read_csv(dxy_file_bytes, encoding='utf-8')
        except Exception: dxy_file_bytes.seek(0); dxy_df = pd.read_csv(dxy_file_bytes, encoding='cp949')
    if pd.api.types.is_numeric_dtype(dxy_df['observation_date']): dxy_df['DATE'] = pd.to_datetime(dxy_df['observation_date'], unit='D', origin='1899-12-30')
    else: dxy_df['DATE'] = pd.to_datetime(dxy_df['observation_date'])
    dxy_df = dxy_df.set_index('DATE')[['DTWEXBGS']].rename(columns={'DTWEXBGS': 'dxy'})
    dxy_df['dxy'] = pd.to_numeric(dxy_df['dxy'], errors='coerce')
    dxy_df.sort_index(inplace=True)

    treasury_file_bytes = io.BytesIO(treasury_content)
    if treasury_filename.lower().endswith('.xlsx'): treasury_df = pd.read_excel(treasury_file_bytes, engine='openpyxl')
    else:
        try: treasury_df = pd.read_csv(treasury_file_bytes, encoding='utf-8')
        except Exception: treasury_file_bytes.seek(0); treasury_df = pd.read_csv(treasury_file_bytes, encoding='cp949')
    treasury_df = treasury_df.rename(columns={'날짜': 'DATE', '종가': 'yield'})
    try: treasury_df['DATE'] = pd.to_datetime(treasury_df['DATE'])
    except ValueError: pass
    treasury_df = treasury_df.set_index(treasury_df['DATE'])[['yield']]
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

    feature_cols = ['return_open', 'return_volume', 'change', 'return_dxy', 'return_yield']
    raw_features = df[feature_cols].values
    raw_target = df['return_open'].values
    return df, raw_features, raw_target


def _normalize_data(raw_features, raw_target, train_split_idx):
    """훈련 구간 통계량으로만 정규화 (누수 수정 로직)"""
    train_features_raw = raw_features[:train_split_idx]
    train_target_raw = raw_target[:train_split_idx]
    feature_means = train_features_raw.mean(axis=0)
    feature_stds = train_features_raw.std(axis=0) + 1e-9
    target_mean = train_target_raw.mean()
    target_std = train_target_raw.std() + 1e-9
    all_features_norm = (raw_features - feature_means) / feature_stds
    all_targets_norm = (raw_target - target_mean) / target_std
    mean_std = {'mean': target_mean, 'std': target_std}
    return all_features_norm, all_targets_norm, mean_std


# (GRUModel, train_gru_network, predict_next_return, create_sequences 등
#  나머지 학습·예측 로직은 src/gru_model.py와 거의 동일한 구조라 생략합니다.
#  차이는 오직 FEATURE_DIM=5와 달러인덱스 관련 전처리 코드뿐입니다.)
