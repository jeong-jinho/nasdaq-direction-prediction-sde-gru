"""
[실험 기록 — 데이터 누수 있던 초기 버전]

이 파일은 README/기술보고서에서 설명한 "GRU 86.67% 비정상 정확도" 문제가
발생했던 실제 코드입니다. 정규화 통계량을 훈련 구간과 테스트 구간을 분리하지 않고
전체 데이터(get_preprocessed_data 내부)로 미리 계산해버려서, 테스트 구간 정보가
간접적으로 학습에 유입되는 데이터 누수가 있었습니다.

→ 수정된 최종 버전은 src/gru_model.py를 참고하세요.
이 파일은 실행용이 아니라 트러블슈팅 과정을 보여주기 위한 기록용입니다.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm

from google.colab import files
import io

# --- 모델 및 분석 환경 설정 ---
class Config:
    TEST_DAYS = 30
    VAL_RATIO = 0.1
    FEATURE_DIM = 5  # [문제] 나스닥(3) + 달러인덱스(1) + 국채금리(1)
    SEQUENCE_LENGTH = 20
    HIDDEN_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 300
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP_VAL = 1.0
    PATIENCE = 50
    RETURN_CLIP_MIN = -0.07
    RETURN_CLIP_MAX = 0.07

plt.rc('font', family='NanumBarunGothic')
plt.rcParams['axes.unicode_minus'] = False


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
    try:
        nasdaq_stream = io.StringIO(nasdaq_content.decode('utf-8'))
    except UnicodeDecodeError:
        try:
            nasdaq_stream = io.StringIO(nasdaq_content.decode('cp949'))
        except UnicodeDecodeError:
            nasdaq_stream = io.StringIO(nasdaq_content.decode('euc-kr'))

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
    df.dropna(inplace=True)

    df['return_open'] = df['open'].pct_change().fillna(0)
    df['return_volume'] = df['volume'].pct_change(fill_method=None).fillna(0)
    df['return_dxy'] = df['dxy'].pct_change().fillna(0)
    df['return_yield'] = df['yield'].diff().fillna(0)

    # [문제 지점] 정규화를 전체 데이터 기준으로 미리 계산 → 테스트 구간 정보가 학습에 유입됨 (데이터 누수)
    df['target_return_norm'] = (df['return_open'] - df['return_open'].mean()) / (df['return_open'].std() + 1e-9)
    features = pd.DataFrame(index=df.index)
    features['return_open_norm'] = (df['return_open'] - df['return_open'].mean()) / (df['return_open'].std() + 1e-9)
    features['return_volume_norm'] = (df['return_volume'] - df['return_volume'].mean()) / (df['return_volume'].std() + 1e-9)
    features['change_norm'] = (df['change'] - df['change'].mean()) / (df['change'].std() + 1e-9)
    features['return_dxy_norm'] = (df['return_dxy'] - df['return_dxy'].mean()) / (df['return_dxy'].std() + 1e-9)
    features['return_yield_norm'] = (df['return_yield'] - df['return_yield'].mean()) / (df['return_yield'].std() + 1e-9)

    mean_std = {'mean': df['return_open'].mean(), 'std': df['return_open'].std()}
    return df, features.values, df['target_return_norm'].values, mean_std


def create_sequences(features, targets, seq_length):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])
        y.append(targets[i + seq_length])
    return np.array(X), np.array(y)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)


# (학습/예측 로직은 src/gru_model.py와 동일한 구조입니다. 문제의 핵심은
#  위 get_preprocessed_data()의 정규화 시점이며, 나머지 학습 루프는 생략합니다.)
