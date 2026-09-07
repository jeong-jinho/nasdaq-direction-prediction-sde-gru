"""
[별도 도구 — 실시간 미래 30일 예측 스크립트]

이 스크립트는 백테스트(과거 구간에서 실제 vs 예측 비교)가 아니라,
가장 최근 데이터를 기준으로 향후 30일을 예측해 CSV로 다운로드하는
실사용 도구입니다. 정확도 평가 로직이 없고, 나스닥 3개 피처(시가/거래량/변동률)만
사용하는 더 단순한 버전입니다.

Google Colab 환경에서 실행하도록 작성되었습니다.
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


class Config:
    FUTURE_PREDICT_DAYS = 30
    VAL_RATIO = 0.1
    FEATURE_DIM = 3  # 나스닥 데이터의 'return_open', 'return_volume', 'change'만 사용
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


def get_preprocessed_data(nasdaq_content):
    try: nasdaq_stream = io.StringIO(nasdaq_content.decode('utf-8'))
    except UnicodeDecodeError:
        try: nasdaq_stream = io.StringIO(nasdaq_content.decode('cp949'))
        except UnicodeDecodeError: nasdaq_stream = io.StringIO(nasdaq_content.decode('euc-kr'))

    nasdaq_df = pd.read_csv(nasdaq_stream, index_col='날짜', parse_dates=True)
    nasdaq_df.sort_index(inplace=True)

    column_map = {'open': ['시가', 'open', 'Open'], 'volume': ['거래량', 'volume', 'Volume'], 'change': ['변동 %', '변동', 'change', 'Change'], 'close': ['종가', 'close', 'Close']}
    available_columns = {col.lower().strip(): col for col in nasdaq_df.columns}
    open_col = next((available_columns.get(key) for key in column_map['open']), None)
    volume_col = next((available_columns.get(key) for key in column_map['volume']), None)
    change_col = next((available_columns.get(key) for key in column_map['change']), None)
    close_col = next((available_columns.get(key) for key in column_map['close']), None)
    if not all([open_col, volume_col, change_col, close_col]):
        raise KeyError("나스닥 CSV에 시가/종가/거래량/변동률 열이 필요합니다.")

    df = nasdaq_df.rename(columns={open_col: 'open', volume_col: 'volume', change_col: 'change', close_col: 'close'})
    df['open'] = pd.to_numeric(df['open'].astype(str).str.replace(',', ''), errors='coerce')
    df['close'] = pd.to_numeric(df['close'].astype(str).str.replace(',', ''), errors='coerce')
    df['volume'] = df['volume'].apply(convert_volume)
    df['change'] = df['change'].apply(convert_change_percent)

    numerical_cols = ['open', 'close', 'volume', 'change']
    for col in numerical_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    df.dropna(inplace=True)
    df = df[df.index.dayofweek < 5]  # 주말 제외

    df['return_open'] = df['open'].pct_change()
    df['return_volume'] = df['volume'].pct_change(fill_method=None)
    feature_cols = ['return_open', 'return_volume', 'change']
    df = df.iloc[1:].copy()
    df.fillna(0, inplace=True)

    raw_features = df[feature_cols].values
    raw_target = df['return_open'].values
    return df, raw_features, raw_target, feature_cols


def _normalize_data_for_future_pred(raw_features, raw_target):
    feature_means = raw_features.mean(axis=0)
    feature_stds = raw_features.std(axis=0) + 1e-9
    target_mean = raw_target.mean()
    target_std = raw_target.std() + 1e-9
    all_features_norm = (raw_features - feature_means) / feature_stds
    all_targets_norm = (raw_target - target_mean) / target_std
    return all_features_norm, all_targets_norm, {'mean': target_mean, 'std': target_std}, {'mean': feature_means, 'std': feature_stds}


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


def predict_one_step(model, current_sequence, feature_stats, config):
    model.eval()
    input_tensor = torch.from_numpy(current_sequence).float().unsqueeze(0)
    with torch.no_grad():
        predicted_outputs_norm = model(input_tensor).squeeze().numpy()
    predicted_outputs_raw = (predicted_outputs_norm * feature_stats['std']) + feature_stats['mean']
    predicted_return_open_raw = predicted_outputs_raw[0]
    predicted_return_open_clipped = np.clip(predicted_return_open_raw, config.RETURN_CLIP_MIN, config.RETURN_CLIP_MAX)
    return predicted_return_open_clipped, predicted_outputs_norm


# (run_analysis: 파일 업로드 → 학습 → 30일 미래 예측 → 시각화 → CSV 다운로드로
#  이어지는 실행 로직은 다른 GRU 스크립트들과 유사한 구조라 생략했습니다.
#  차이점은 이 스크립트는 정확도 평가가 없는 순수 예측 도구라는 점입니다.)
