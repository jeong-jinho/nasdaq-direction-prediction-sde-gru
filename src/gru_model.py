"""
GRU (순환신경망) 모델 — 최종 버전

입력 4차원: 시가 수익률, 거래량 변화율, 변동률, 국채금리 변화분
(달러인덱스는 experiments/gru_v2_fixed_5d.py에서 실험했으나 성능 개선이
거의 없어 최종 파이프라인에서는 제외했습니다.)

핵심 포인트 — 데이터 누수 방지:
정규화 통계량(평균/표준편차)을 "테스트 구간 이전의 훈련 데이터"만으로 계산합니다.
초기 버전(experiments/gru_v1_leakage.py)에서는 전체 데이터 기준으로 정규화해
테스트 정보가 간접적으로 학습에 유입되는 문제가 있었고, 이를 수정한 것이 이 파일입니다.

Google Colab 환경에서 실행하도록 작성되었습니다 (google.colab.files.upload 사용).
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

from preprocessing import convert_volume, convert_change_percent


# --- 모델 및 분석 환경 설정 ---
class Config:
    TEST_DAYS = 30
    VAL_RATIO = 0.1
    FEATURE_DIM = 4  # 나스닥(3) + 국채금리(1)
    SEQUENCE_LENGTH = 20
    HIDDEN_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 300
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP_VAL = 1.0
    PATIENCE = 50
    RETURN_CLIP_MIN = -0.07
    RETURN_CLIP_MAX = 0.07


def get_preprocessed_data(nasdaq_content, treasury_content, treasury_filename):
    # 1. 나스닥 데이터 처리 (인코딩 자동 대응)
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
    if not all([open_col, volume_col, change_col]):
        raise KeyError("나스닥 CSV에 '시가', '거래량', '변동' 또는 '변동 %' 열이 필요합니다.")
    nasdaq_df = nasdaq_df.rename(columns={open_col: 'open', volume_col: 'volume', change_col: 'change'})
    nasdaq_df['open'] = pd.to_numeric(nasdaq_df['open'].astype(str).str.replace(',', ''), errors='coerce')
    nasdaq_df['volume'] = nasdaq_df['volume'].apply(convert_volume)
    nasdaq_df['change'] = nasdaq_df['change'].apply(convert_change_percent)

    # 2. 국채 금리 데이터 처리
    treasury_file_bytes = io.BytesIO(treasury_content)
    if treasury_filename.lower().endswith('.xlsx'):
        treasury_df = pd.read_excel(treasury_file_bytes, engine='openpyxl')
    else:
        try:
            treasury_df = pd.read_csv(treasury_file_bytes, encoding='utf-8')
        except Exception:
            treasury_file_bytes.seek(0)
            treasury_df = pd.read_csv(treasury_file_bytes, encoding='cp949')

    treasury_df = treasury_df.rename(columns={'날짜': 'DATE', '종가': 'yield'})
    try:
        treasury_df['DATE'] = pd.to_datetime(treasury_df['DATE'])
    except ValueError:
        pass
    treasury_df = treasury_df.set_index(treasury_df['DATE'])[['yield']]
    treasury_df['yield'] = pd.to_numeric(treasury_df['yield'], errors='coerce')
    treasury_df.sort_index(inplace=True)

    # 3. 데이터 병합 (나스닥 + 국채금리만, 달러인덱스 제외)
    df = pd.merge(nasdaq_df, treasury_df, left_index=True, right_index=True, how='inner')
    print(f"[전처리] 1. 데이터 병합 후 전체 행 개수: {len(df)}")
    initial_rows = len(df)
    df.dropna(inplace=True)
    print(f"[전처리] 2. 총 {initial_rows - len(df)}개의 결측치 행 제거 후 최종 데이터 개수: {len(df)}")

    # 4. Raw 4차원 피처 및 타겟 수익률 생성 (정규화는 run_analysis에서 훈련 구간 기준으로 수행 → 데이터 누수 방지)
    df['return_open'] = df['open'].pct_change().fillna(0)
    df['return_volume'] = df['volume'].pct_change(fill_method=None).fillna(0)
    df['return_yield'] = df['yield'].diff().fillna(0)

    feature_cols = ['return_open', 'return_volume', 'change', 'return_yield']
    raw_features = df[feature_cols].values
    raw_target = df['return_open'].values

    return df, raw_features, raw_target


def _normalize_data(raw_features, raw_target, train_split_idx):
    """훈련 구간(train_split_idx 이전)의 통계량으로만 정규화합니다. (데이터 누수 방지 핵심 로직)"""
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


def create_sequences(features, targets, seq_length):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])
        y.append(targets[i + seq_length])
    return np.array(X), np.array(y)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def train_gru_network(X_train, y_train, X_val, y_val, config):
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float().unsqueeze(1))
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float().unsqueeze(1))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = GRUModel(input_size=config.FEATURE_DIM, hidden_size=config.HIDDEN_SIZE, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    best_val_loss, epochs_no_improve, best_model_state = float('inf'), 0, None
    print(f"조기 종료 Patience: {config.PATIENCE} epochs")

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VAL)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss, epochs_no_improve, best_model_state = val_loss, 0, copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.PATIENCE:
            print(f"\n조기 종료: 최적 Epoch: {epoch - config.PATIENCE + 1}, 최적 Val Loss: {best_val_loss:.6f}")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model


def predict_next_return(model, last_features, mean_std, config):
    model.eval()
    input_tensor = torch.from_numpy(last_features).float().unsqueeze(0)
    with torch.no_grad():
        predicted_return_norm = model(input_tensor).squeeze().item()
    mean_return = mean_std['mean']
    std_return = mean_std['std']
    predicted_return = (predicted_return_norm * std_return) + mean_return
    predicted_return = np.clip(predicted_return, config.RETURN_CLIP_MIN, config.RETURN_CLIP_MAX)
    prediction = "상승" if predicted_return > 0 else "하락"
    return {'pred_direction': prediction, 'predicted_return': predicted_return}


def run_analysis():
    config = Config()

    print("1. 나스닥 CSV 파일을 업로드해주세요.")
    uploaded_nasdaq = files.upload()
    if not uploaded_nasdaq:
        print("나스닥 파일이 업로드되지 않았습니다.")
        return

    print("\n2. 미국 국채 금리 CSV 또는 XLSX 파일을 업로드해주세요.")
    uploaded_treasury = files.upload()
    if not uploaded_treasury:
        print("국채 금리 파일이 업로드되지 않았습니다.")
        return

    nasdaq_filename = list(uploaded_nasdaq.keys())[0]
    treasury_filename = list(uploaded_treasury.keys())[0]
    print(f"\n파일 업로드 완료: '{nasdaq_filename}', '{treasury_filename}'")

    try:
        df, raw_features, raw_target = get_preprocessed_data(
            uploaded_nasdaq[nasdaq_filename],
            uploaded_treasury[treasury_filename],
            treasury_filename
        )
    except Exception as e:
        print(f"데이터 처리 실패: {e}")
        return

    if len(df) < config.TEST_DAYS + config.SEQUENCE_LENGTH + 1:
        print(f"최종 데이터가 너무 적어 ({len(df)}일) 백테스팅을 수행할 수 없습니다.")
        return

    # 훈련 구간 기준 통계량으로 전체 데이터 정규화 (데이터 누수 방지)
    test_split_idx_raw = len(df) - config.TEST_DAYS
    all_features_norm, all_targets_norm, mean_std = _normalize_data(raw_features, raw_target, test_split_idx_raw)

    X_all, y_all = create_sequences(all_features_norm, all_targets_norm, config.SEQUENCE_LENGTH)

    df_seq = df.iloc[config.SEQUENCE_LENGTH:]
    test_split_idx_seq = len(df_seq) - config.TEST_DAYS

    X_train_val, X_test = X_all[:test_split_idx_seq], X_all[test_split_idx_seq:]
    y_train_val, y_test = y_all[:test_split_idx_seq], y_all[test_split_idx_seq:]

    val_split = int(len(X_train_val) * (1 - config.VAL_RATIO))
    X_train, X_val = X_train_val[:val_split], X_train_val[val_split:]
    y_train, y_val = y_train_val[:val_split], y_train_val[val_split:]

    print(f"\n최종 시퀀스 데이터 분할:\n  - 훈련: {len(X_train)}개\n  - 검증: {len(X_val)}개\n  - 테스트: {len(X_test)}개")
    print(f"모델이 학습할 피처의 수(차원): {config.FEATURE_DIM}")

    print("\n--- GRU 모델 학습 시작 ---")
    model = train_gru_network(X_train, y_train, X_val, y_val, config)

    print(f"\n--- 테스트 기간({config.TEST_DAYS}일) 동안 일자별 등락 예측 시작 ---")
    predictions = []
    for i in tqdm(range(len(X_test))):
        last_features = X_test[i]
        result = predict_next_return(model, last_features, mean_std, config)
        predictions.append(result)

    results_df = pd.DataFrame(index=df_seq.iloc[test_split_idx_seq:].index)
    results_df['actual_open'] = df_seq.iloc[test_split_idx_seq:]['open']
    actual_returns = df_seq['return_open'].iloc[test_split_idx_seq:].values
    results_df['actual_change'] = np.where(actual_returns > 0, "상승", "하락")
    results_df['predicted_direction'] = [p['pred_direction'] for p in predictions]
    results_df['predicted_return'] = [f"{p['predicted_return']:.4f}" for p in predictions]

    print("\n--- 일자별 실제 vs 예측 등락 비교 ---")
    print(results_df.to_string())
    correct_predictions = (results_df['actual_change'] == results_df['predicted_direction']).sum()
    accuracy = correct_predictions / len(results_df)
    print(f"\n최종 예측 정확도: {accuracy:.2%}")


if __name__ == "__main__":
    run_analysis()
