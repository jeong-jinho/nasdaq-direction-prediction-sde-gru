"""
SDE (확률미분방정식 기반 점수추정) 모델 — 최종 버전

입력 4차원: 시가 수익률, 거래량 변화율, 변동률, 국채금리 변화분
(달러인덱스는 별도 실험(experiments/sde_v1_with_dxy.py)에서 성능 개선이
거의 없는 것을 확인하고 연산 효율성을 위해 최종 파이프라인에서 제외했습니다.)

Google Colab 환경에서 실행하도록 작성되었습니다 (google.colab.files.upload 사용).
로컬에서 실행하려면 run_analysis()의 파일 업로드 부분을 직접 경로 지정으로 바꿔주세요.
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


# --- 모델 및 분석 환경 설정 ---
class Config:
    TEST_DAYS = 30
    VAL_RATIO = 0.1
    FEATURE_DIM = 4  # 나스닥(3) + 국채금리(1)
    HIDDEN_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 1000
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP_VAL = 1.0
    PATIENCE = 100
    SIMULATION_PATHS = 100
    RETURN_CLIP_MIN = -0.07
    RETURN_CLIP_MAX = 0.07


plt.rc('font', family='NanumBarunGothic')
plt.rcParams['axes.unicode_minus'] = False


def convert_volume(value):
    if isinstance(value, str):
        value = value.replace(',', '').strip().upper()
        if value == '-' or not value:
            return np.nan
        if value.endswith('K'):
            return float(value[:-1]) * 1_000
        if value.endswith('M'):
            return float(value[:-1]) * 1_000_000
        if value.endswith('B'):
            return float(value[:-1]) * 1_000_000_000
        try:
            return float(value)
        except ValueError:
            return np.nan
    elif isinstance(value, (int, float)):
        return float(value)
    return np.nan


def convert_change_percent(value):
    if isinstance(value, str):
        value = value.replace('%', '').strip()
        if value == '-' or not value:
            return np.nan
        try:
            return float(value) / 100.0
        except ValueError:
            return np.nan
    elif isinstance(value, (int, float)):
        return float(value) / 100.0
    return np.nan


def get_preprocessed_data(nasdaq_content, treasury_content, treasury_filename):
    # 1. 나스닥 데이터 처리
    nasdaq_stream = io.StringIO(nasdaq_content.decode('utf-8'))
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

    # 2. 국채 금리 데이터 처리 (열 이름 유연하게 매칭)
    treasury_file_bytes = io.BytesIO(treasury_content)
    if treasury_filename.lower().endswith('.xlsx'):
        treasury_df = pd.read_excel(treasury_file_bytes, engine='openpyxl')
    else:
        treasury_df = pd.read_csv(treasury_file_bytes)

    date_col_map = ['날짜', 'DATE', 'observation_date']
    yield_col_map = ['종가', 'Close', 'yield']
    date_col = next((col for col in date_col_map if col in treasury_df.columns), None)
    yield_col = next((col for col in yield_col_map if col in treasury_df.columns), None)
    if not date_col or not yield_col:
        raise KeyError("국채 금리 파일에 날짜('날짜' 등)와 가격('종가' 등) 열이 필요합니다.")

    treasury_df = treasury_df.rename(columns={date_col: 'DATE', yield_col: 'yield'})
    treasury_df = treasury_df.set_index(pd.to_datetime(treasury_df['DATE']))[['yield']]
    treasury_df['yield'] = pd.to_numeric(treasury_df['yield'], errors='coerce')
    treasury_df.sort_index(inplace=True)

    # 3. 데이터 병합
    df = pd.merge(nasdaq_df, treasury_df, left_index=True, right_index=True, how='inner')
    print(f"[전처리] 1. 두 데이터 병합 후 전체 행 개수: {len(df)}")
    initial_rows = len(df)
    df.dropna(inplace=True)
    print(f"[전처리] 2. 총 {initial_rows - len(df)}개의 결측치 행 제거 후 최종 데이터 개수: {len(df)}")

    # 4. 4차원 피처 생성
    df['return_open'] = df['open'].pct_change().fillna(0)
    df['return_volume'] = df['volume'].pct_change(fill_method=None).fillna(0)
    df['return_yield'] = df['yield'].diff().fillna(0)

    features = pd.DataFrame(index=df.index)
    features['return_open_norm'] = (df['return_open'] - df['return_open'].mean()) / (df['return_open'].std() + 1e-9)
    features['return_volume_norm'] = (df['return_volume'] - df['return_volume'].mean()) / (df['return_volume'].std() + 1e-9)
    features['change_norm'] = (df['change'] - df['change'].mean()) / (df['change'].std() + 1e-9)
    features['return_yield_norm'] = (df['return_yield'] - df['return_yield'].mean()) / (df['return_yield'].std() + 1e-9)
    return df, features.values


# --- 점수추정(Score-based) 네트워크 및 확산/역확산 로직 ---
class ScoreNetwork(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d, h), nn.Softplus(),
            nn.Linear(h, h), nn.Softplus(),
            nn.Linear(h, d)
        )

    def forward(self, x):
        return self.network(x)


def beta(t, b=0.1):
    t = torch.tensor(t, dtype=torch.float) if not isinstance(t, torch.Tensor) else t
    return b * (1 - t + 1e-9) ** -1


def tau(t, b=0.1):
    t = torch.tensor(t, dtype=torch.float) if not isinstance(t, torch.Tensor) else t
    return -b * torch.log(1 - t + 1e-9)


def _calculate_loss(model, X_data, t_points, d, D):
    nodes, weights = roots_hermite(D)
    nodes, weights = torch.from_numpy(nodes).float(), torch.from_numpy(weights).float()
    z_components = [nodes] * d
    z = torch.cartesian_prod(*z_components)
    w = torch.prod(torch.cartesian_prod(*[weights] * d), dim=1)
    pre_factor, total_loss = 1.0 / (np.pi ** (d / 2)), 0.0
    with torch.set_grad_enabled(model.training):
        for t in t_points:
            tau_t, C_t = tau(t), 1 - torch.exp(-tau(t))
            mu_t = X_data * torch.exp(-0.5 * tau_t)
            x = torch.sqrt(C_t) * z.unsqueeze(0) + mu_t.unsqueeze(1)
            K_x = model(x.reshape(-1, d)).reshape(X_data.shape[0], -1, d)
            term = K_x + (x - mu_t.unsqueeze(1))
            total_loss += (pre_factor * (w.unsqueeze(0) * (term ** 2).sum(-1)).sum(1)).mean()
    return total_loss * (t_points[1] - t_points[0]) / 2.0


def train_score_network(train_features, val_features, config, d, D=4, num_t=10):
    model = ScoreNetwork(d, config.HIDDEN_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    X_train, X_val = torch.from_numpy(train_features).float(), torch.from_numpy(val_features).float()
    t_points = torch.linspace(0.01, 0.99, num_t)
    best_val_loss, epochs_no_improve, best_model_state = float('inf'), 0, None
    print(f"조기 종료 Patience: {config.PATIENCE} epochs")
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = _calculate_loss(model, X_train, t_points, d, D)
        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VAL)
        optimizer.step()
        model.eval()
        val_loss = _calculate_loss(model, X_val, t_points, d, D)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
        if val_loss < best_val_loss:
            best_val_loss, epochs_no_improve, best_model_state = val_loss, 0, copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= config.PATIENCE:
            print(f"\n조기 종료: 최적 Epoch: {epoch - config.PATIENCE + 1}")
            break
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model


def generate_synthetic_features(model, xi, d, K=100, b=0.1):
    delta = 1.0 / K
    X = xi.copy()
    for j in range(K):
        t = j * delta
        bet = beta(t, b).item()
        X += -0.5 * bet * X * delta + np.sqrt(bet) * np.random.randn(d) * np.sqrt(delta)
    for j in range(K, 0, -1):
        t = (j - 1) * delta
        bet = beta(t, b).item()
        C_t = (1 - np.exp(-tau(t, b))).item()
        s_val = model(torch.from_numpy(X).float().unsqueeze(0)).detach().squeeze(0).numpy() / C_t if C_t > 1e-9 else np.zeros(d)
        X += (-0.5 * bet * X - bet * s_val) * delta + np.sqrt(bet) * np.random.randn(d) * np.sqrt(delta)
    return X


def simulate_one_step(model, train_df, train_features, y0, config):
    recent_returns = train_df['open'].pct_change().dropna()
    mean_recent, std_recent = recent_returns[-120:].mean(), recent_returns[-120:].std()
    next_day_prices = []
    for _ in range(config.SIMULATION_PATHS):
        idx = np.random.randint(0, len(train_features))
        generated_features = generate_synthetic_features(model, train_features[idx], config.FEATURE_DIM)
        return_feature = generated_features[0]
        r_adjusted_open = np.clip(return_feature * std_recent + mean_recent, config.RETURN_CLIP_MIN, config.RETURN_CLIP_MAX)
        next_day_prices.append(y0 * (1 + r_adjusted_open))
    return np.array(next_day_prices)


def analyze_classification_from_prices(next_day_prices, y0):
    total_paths = len(next_day_prices)
    up_count = np.sum(next_day_prices > y0)
    up_probability = up_count / total_paths if total_paths > 0 else 0
    prediction = "상승" if up_probability >= 0.5 else "하락"
    return {'pred_direction': prediction, 'up_prob': up_probability}


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
        df, all_features = get_preprocessed_data(uploaded_nasdaq[nasdaq_filename], uploaded_treasury[treasury_filename], treasury_filename)
    except Exception as e:
        print(f"데이터 처리 실패: {e}")
        return

    if len(df) < config.TEST_DAYS + 120:
        print("최종 데이터가 너무 적어 백테스팅을 수행할 수 없습니다.")
        return

    test_split = len(df) - config.TEST_DAYS
    train_val_df, test_df = df.iloc[:test_split], df.iloc[test_split:]
    val_split = int(len(train_val_df) * (1 - config.VAL_RATIO))
    train_df, val_df = train_val_df.iloc[:val_split], train_val_df.iloc[val_split:]
    train_features, val_features = all_features[:val_split], all_features[val_split:test_split]

    print(f"\n최종 데이터 분할:\n  - 훈련: {len(train_df)}일\n  - 검증: {len(val_df)}일\n  - 테스트: {len(test_df)}일")
    print(f"모델이 학습할 피처의 수(차원): {train_features.shape[1]}")

    print("\n--- 모델 학습 시작 ---")
    model = train_score_network(train_features, val_features, config, d=config.FEATURE_DIM)

    print(f"\n--- 테스트 기간({config.TEST_DAYS}일) 동안 일자별 등락 예측 시작 ---")
    predictions = []
    full_data_for_sim = train_val_df.copy()
    full_features_for_sim = all_features[:len(train_val_df)]

    for i in tqdm(range(len(test_df))):
        y0 = full_data_for_sim['open'].values[-1]
        next_day_prices = simulate_one_step(model, full_data_for_sim, full_features_for_sim, y0, config)
        result = analyze_classification_from_prices(next_day_prices, y0)
        predictions.append(result)

        next_day_data = test_df.iloc[[i]]
        full_data_for_sim = pd.concat([full_data_for_sim, next_day_data])
        full_features_for_sim = all_features[:len(full_data_for_sim)]

    results_df = pd.DataFrame(index=test_df.index)
    results_df['open'] = test_df['open']
    results_df['actual_change'] = np.where(test_df['open'].diff() > 0, "상승", "하락")
    results_df['predicted_direction'] = [p['pred_direction'] for p in predictions]
    results_df['predicted_up_prob'] = [f"{p['up_prob']:.2%}" for p in predictions]

    print("\n--- 일자별 실제 vs 예측 등락 비교 ---")
    print(results_df.to_string())
    correct_predictions = (results_df['actual_change'] == results_df['predicted_direction']).sum()
    accuracy = correct_predictions / len(results_df)
    print(f"\n최종 예측 정확도: {accuracy:.2%}")


if __name__ == "__main__":
    run_analysis()
