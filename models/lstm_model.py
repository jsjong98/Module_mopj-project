import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import optuna # optimize_hyperparameters_semimonthly_kfold에서 사용
import logging
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # prepare_data에서 사용

from app.utils.file_utils import set_seed
from app.prediction.metrics import calculate_f1_score, calculate_direction_accuracy, calculate_mape, calculate_direction_weighted_score
from app.models.loss_functions import DirectionalLoss
from app.data.preprocessor import prepare_data
from app.data.loader import load_data
from app.data.cache_manager import get_file_cache_dirs, find_compatible_hyperparameters
from app.core.gpu_manager import log_device_usage, check_gpu_availability
from app.core import DEFAULT_DEVICE, CUDA_AVAILABLE
from app.utils.date_utils import format_date, get_semimonthly_period
from app.core.state_manager import prediction_state

import traceback
import json
import os

DEFAULT_DEVICE, CUDA_AVAILABLE = check_gpu_availability()

# 데이터 로더의 워커 시드 고정을 위한 함수
def seed_worker(worker_id): # 여기서 seed_worker 정의
    set_seed(np.random.get_seed_value() + worker_id) # numpy seed를 사용하여 다양성 확보
g = torch.Generator() # DataLoader의 generator는 여기에 정의
SEED = 42
g.manual_seed(SEED)

logger = logging.getLogger(__name__)

# 개선된 LSTM 예측 모델
class ImprovedLSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2, output_size=23):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # hidden_size를 8의 배수로 조정
        self.adjusted_hidden = (hidden_size // 8) * 8
        if self.adjusted_hidden < 32:
            self.adjusted_hidden = 32
        
        # LSTM dropout 설정
        self.lstm_dropout = 0.0 if num_layers == 1 else dropout
        
        # 계층적 LSTM 구조
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=input_size if i == 0 else self.adjusted_hidden,
                hidden_size=self.adjusted_hidden,
                num_layers=1,
                batch_first=True
            ) for i in range(num_layers)
        ])
        
        # 듀얼 어텐션 메커니즘
        self.temporal_attention = nn.MultiheadAttention(
            self.adjusted_hidden,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.feature_attention = nn.MultiheadAttention(
            self.adjusted_hidden,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.adjusted_hidden) for _ in range(num_layers)
        ])
        
        self.final_layer_norm = nn.LayerNorm(self.adjusted_hidden)
        
        # Dropout 레이어
        self.dropout_layer = nn.Dropout(dropout)
        
        # 이전 값 정보를 결합하기 위한 레이어
        self.prev_value_encoder = nn.Sequential(
            nn.Linear(1, self.adjusted_hidden // 4),
            nn.ReLU(),
            nn.Linear(self.adjusted_hidden // 4, self.adjusted_hidden)
        )
        
        # 시계열 특성 추출을 위한 컨볼루션 레이어
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.adjusted_hidden, self.adjusted_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.adjusted_hidden, self.adjusted_hidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 출력 레이어 - 계층적 구조
        self.output_layers = nn.ModuleList([
            nn.Linear(self.adjusted_hidden, self.adjusted_hidden // 2),
            nn.Linear(self.adjusted_hidden // 2, self.adjusted_hidden // 4),
            nn.Linear(self.adjusted_hidden // 4, output_size)
        ])
        
        # 잔차 연결을 위한 프로젝션 레이어
        self.residual_proj = nn.Linear(self.adjusted_hidden, output_size)
        
    def forward(self, x, prev_value=None, return_attention=False):
        batch_size = x.size(0)
        
        # 계층적 LSTM 처리
        lstm_out = x
        skip_connections = []
        
        for i, (lstm, layer_norm) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            lstm_out, _ = lstm(lstm_out)
            lstm_out = layer_norm(lstm_out)
            lstm_out = self.dropout_layer(lstm_out)
            skip_connections.append(lstm_out)
        
        # 시간적 어텐션
        temporal_context, temporal_weights = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        temporal_context = self.dropout_layer(temporal_context)
        
        # 특징 어텐션
        # 특징 차원으로 변환 (B, L, H) -> (B, H, L)
        feature_input = lstm_out.transpose(1, 2)
        feature_input = self.conv_layers(feature_input)
        feature_input = feature_input.transpose(1, 2)
        
        feature_context, feature_weights = self.feature_attention(feature_input, feature_input, feature_input)
        feature_context = self.dropout_layer(feature_context)
        
        # 컨텍스트 결합
        combined_context = temporal_context + feature_context
        for skip in skip_connections:
            combined_context = combined_context + skip
        
        combined_context = self.final_layer_norm(combined_context)
        
        # 이전 값 정보 처리
        if prev_value is not None:
            prev_value = prev_value.unsqueeze(1) if len(prev_value.shape) == 1 else prev_value
            prev_encoded = self.prev_value_encoder(prev_value)
            combined_context = combined_context + prev_encoded.unsqueeze(1)
        
        # 최종 특징 추출 (마지막 시퀀스)
        final_features = combined_context[:, -1, :]
        
        # 계층적 출력 처리
        out = final_features
        residual = self.residual_proj(final_features)
        
        for i, layer in enumerate(self.output_layers):
            out = layer(out)
            if i < len(self.output_layers) - 1:
                out = F.relu(out)
                out = self.dropout_layer(out)
        
        # 잔차 연결 추가
        out = out + residual
        
        if return_attention:
            attention_weights = {
                'temporal_weights': temporal_weights,
                'feature_weights': feature_weights
            }
            return out, attention_weights
        
        return out
        
    def get_attention_maps(self, x, prev_value=None):
        """어텐션 가중치 맵을 반환하는 함수"""
        with torch.no_grad():
            # forward 메서드에 return_attention=True 전달
            _, attention_weights = self.forward(x, prev_value, return_attention=True)
            
            # 어텐션 가중치 평균 계산 (multi-head -> single map)
            temporal_weights = attention_weights['temporal_weights'].mean(dim=1)  # 헤드 평균
            feature_weights = attention_weights['feature_weights'].mean(dim=1)    # 헤드 평균
            
            return {
                'temporal_weights': temporal_weights.cpu().numpy(),
                'feature_weights': feature_weights.cpu().numpy()
            }

# TimeSeriesDataset 및 평가 메트릭스
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, device, prev_values=None):
        if isinstance(X, torch.Tensor):
            self.X = X
            self.y = y
        else:
            self.X = torch.FloatTensor(X).to(device)
            self.y = torch.FloatTensor(y).to(device)
        
        if prev_values is not None:
            if isinstance(prev_values, torch.Tensor):
                self.prev_values = prev_values
            else:
                self.prev_values = torch.FloatTensor(prev_values).to(device)
        else:
            self.prev_values = None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.prev_values is not None:
            return self.X[idx], self.y[idx], self.prev_values[idx]
        return self.X[idx], self.y[idx]

def train_model(features, target_col, current_date, historical_data, device, params):
    """LSTM 모델 학습"""
    try:
        # 일관된 학습 결과를 위한 시드 고정
        set_seed()
        
        # 디바이스 사용 정보 로깅
        log_device_usage(device, "LSTM 모델 학습 시작")
        
        # 특성 이름 확인
        if target_col not in features:
            features.append(target_col)
        
        # 학습 데이터 준비 (현재 날짜까지)
        train_df = historical_data[features].copy()
        target_col_idx = train_df.columns.get_loc(target_col)
        
        # 스케일링
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_df)
        
        # 하이퍼파라미터
        sequence_length = params.get('sequence_length', 20)
        hidden_size = params.get('hidden_size', 128)
        num_layers = params.get('num_layers', 2)
        dropout = params.get('dropout', 0.2)
        learning_rate = params.get('learning_rate', 0.001)
        num_epochs = params.get('num_epochs', 100)
        batch_size = params.get('batch_size', 32)
        alpha = params.get('loss_alpha', 0.7)  # DirectionalLoss alpha
        beta = params.get('loss_beta', 0.2)    # DirectionalLoss beta
        patience = params.get('patience', 20)   # 조기 종료 인내
        predict_window = params.get('predict_window', 23)  # 예측 기간
        
        # 80/20 분할 (연대순)
        train_size = int(len(train_data) * 0.8)
        train_set = train_data[:train_size]
        val_set = train_data[train_size:]
        
        # 시퀀스 데이터 준비
        X_train, y_train, prev_train, X_val, y_val, prev_val = prepare_data(
            train_set, val_set, sequence_length, predict_window, target_col_idx
        )
        
        # 충분한 데이터가 있는지 확인
        if len(X_train) < batch_size:
            batch_size = max(1, len(X_train) // 2)
            logger.warning(f"배치 크기가 데이터 크기보다 커서 조정: {batch_size} (데이터: {len(X_train)})")
        
        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError("Insufficient data for training")
        
        logger.info(f"🎯 사용할 배치 크기: {batch_size}")
        
        # 데이터셋 및 로더 생성 (CPU에서 생성, 학습 시 GPU로 이동)
        train_dataset = TimeSeriesDataset(X_train, y_train, torch.device('cpu'), prev_train)
        
        # GPU 활용률 최적화를 위한 DataLoader 설정
        num_workers = 0 if device.type == 'cuda' else 2  # CUDA에서는 멀티프로세싱 비활성화
        pin_memory = device.type == 'cuda'  # GPU 사용 시 pin_memory 활성화
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False if num_workers == 0 else True
        )
        
        val_dataset = TimeSeriesDataset(X_val, y_val, torch.device('cpu'), prev_val)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=min(batch_size, len(X_val)),  # 검증에서도 배치 크기 최적화
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        logger.info(f"🔧 DataLoader 설정: workers={num_workers}, pin_memory={pin_memory}, train_batch={batch_size}, val_batch={min(batch_size, len(X_val))}")
        
        # 모델 생성
        logger.info("📈 ImprovedLSTMPredictor 사용")
        model = ImprovedLSTMPredictor(
            input_size=train_data.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=predict_window
        ).to(device)
        
        # 모델이 GPU에 올라갔는지 확인
        model_device = next(model.parameters()).device
        logger.info(f"🤖 ImprovedLSTM 모델이 {model_device}에 로드되었습니다")
        log_device_usage(model_device, "모델 로드 완료")
        
        # 손실 함수 생성
        logger.info(f"📈 DirectionalLoss 사용: alpha={alpha}, beta={beta}")
        criterion = DirectionalLoss(alpha=alpha, beta=beta)
        
        # 최적화기 및 스케줄러
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5,
            patience=patience//2
        )
        
        # 학습
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # GPU 최적화 설정
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True  # 입력 크기가 일정할 때 성능 향상
            torch.cuda.empty_cache()  # 캐시 정리
            
        log_device_usage(device, "모델 학습 중")
        
        for epoch in range(num_epochs):
            # 학습 모드
            model.train()
            train_loss = 0
            batch_count = 0
            
            for X_batch, y_batch, prev_batch in train_loader:
                optimizer.zero_grad()
                
                # 모델과 같은 디바이스로 데이터 이동
                X_batch = X_batch.to(device, non_blocking=device.type=='cuda')
                y_batch = y_batch.to(device, non_blocking=device.type=='cuda')
                prev_batch = prev_batch.to(device, non_blocking=device.type=='cuda')
                
                # 모델 예측 및 손실 계산
                y_pred = model(X_batch, prev_batch)
                loss = criterion(y_pred, y_batch, prev_batch)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                batch_count += 1
                
            # 첫 번째 에포크와 주기적으로 GPU 상태 로깅
            if epoch == 0 or (epoch + 1) % 10 == 0:
                log_device_usage(device, f"에포크 {epoch+1}/{num_epochs}")
            
            # 검증 모드
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for X_batch, y_batch, prev_batch in val_loader:
                    # 모델과 같은 디바이스로 데이터 이동
                    X_batch = X_batch.to(device, non_blocking=device.type=='cuda')
                    y_batch = y_batch.to(device, non_blocking=device.type=='cuda')
                    prev_batch = prev_batch.to(device, non_blocking=device.type=='cuda')
                    
                    # 모델 예측 및 손실 계산
                    y_pred = model(X_batch, prev_batch)
                    loss = criterion(y_pred, y_batch, prev_batch)
                    
                    val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                # 스케줄러 업데이트
                scheduler.step(val_loss)
                
                # 모델 저장 (최적)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # 조기 종료
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 최적 모델 복원
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        logger.info(f"Model training completed with best validation loss: {best_val_loss:.4f}")
        
        # 학습 완료 후 GPU 상태 확인
        log_device_usage(device, "모델 학습 완료")
        
        # GPU 캐시 정리
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.info("🧹 GPU 캐시 정리 완료")
        
        # 모델, 스케일러, 파라미터 반환
        return model, scaler, target_col_idx
    
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        logger.error(traceback.format_exc())
        raise e
    
def optimize_hyperparameters_semimonthly_kfold(train_data, input_size, target_col_idx, device, current_period, file_path=None, n_trials=30, k_folds=10, use_cache=True):
    """
    시계열 K-fold 교차 검증을 사용하여 반월별 데이터에 대한 하이퍼파라미터 최적화 (Purchase_decision_5days.py 방식)
    """
    # 일관된 하이퍼파라미터 최적화를 위한 시드 고정
    set_seed()
    
    logger.info(f"\n===== {current_period} 하이퍼파라미터 최적화 시작 (시계열 {k_folds}-fold 교차 검증) =====")
    
    # 🔧 확장된 하이퍼파라미터 캐시 로직 - 기존 파일의 하이퍼파라미터도 탐색
    file_cache_dir = get_file_cache_dirs(file_path)['models']
    cache_file = os.path.join(file_cache_dir, f"hyperparams_kfold_{current_period.replace('-', '_')}.json")
    logger.info(f"📁 하이퍼파라미터 캐시 파일: {cache_file}")
    
    # models 디렉토리 생성
    os.makedirs(file_cache_dir, exist_ok=True)
    
    # 🔍 1단계: 하이퍼파라미터 캐시 확인
    if use_cache:
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_params = json.load(f)
                logger.info(f"✅ [{current_period}] 하이퍼파라미터 로드 완료")
                return cached_params
            except Exception as e:
                logger.error(f"캐시 파일 로드 오류: {str(e)}")
    
    # 🔍 2단계: 데이터 확장 시 기존 파일의 동일 기간 하이퍼파라미터 탐색
    if use_cache:
        logger.info(f"🔍 [{current_period}] 현재 파일에 캐시가 없습니다. 기존 파일에서 동일 기간의 하이퍼파라미터를 탐색합니다...")
        compatible_hyperparams = find_compatible_hyperparameters(file_path, current_period)
        if compatible_hyperparams:
            logger.info(f"✅ [{current_period}] 동일 기간의 호환 가능한 하이퍼파라미터를 발견했습니다!")
            logger.info(f"    📁 Source: {compatible_hyperparams['source_file']}")
            logger.info(f"    📊 Extension info: {compatible_hyperparams['extension_info']}")
            
            # 🔧 수정: 캐시 저장에 실패해도 기존 하이퍼파라미터 반환
            try:
                with open(cache_file, 'w') as f:
                    json.dump(compatible_hyperparams['hyperparams'], f, indent=2)
                logger.info(f"💾 [{current_period}] 호환 하이퍼파라미터를 현재 파일에 저장했습니다.")
            except Exception as e:
                logger.warning(f"⚠️ 하이퍼파라미터 저장 실패, 하지만 기존 하이퍼파라미터를 사용합니다: {str(e)}")
            
            # 🔑 핵심: 저장 성공/실패 여부와 관계없이 기존 하이퍼파라미터 반환
            logger.info(f"🚀 [{current_period}] 기존 하이퍼파라미터를 사용하여 최적화를 건너뜁니다.")
            return compatible_hyperparams['hyperparams']
                
        logger.info(f"🆕 [{current_period}] 동일 기간의 기존 하이퍼파라미터가 없습니다. 새로운 최적화를 진행합니다.")
    
            # 기본 하이퍼파라미터 정의 (최적화 실패 시 사용)
    default_params = {
        'sequence_length': 46,
        'hidden_size': 224,
        'num_layers': 6,
        'dropout': 0.318369281841675,
        'batch_size': 49,
        'learning_rate': 0.0009452489017042499,
        'num_epochs': 72,
        'loss_alpha': 0.7,  # DirectionalLoss alpha
        'loss_beta': 0.2,   # DirectionalLoss beta
        'patience': 14
    }
    
    # 데이터 길이 확인 - 충분하지 않으면 바로 기본값 반환
    MIN_DATA_SIZE = 100
    if len(train_data) < MIN_DATA_SIZE:
        logger.warning(f"훈련 데이터가 너무 적습니다 ({len(train_data)} 데이터 포인트 < {MIN_DATA_SIZE}). 기본 파라미터를 사용합니다.")
        return default_params
    
    # K-fold 분할 로직
    predict_window = 23  # 예측 윈도우 크기
    min_fold_size = 20 + predict_window + 5  # 최소 시퀀스 길이 + 예측 윈도우 + 여유
    max_possible_folds = len(train_data) // min_fold_size
    
    if max_possible_folds < 2:
        logger.warning(f"데이터가 충분하지 않아 k-fold를 수행할 수 없습니다 (가능한 fold: {max_possible_folds} < 2). 기본 파라미터를 사용합니다.")
        return default_params
    
    # 실제 사용 가능한 fold 수 조정
    k_folds = min(k_folds, max_possible_folds)
    fold_size = len(train_data) // (k_folds + 1)  # +1은 예측 윈도우를 위한 추가 부분

    logger.info(f"데이터 크기: {len(train_data)}, Fold 수: {k_folds}, 각 Fold 크기: {fold_size}")

    # fold 분할을 위한 인덱스 생성
    folds = []
    for i in range(k_folds):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        
        train_indices = list(range(0, test_start)) + list(range(test_end, len(train_data)))
        test_indices = list(range(test_start, test_end))
        
        folds.append((train_indices, test_indices))
    
    # Optuna 목적 함수 정의
    def objective(trial):
        # 일관된 하이퍼파라미터 최적화를 위한 시드 고정
        set_seed(SEED + trial.number)  # trial마다 다른 시드로 다양성 보장하면서도 재현 가능
        
        # 하이퍼파라미터 범위 수정 - 시퀀스 길이 최대값 제한
        max_seq_length = min(fold_size - predict_window - 5, 60)
        
        # 최소 시퀀스 길이도 제한
        min_seq_length = min(10, max_seq_length)
        
        if max_seq_length <= min_seq_length:
            logger.warning(f"시퀀스 길이 범위가 너무 제한적입니다 (min={min_seq_length}, max={max_seq_length}). 해당 trial 건너뛰기.")
            return float('inf')
        
        params = {
            'sequence_length': trial.suggest_int('sequence_length', min_seq_length, max_seq_length),
            'hidden_size': trial.suggest_int('hidden_size', 32, 256, step=8),
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'batch_size': trial.suggest_int('batch_size', 16, min(128, fold_size)),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'num_epochs': trial.suggest_int('num_epochs', 50, 200),
            'patience': trial.suggest_int('patience', 10, 30),
            'loss_alpha': trial.suggest_float('loss_alpha', 0.5, 0.9),
            'loss_beta': trial.suggest_float('loss_beta', 0.1, 0.3)
        }
        
        # loss_gamma 제거됨 - 단순화를 위해 DirectionalLoss만 사용
        
        # K-fold 교차 검증
        fold_losses = []
        valid_fold_count = 0
        
        for fold_idx, (train_indices, test_indices) in enumerate(folds):
            try:
                # 시퀀스 길이가 fold 크기보다 크면 건너뛰기
                if params['sequence_length'] >= len(test_indices):
                    logger.warning(f"Fold {fold_idx+1}: 시퀀스 길이({params['sequence_length']})가 테스트 데이터({len(test_indices)})보다 큽니다.")
                    continue
                
                # fold별 훈련/테스트 데이터 준비
                fold_train_data = train_data[train_indices]
                fold_test_data = train_data[test_indices]
                
                # 데이터 준비
                X_train, y_train, prev_train, X_val, y_val, prev_val = prepare_data(
                    fold_train_data, fold_test_data, params['sequence_length'],
                    predict_window, target_col_idx, augment=False
                )
                
                # 데이터가 충분한지 확인
                if len(X_train) < params['batch_size'] or len(X_val) < 1:
                    logger.warning(f"Fold {fold_idx+1}: 데이터 불충분 (훈련: {len(X_train)}, 검증: {len(X_val)})")
                    continue
                
                # 데이터셋 및 로더 생성 (CPU에서 생성, 학습 시 GPU로 이동)
                train_dataset = TimeSeriesDataset(X_train, y_train, torch.device('cpu'), prev_train)
                batch_size = min(params['batch_size'], len(X_train))
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    worker_init_fn=seed_worker,
                    generator=g
                )
                
                val_dataset = TimeSeriesDataset(X_val, y_val, torch.device('cpu'), prev_val)
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
                
                # 모델 생성
                model = ImprovedLSTMPredictor(
                    input_size=input_size,
                    hidden_size=params['hidden_size'],
                    num_layers=params['num_layers'],
                    dropout=params['dropout'],
                    output_size=predict_window
                ).to(device)

                # 손실 함수 생성
                criterion = DirectionalLoss(
                    alpha=params['loss_alpha'],
                    beta=params['loss_beta']
                )

                optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
                
                # 스케줄러 설정
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5,
                    patience=params['patience']//2
                )

                # best_val_loss 변수 명시적 정의
                best_val_loss = float('inf')
                patience_counter = 0
                
                for epoch in range(params['num_epochs']):
                    # 학습
                    model.train()
                    train_loss = 0
                    for X_batch, y_batch, prev_batch in train_loader:
                        optimizer.zero_grad()
                        
                        # 모델과 같은 디바이스로 데이터 이동
                        X_batch = X_batch.to(device, non_blocking=device.type=='cuda')
                        y_batch = y_batch.to(device, non_blocking=device.type=='cuda')
                        prev_batch = prev_batch.to(device, non_blocking=device.type=='cuda')
                        
                        # 모델 예측 및 손실 계산
                        y_pred = model(X_batch, prev_batch)
                        loss = criterion(y_pred, y_batch, prev_batch)
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        train_loss += loss.item()
                    
                    # 검증
                    model.eval()
                    val_loss = 0
                    
                    with torch.no_grad():
                        for X_batch, y_batch, prev_batch in val_loader:
                            # 모델과 같은 디바이스로 데이터 이동
                            X_batch = X_batch.to(device, non_blocking=device.type=='cuda')
                            y_batch = y_batch.to(device, non_blocking=device.type=='cuda')
                            prev_batch = prev_batch.to(device, non_blocking=device.type=='cuda')
                            
                            # 모델 예측 및 손실 계산
                            y_pred = model(X_batch, prev_batch)
                            loss = criterion(y_pred, y_batch, prev_batch)
                            
                            val_loss += loss.item()
                        
                        val_loss /= len(val_loader)
                        
                        # 스케줄러 업데이트
                        scheduler.step(val_loss)
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        
                        if patience_counter >= params['patience']:
                            break
                
                valid_fold_count += 1
                fold_losses.append(best_val_loss)
                
            except Exception as e:
                logger.error(f"Error in fold {fold_idx+1}: {str(e)}")
                continue
        
        # 모든 fold가 실패한 경우 매우 큰 손실값 반환
        if not fold_losses:
            logger.warning("모든 fold가 실패했습니다. 이 파라미터 조합은 건너뜁니다.")
            return float('inf')
        
        # 성공한 fold의 평균 손실값 반환
        return sum(fold_losses) / len(fold_losses)
    
    # Optuna 최적화 시도
    try:
        import optuna
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        # 최적 하이퍼파라미터
        if study.best_trial.value == float('inf'):
            logger.warning(f"모든 trial이 실패했습니다. 기본 하이퍼파라미터를 사용합니다.")
            return default_params
            
        best_params = study.best_params
        logger.info(f"\n{current_period} 최적 하이퍼파라미터 (K-fold):")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
        
        # 모든 필수 키가 있는지 확인
        required_keys = ['sequence_length', 'hidden_size', 'num_layers', 'dropout', 
                        'batch_size', 'learning_rate', 'num_epochs', 'patience',
                        'warmup_steps', 'lr_factor', 'lr_patience', 'min_lr',
                        'loss_alpha', 'loss_beta', 'loss_gamma', 'loss_delta']
        
        for key in required_keys:
            if key not in best_params:
                # 누락된 키가 있으면 기본값 할당
                if key == 'warmup_steps':
                    best_params[key] = 382
                elif key == 'lr_factor':
                    best_params[key] = 0.49
                elif key == 'lr_patience':
                    best_params[key] = 8
                elif key == 'min_lr':
                    best_params[key] = 1e-7
                elif key == 'loss_gamma':
                    best_params[key] = 0.07
                elif key == 'loss_delta':
                    best_params[key] = 0.07
                else:
                    best_params[key] = default_params[key]
        
        # 캐시에 저장
        with open(cache_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        logger.info(f"하이퍼파라미터가 {cache_file}에 저장되었습니다.")
        
        return best_params
        
    except Exception as e:
        logger.error(f"하이퍼파라미터 최적화 오류: {str(e)}")
        traceback.print_exc()
        
        # 오류 시 기본 하이퍼파라미터 반환
        return default_params
    
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, max_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        # warmup 단계 동안 선형 증가
        lr = self.max_lr * self.current_step / self.warmup_steps
        # warmup 단계를 초과하면 max_lr로 고정
        if self.current_step > self.warmup_steps:
            lr = self.max_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr