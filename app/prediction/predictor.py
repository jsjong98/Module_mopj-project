import pandas as pd
import numpy as np
import torch
import logging
import traceback
from datetime import timedelta
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
import json

from app.utils.file_utils import set_seed
from app.data.loader import load_data # generate_predictions에서 사용
from app.data.preprocessor import variable_groups
from app.utils.date_utils import is_holiday # is_holiday와 variable_groups는 여기에 필요
from app.data.cache_manager import get_file_cache_dirs, save_prediction_simple, find_compatible_cache_file
from app.models.lstm_model import ImprovedLSTMPredictor, optimize_hyperparameters_semimonthly_kfold
from app.core.gpu_manager import log_device_usage, check_gpu_availability
from app.utils.date_utils import get_next_n_business_days, get_semimonthly_period, get_next_semimonthly_period, get_semimonthly_date_range, format_date
from app.prediction.metrics import calculate_interval_averages_and_scores, decide_purchase_interval, compute_performance_metrics_improved, calculate_moving_averages_with_history
from app.visualization.attention_viz import visualize_attention_weights
from app.visualization.plotter import plot_prediction_basic, plot_moving_average_analysis
from app.utils.serialization import safe_serialize_value, convert_to_legacy_format
from app.core.state_manager import prediction_state
from app.config import UPLOAD_FOLDER

DEFAULT_DEVICE, CUDA_AVAILABLE = check_gpu_availability()
logger = logging.getLogger(__name__)

def generate_predictions(df, current_date, predict_window=23, features=None, target_col='MOPJ', file_path=None):
    """
    개선된 예측 수행 함수 - 예측 시작일의 반월 기간 하이퍼파라미터 사용
    🔑 데이터 누출 방지: current_date 이후의 실제값은 사용하지 않음
    """
    try:
        from app.data.preprocessor import select_features_from_groups
        from app.utils.date_utils import get_next_semimonthly_dates
        from app.models.lstm_model import train_model
        
        # 일관된 예측 결과를 위한 시드 고정
        set_seed()
        
        # 디바이스 설정
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_device_usage(device, "모델 학습 시작")
        
        # 현재 날짜가 문자열이면 datetime으로 변환
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
        
        # 현재 날짜 검증 (데이터 기준일)
        if current_date not in df.index:
            closest_date = df.index[df.index <= current_date][-1]
            logger.warning(f"Current date {current_date} not found in dataframe. Using closest date: {closest_date}")
            current_date = closest_date
        
        # 예측 시작일 계산
        prediction_start_date = current_date + pd.Timedelta(days=1)
        while prediction_start_date.weekday() >= 5 or is_holiday(prediction_start_date):
            prediction_start_date += pd.Timedelta(days=1)
        
        # 반월 기간 계산
        data_semimonthly_period = get_semimonthly_period(current_date)
        prediction_semimonthly_period = get_semimonthly_period(prediction_start_date)
        
        # ✅ 핵심 수정: 예측 시작일 기준으로 다음 반월 계산
        next_semimonthly_period = get_next_semimonthly_period(prediction_start_date)
        
        logger.info(f"🎯 Prediction Setup:")
        logger.info(f"  📅 Data base date: {current_date} (period: {data_semimonthly_period})")
        logger.info(f"  🚀 Prediction start date: {prediction_start_date} (period: {prediction_semimonthly_period})")
        logger.info(f"  🎯 Purchase interval target period: {next_semimonthly_period}")
        
        # 23일치 예측을 위한 날짜 생성
        all_business_days = get_next_n_business_days(current_date, df, predict_window)
        
        # ✅ 핵심 수정: 예측 시작일 기준으로 구매 구간 계산
        semimonthly_business_days, purchase_target_period = get_next_semimonthly_dates(prediction_start_date, df)
        
        logger.info(f"  📊 Total predictions: {len(all_business_days)} days")
        logger.info(f"  🛒 Purchase target period: {purchase_target_period}")
        logger.info(f"  📈 Purchase interval business days: {len(semimonthly_business_days)}")
        
        if not all_business_days:
            raise ValueError(f"No future business days found after {current_date}")

        # ✅ 핵심 수정: LSTM 단기 예측을 위해 2022년 이후 데이터만 사용
        cutoff_date_2022 = pd.to_datetime('2022-01-01')
        available_data = df[df.index <= current_date].copy()
        
        # 2022년 이후 데이터가 충분한 경우 해당 기간만 사용 (단기 예측 정확도 향상)
        recent_data = available_data[available_data.index >= cutoff_date_2022]
        if len(recent_data) >= 50:
            historical_data = recent_data.copy()
            logger.info(f"  🎯 Using recent data for LSTM: 2022+ ({len(historical_data)} records)")
        else:
            historical_data = available_data.copy()
            logger.info(f"  📊 Using full available data: insufficient recent data ({len(recent_data)} < 50)")
        
        logger.info(f"  📊 Training data: {len(historical_data)} records up to {format_date(current_date)}")
        logger.info(f"  📊 Training data range: {format_date(historical_data.index.min())} ~ {format_date(historical_data.index.max())}")
        
        # 최소 데이터 요구사항 확인
        if len(historical_data) < 50:
            raise ValueError(f"Insufficient training data: {len(historical_data)} records (minimum 50 required)")
        
        if features is None:
            selected_features, _ = select_features_from_groups(
                historical_data, 
                variable_groups,
                target_col=target_col,
                vif_threshold=50.0,
                corr_threshold=0.8
            )
        else:
            selected_features = features
            
        if target_col not in selected_features:
            selected_features.append(target_col)
        
        logger.info(f"  🔧 Selected features ({len(selected_features)}): {selected_features}")
        
        # ✅ 핵심 수정: 날짜별 다른 스케일링 보장
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(historical_data[selected_features])
        target_col_idx = selected_features.index(target_col)
        
        logger.info(f"  ⚖️  Scaler fitted on data up to {format_date(current_date)}")
        logger.info(f"  📊 Scaled data shape: {scaled_data.shape}")
        
        # ✅ 핵심: 예측 시작일의 반월 기간 하이퍼파라미터 사용
        optimized_params = optimize_hyperparameters_semimonthly_kfold(
            train_data=scaled_data,
            input_size=len(selected_features),
            target_col_idx=target_col_idx,
            device=device,
            current_period=prediction_semimonthly_period,  # ✅ 예측 시작일의 반월 기간
            file_path=file_path,  # 🔑 파일 경로 전달
            n_trials=30,
            k_folds=10,
            use_cache=True
        )
        
        logger.info(f"✅ Using hyperparameters for prediction start period: {prediction_semimonthly_period}")
        
        # ✅ 핵심 수정: 모델 학습 시 현재 날짜 기준으로 데이터 분할 보장
        logger.info(f"  🚀 Training model with data up to {format_date(current_date)}")
        model, model_scaler, model_target_col_idx = train_model(
            selected_features,
            target_col,
            current_date,
            historical_data,
            device,
            optimized_params
        )

        model.eval()
        
        # 스케일러 일관성 확인
        if model_target_col_idx != target_col_idx:
            logger.warning(f"Target column index mismatch: {model_target_col_idx} vs {target_col_idx}")
            target_col_idx = model_target_col_idx
        
        logger.info(f"  ✅ Model trained successfully for prediction starting {format_date(prediction_start_date)}")
        
        # ✅ 핵심 수정: 예측 데이터 준비 시 날짜별 다른 시퀀스 보장 (데이터 누출 방지)
        seq_len = optimized_params['sequence_length']
        
        # 🔑 중요: current_date를 예측하려면 current_date 이전의 데이터만 사용
        available_dates_before_current = [d for d in df.index if d < current_date]
        
        if len(available_dates_before_current) < seq_len:
            logger.warning(f"⚠️  Insufficient historical data before {format_date(current_date)}: {len(available_dates_before_current)} < {seq_len}")
            # 사용 가능한 모든 이전 데이터 사용
            sequence_dates = available_dates_before_current
        else:
            # 마지막 seq_len개의 이전 날짜 사용
            sequence_dates = available_dates_before_current[-seq_len:]
        
        # 시퀀스 데이터 추출 (current_date 제외!)
        sequence = df.loc[sequence_dates][selected_features].values
        
        logger.info(f"  📊 Sequence data: {sequence.shape} from {format_date(sequence_dates[0])} to {format_date(sequence_dates[-1])}")
        logger.info(f"  🚫 Excluded current_date: {format_date(current_date)} (preventing data leakage)")
        
        # 모델에서 반환된 스케일러 사용 (일관성 보장)
        sequence = model_scaler.transform(sequence)
        prev_value = sequence[-1, target_col_idx]
        
        logger.info(f"  📈 Previous value (scaled): {prev_value:.4f}")
        logger.info(f"  📊 Sequence length used: {len(sequence)} (required: {seq_len})")
        
        # 예측 수행
        future_predictions = []  # 미래 예측 (실제값 없음)
        validation_data = []     # 검증 데이터 (실제값 있음)
        
        with torch.no_grad():
            # 23영업일 전체에 대해 예측 수행
            max_pred_days = min(predict_window, len(all_business_days))
            current_sequence = sequence.copy()
            
            # 텐서로 변환
            X = torch.FloatTensor(current_sequence).unsqueeze(0).to(device)
            prev_tensor = torch.FloatTensor([prev_value]).to(device)
            
            # 전체 시퀀스 예측
            pred = model(X, prev_tensor).cpu().numpy()[0]
            
            # ✅ 핵심 수정: 각 날짜별 예측 생성 (데이터 누출 방지)
            for j, pred_date in enumerate(all_business_days[:max_pred_days]):
                # ✅ 스케일 역변환 시 일관된 스케일러 사용
                dummy_matrix = np.zeros((1, len(selected_features)))
                dummy_matrix[0, target_col_idx] = pred[j]
                pred_value = model_scaler.inverse_transform(dummy_matrix)[0, target_col_idx]
                
                # 예측값 검증 및 정리
                if np.isnan(pred_value) or np.isinf(pred_value):
                    logger.warning(f"Invalid prediction value for {pred_date}: {pred_value}, skipping")
                    continue
                
                pred_value = float(pred_value)
                
                # ✅ 실제 데이터 마지막 날짜 확인
                last_data_date = df.index.max()
                actual_value = None
                
                # ✅ 실제값 존재 여부 확인 및 설정
                if (pred_date in df.index and 
                    pd.notna(df.loc[pred_date, target_col]) and 
                    pred_date <= last_data_date):
                    
                    actual_value = float(df.loc[pred_date, target_col])
                    
                    if np.isnan(actual_value) or np.isinf(actual_value):
                        actual_value = None
                
                # 기본 예측 정보 (실제값 포함)
                prediction_item = {
                    'date': format_date(pred_date, '%Y-%m-%d'),
                    'prediction': pred_value,
                    'actual': actual_value,  # 🔑 실제값 항상 포함
                    'prediction_from': format_date(current_date, '%Y-%m-%d'),
                    'day_offset': j + 1,
                    'is_business_day': pred_date.weekday() < 5 and not is_holiday(pred_date),
                    'is_synthetic': pred_date not in df.index,
                    'semimonthly_period': data_semimonthly_period,
                    'next_semimonthly_period': next_semimonthly_period
                }
                
                # ✅ 실제값이 있는 경우 검증 데이터에도 추가
                if actual_value is not None:
                    validation_item = {
                        **prediction_item,
                        'error': abs(pred_value - actual_value),
                        'error_pct': abs(pred_value - actual_value) / actual_value * 100 if actual_value != 0 else 0.0
                    }
                    validation_data.append(validation_item)
                    
                    # 📊 검증 타입 구분 로그
                    if pred_date <= current_date:
                        logger.debug(f"  ✅ Training validation: {format_date(pred_date)} - Pred: {pred_value:.2f}, Actual: {actual_value:.2f}")
                    else:
                        logger.debug(f"  🎯 Test validation: {format_date(pred_date)} - Pred: {pred_value:.2f}, Actual: {actual_value:.2f}")
                elif pred_date > last_data_date:
                    logger.debug(f"  🔮 Future: {format_date(pred_date)} - Pred: {pred_value:.2f} (no actual - beyond data)")
                
                future_predictions.append(prediction_item)
        
        # 📊 검증 데이터 통계
        training_validation = len([v for v in validation_data if pd.to_datetime(v['date']) <= current_date])
        test_validation = len([v for v in validation_data if pd.to_datetime(v['date']) > current_date])
        
        logger.info(f"📊 Prediction Results:")
        logger.info(f"  📈 Total predictions: {len(future_predictions)}")
        logger.info(f"  ✅ Training validation (≤ {format_date(current_date)}): {training_validation}")
        logger.info(f"  🎯 Test validation (> {format_date(current_date)}): {test_validation}")
        logger.info(f"  📋 Total validation points: {len(validation_data)}")
        logger.info(f"  🔮 Pure future predictions (> {format_date(df.index.max())}): {len(future_predictions) - len(validation_data)}")
        
        if len(validation_data) == 0:
            logger.info("  ℹ️  Pure future prediction - no validation data available")
        
        # ✅ 구간 평균 및 점수 계산 - 올바른 구매 대상 기간 사용
        temp_predictions_for_interval = []
        for pred in future_predictions:
            pred_date = pd.to_datetime(pred['date'])
            if pred_date in semimonthly_business_days:  # 이제 올바른 다음 반월 날짜들
                temp_predictions_for_interval.append({
                    'Date': pred_date,
                    'Prediction': pred['prediction']
                })
        
        logger.info(f"  🛒 Predictions for interval calculation: {len(temp_predictions_for_interval)} (target period: {purchase_target_period})")
        
        interval_averages, interval_scores, analysis_info = calculate_interval_averages_and_scores(
            temp_predictions_for_interval, 
            semimonthly_business_days
        )

        # 최종 구매 구간 결정
        best_interval = decide_purchase_interval(interval_scores)

        # 성능 메트릭 계산 (검증 데이터가 있을 때만)
        metrics = None
        if validation_data:
            start_day_value = df.loc[current_date, target_col]
            if not (pd.isna(start_day_value) or np.isnan(start_day_value) or np.isinf(start_day_value)):
                try:
                    temp_df_for_metrics = pd.DataFrame([
                        {
                            'Date': pd.to_datetime(item['date']),
                            'Prediction': item['prediction'],
                            'Actual': item['actual']
                        } for item in validation_data
                    ])
                    
                    if not temp_df_for_metrics.empty:
                        metrics = compute_performance_metrics_improved(temp_df_for_metrics, start_day_value)
                        logger.info(f"  📊 Computed metrics from {len(validation_data)} validation points")
                    else:
                        logger.info("  ⚠️  No valid data for metrics computation")
                except Exception as e:
                    logger.error(f"Error computing metrics: {str(e)}")
                    metrics = None
            else:
                logger.warning("Invalid start_day_value for metrics computation")
        else:
            logger.info("  ℹ️  No validation data available - pure future prediction")
        
        # ✅ 이동평균 계산 시 실제값도 포함 (검증 데이터가 있는 경우)
        temp_predictions_for_ma = []
        for pred in future_predictions:
            pred_date = pd.to_datetime(pred['date'])
            actual_val = None
            
            # 실제 데이터가 존재하는 날짜면 실제값 설정
            if (pred_date in df.index and 
                pd.notna(df.loc[pred_date, target_col]) and 
                pred_date <= df.index.max()):
                actual_val = float(df.loc[pred_date, target_col])
            
            temp_predictions_for_ma.append({
                'Date': pred_date,
                'Prediction': pred['prediction'],
                'Actual': actual_val
            })
        
        logger.info(f"  📈 Calculating moving averages with historical data up to {format_date(current_date)}")
        ma_results = calculate_moving_averages_with_history(
            temp_predictions_for_ma, 
            historical_data,  # 이미 current_date까지로 필터링됨
            target_col=target_col
        )
        
        # 특성 중요도 분석
        attention_data = None
        try:
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            prev_tensor = torch.FloatTensor([float(prev_value)]).to(device)
            
            # 실제 시퀀스 날짜 정보 전달 (sequence_dates 변수 사용)
            actual_sequence_end_date = current_date  # current_date가 실제 데이터의 마지막 날짜
            attention_file, attention_img, feature_importance = visualize_attention_weights(
                model, sequence_tensor, prev_tensor, actual_sequence_end_date, selected_features, sequence_dates
            )
            
            attention_data = {
                'image': attention_img,
                'file_path': attention_file,
                'feature_importance': feature_importance
            }
        except Exception as e:
            logger.error(f"Error in attention analysis: {str(e)}")
        
        # 시각화 생성
        plots = {
            'basic_plot': {'file': None, 'image': None},
            'ma_plot': {'file': None, 'image': None}
        }

        start_day_value = None
        if current_date in df.index and not pd.isna(df.loc[current_date, target_col]):
            start_day_value = df.loc[current_date, target_col]

        # 📊 시각화용 데이터 준비 - 실제값 포함
        temp_df_for_plot_data = []
        for item in future_predictions:
            pred_date = pd.to_datetime(item['date'])
            actual_val = None
            
            # 실제 데이터가 존재하는 날짜면 실제값 설정
            if (pred_date in df.index and 
                pd.notna(df.loc[pred_date, target_col]) and 
                pred_date <= df.index.max()):
                actual_val = float(df.loc[pred_date, target_col])
            
            temp_df_for_plot_data.append({
                'Date': pred_date,
                'Prediction': item['prediction'],
                'Actual': actual_val
            })
        
        temp_df_for_plot = pd.DataFrame(temp_df_for_plot_data)

        if metrics:
            f1_score = metrics['f1']
            accuracy = metrics['accuracy']
            mape = metrics['mape']
            weighted_score = metrics['weighted_score']
            visualization_type = "with validation data"
        else:
            f1_score = accuracy = mape = weighted_score = 0.0
            visualization_type = "future prediction only"

        if start_day_value is not None and not temp_df_for_plot.empty:
            try:
                basic_plot_file, basic_plot_img = plot_prediction_basic(
                    temp_df_for_plot,
                    prediction_start_date,
                    start_day_value,
                    f1_score,
                    accuracy,
                    mape,
                    weighted_score,
                    current_date=current_date,  # 🔑 데이터 컷오프 날짜 전달
                    save_prefix=None,  # 파일별 캐시 시스템 사용
                    title_prefix=f"Prediction Graph ({visualization_type})",
                    file_path=file_path
                )
                
                ma_plot_file, ma_plot_img = plot_moving_average_analysis(
                    ma_results,
                    prediction_start_date,
                    save_prefix=None,  # 파일별 캐시 시스템 사용
                    title_prefix=f"Moving Average Analysis ({visualization_type})",
                    file_path=file_path
                )
                
                plots['basic_plot'] = {'file': basic_plot_file, 'image': basic_plot_img}
                plots['ma_plot'] = {'file': ma_plot_file, 'image': ma_plot_img}
                
                logger.info(f"  📊 Visualizations created ({visualization_type})")
                
            except Exception as e:
                logger.error(f"Error creating visualizations: {str(e)}")
                logger.error(traceback.format_exc())
        else:
            logger.warning("  ⚠️  No start day value or empty predictions - skipping visualizations")
        
        # 결과 반환 (급등락 모드 정보 포함)
        return {
            'predictions': future_predictions,
            'predictions_flat': future_predictions,  # 호환성을 위한 추가
            'validation_data': validation_data,
            'interval_scores': interval_scores,
            'interval_averages': interval_averages,
            'best_interval': best_interval,
            'ma_results': ma_results,
            'metrics': metrics,
            'selected_features': selected_features,
            'attention_data': attention_data,
            'plots': plots,
            'current_date': format_date(current_date, '%Y-%m-%d'),
            'data_end_date': format_date(current_date, '%Y-%m-%d'),
            'semimonthly_period': data_semimonthly_period,
            'next_semimonthly_period': purchase_target_period,  # ✅ 수정: 올바른 구매 대상 기간
            'prediction_semimonthly_period': prediction_semimonthly_period,
            'hyperparameter_period_used': prediction_semimonthly_period,
            'purchase_target_period': purchase_target_period,  # ✅ 추가
            'model_type': 'ImprovedLSTMPredictor',
            'loss_function': 'DirectionalLoss'
        }
        
    except Exception as e:
        logger.error(f"Error in prediction generation: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

def generate_predictions_compatible(df, current_date, predict_window=23, features=None, target_col='MOPJ'):
    """
    기존 프론트엔드와 호환되는 예측 함수
    (새로운 구조 + 기존 형태 변환)
    """
    try:
        # 일관된 예측 결과를 위한 시드 고정
        set_seed()
        
        # 새로운 generate_predictions 함수 실행
        new_results = generate_predictions(df, current_date, predict_window, features, target_col)
        
        # 기존 형태로 변환
        if isinstance(new_results.get('predictions'), dict):
            # 새로운 구조인 경우
            future_predictions = new_results['predictions']['future']
            validation_data = new_results['predictions']['validation']
            
            # future와 validation을 합쳐서 기존 형태로 변환
            all_predictions = future_predictions + validation_data
        else:
            # 기존 구조인 경우
            all_predictions = new_results.get('predictions_flat', new_results.get('predictions', []))
        
        # 기존 필드명으로 변환
        compatible_predictions = convert_to_legacy_format(all_predictions)
        
        # 결과에 기존 형태 추가
        new_results['predictions'] = compatible_predictions  # 기존 호환성
        new_results['predictions_new'] = new_results.get('predictions')  # 새로운 구조도 유지
        
        logger.info(f"Generated {len(compatible_predictions)} compatible predictions")
        
        return new_results
        
    except Exception as e:
        logger.error(f"Error in compatible prediction generation: {str(e)}")
        raise e

def generate_predictions_with_save(df, current_date, predict_window=23, features=None, target_col='MOPJ', save_to_csv=True, file_path=None):
    """
    예측 수행 및 스마트 캐시 저장이 포함된 함수 (수정됨)
    """
    try:
        # 일관된 예측 결과를 위한 시드 고정
        set_seed()
        
        logger.info(f"Starting prediction with smart cache save for {current_date}")
        
        # 기존 generate_predictions 함수 실행
        results = generate_predictions(df, current_date, predict_window, features, target_col, file_path)
        
        # 스마트 캐시 저장 옵션이 활성화된 경우
        if save_to_csv:
            logger.info("Saving prediction with smart cache system...")
            
            # 새로운 스마트 캐시 저장 함수 사용
            save_result = save_prediction_simple(results, current_date)
            results['save_info'] = save_result
            
            if save_result['success']:
                logger.info(f"✅ Smart cache save completed successfully")
                logger.info(f"  - Prediction Start Date: {save_result.get('prediction_start_date')}")
                logger.info(f"  - File: {save_result.get('file', 'N/A')}")
                
                # 캐시 정보 추가 (안전한 키 접근)
                results['cache_info'] = {
                    'saved': True,
                    'prediction_start_date': save_result.get('prediction_start_date'),
                    'file': save_result.get('file'),
                    'success': save_result.get('success', False)
                }
            else:
                logger.warning(f"❌ Failed to save prediction with smart cache: {save_result.get('error')}")
                results['cache_info'] = {
                    'saved': False,
                    'error': save_result.get('error')
                }
        else:
            logger.info("Skipping smart cache save (save_to_csv=False)")
            results['save_info'] = {'success': False, 'reason': 'save_to_csv=False'}
            results['cache_info'] = {
                'saved': False,
                'reason': 'save_to_csv=False'
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in generate_predictions_with_save: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 오류 발생 시에도 예측 결과는 반환하되, 저장 실패 정보 포함
        if 'results' in locals():
            results['save_info'] = {'success': False, 'error': str(e)}
            results['cache_info'] = {'saved': False, 'error': str(e)}
            return results
        else:
            # 예측 자체가 실패한 경우
            raise e

def generate_predictions_with_attention_save(df, current_date, predict_window=23, features=None, target_col='MOPJ', save_to_csv=True, file_path=None):
    """
    예측 수행 및 attention 포함 CSV 저장 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        전체 데이터
    current_date : str or datetime
        현재 날짜 (데이터 기준일)
    predict_window : int
        예측 기간 (기본 23일)
    features : list, optional
        사용할 특성 목록
    target_col : str
        타겟 컬럼명 (기본 'MOPJ')
    save_to_csv : bool
        CSV 저장 여부 (기본 True)
    
    Returns:
    --------
    dict : 예측 결과 (attention 데이터 포함)
    """
    try:
        # 일관된 예측 결과를 위한 시드 고정
        set_seed()
        
        logger.info(f"Starting prediction with attention save for {current_date}")
        
        # 기존 generate_predictions 함수 실행
        results = generate_predictions(df, current_date, predict_window, features, target_col, file_path)
        
        # attention 포함 저장 옵션이 활성화된 경우
        if save_to_csv:
            logger.info("Saving prediction with attention data...")
            save_result = save_prediction_simple(results, current_date)
            results['save_info'] = save_result
            
            if save_result['success']:
                logger.info(f"SUCCESS: Prediction with attention saved successfully")
                logger.info(f"  - CSV: {save_result['csv_file']}")
                logger.info(f"  - Metadata: {save_result['meta_file']}")
                logger.info(f"  - Attention: {save_result['attention_file'] if save_result.get('attention_file') else 'Not saved'}")
            else:
                logger.warning(f"❌ Failed to save prediction with attention: {save_result.get('error')}")
        else:
            logger.info("Skipping CSV save (save_to_csv=False)")
            results['save_info'] = {'success': False, 'reason': 'save_to_csv=False'}
        
        return results
        
    except Exception as e:
        logger.error(f"Error in generate_predictions_with_attention_save: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 오류 발생 시에도 예측 결과는 반환하되, 저장 실패 정보 포함
        if 'results' in locals():
            results['save_info'] = {'success': False, 'error': str(e)}
            return results
        else:
            # 예측 자체가 실패한 경우
            raise e

#######################################################################
# 백그라운드 작업 처리
#######################################################################
# 🔧 SyntaxError 수정 - check_existing_prediction 함수 (3987라인 근처)

def find_compatible_hyperparameters(current_file_path, current_period):
    """
    현재 파일이 기존 파일의 확장인 경우, 기존 파일의 호환 가능한 하이퍼파라미터를 찾는 함수
    
    Parameters:
    -----------
    current_file_path : str
        현재 파일 경로
    current_period : str
        현재 예측 기간
        
    Returns:
    --------
    dict or None: {
        'hyperparams': dict,
        'source_file': str,
        'extension_info': dict
    } 또는 None (호환 가능한 하이퍼파라미터가 없을 경우)
    """
    try:
        # uploads 폴더의 다른 파일들을 확인 (🔧 수정: xlsx 파일도 포함)
        upload_dir = Path(UPLOAD_FOLDER)
        existing_files = [f for f in upload_dir.glob('*.xlsx') if str(f) != current_file_path]
        logger.info(f"🔍 [HYPERPARAMS_SEARCH] 탐색할 기존 파일 수: {len(existing_files)}")
        for i, file in enumerate(existing_files):
            logger.info(f"    {i+1}. {file.name}")
        
        for existing_file in existing_files:
            try:
                from app.data.cache_manager import check_data_extension
                # 🔧 수정: 확장 관계 확인 + 단순 파일명 유사성 확인
                extension_result = check_data_extension(str(existing_file), current_file_path)
                is_extension = extension_result.get('is_extension', False)
                
                # 📝 확장 관계가 인식되지 않는 경우 파일명 유사성으로 대체 확인
                if not is_extension:
                    existing_name = existing_file.stem.lower()
                    current_name = Path(current_file_path).stem.lower()
                    # 기본 이름이 같거나 하나가 다른 하나를 포함하는 경우
                    if (existing_name in current_name or current_name in existing_name or 
                        existing_name.replace('_', '') == current_name.replace('_', '')):
                        is_extension = True
                        logger.info(f"🔍 [HYPERPARAMS_SEARCH] 파일명 유사성으로 확장 관계 인정: {existing_file.name} -> {Path(current_file_path).name}")
                
                if is_extension:
                    if extension_result.get('is_extension', False):
                        logger.info(f"🔍 [HYPERPARAMS_SEARCH] 확장 관계 발견: {existing_file.name} -> {Path(current_file_path).name}")
                        logger.info(f"    📈 Extension type: {extension_result.get('validation_details', {}).get('extension_type', 'Unknown')}")
                        logger.info(f"    ➕ New rows: {extension_result.get('new_rows_count', 0)}")
                    else:
                        logger.info(f"🔍 [HYPERPARAMS_SEARCH] 파일명 유사성 기반 호환성 인정: {existing_file.name} -> {Path(current_file_path).name}")
                    
                    # 기존 파일의 하이퍼파라미터 캐시 확인
                    existing_cache_dirs = get_file_cache_dirs(str(existing_file))
                    existing_models_dir = existing_cache_dirs['models']
                    
                    if os.path.exists(existing_models_dir):
                        # 해당 기간의 하이퍼파라미터 파일 찾기
                        hyperparams_pattern = f"hyperparams_kfold_{current_period.replace('-', '_')}.json"
                        hyperparams_file = os.path.join(existing_models_dir, hyperparams_pattern)
                        
                        if os.path.exists(hyperparams_file):
                            try:
                                with open(hyperparams_file, 'r') as f:
                                    hyperparams = json.load(f)
                                
                                logger.info(f"✅ [HYPERPARAMS_SEARCH] 기존 파일에서 호환 하이퍼파라미터 발견!")
                                logger.info(f"    📁 Source file: {existing_file.name}")
                                logger.info(f"    📊 Hyperparams file: {hyperparams_pattern}")
                                
                                return {
                                    'hyperparams': hyperparams,
                                    'source_file': str(existing_file),
                                    'extension_info': extension_result,
                                    'period': current_period
                                }
                                
                            except Exception as e:
                                logger.warning(f"기존 하이퍼파라미터 파일 로드 실패 ({existing_file.name}): {str(e)}")
                        else:
                            # ❌ 삭제된 부분: 다른 기간의 하이퍼파라미터를 대체로 사용하는 로직 제거
                            logger.info(f"🔍 [HYPERPARAMS_SEARCH] {current_period} 기간의 하이퍼파라미터가 없습니다. 새로운 최적화가 필요합니다.")
                    
            except Exception as e:
                logger.warning(f"파일 확장 관계 확인 실패 ({existing_file.name}): {str(e)}")
                continue
        
        logger.info(f"❌ [HYPERPARAMS_SEARCH] 호환 가능한 하이퍼파라미터를 찾지 못했습니다.")
        return None
        
    except Exception as e:
        logger.error(f"하이퍼파라미터 호환성 탐색 중 오류: {str(e)}")
        return None
