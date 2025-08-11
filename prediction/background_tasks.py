import pandas as pd
import numpy as np
import time
import logging
from threading import Thread
from datetime import datetime, timedelta
import traceback
import logging
import json
import os

from app.data.loader import load_data
from app.data.cache_manager import check_existing_prediction, load_accumulated_predictions_from_csv, rebuild_predictions_index_from_existing_files, get_file_cache_dirs
from app.prediction.predictor import generate_predictions_with_save
from app.prediction.metrics import calculate_prediction_consistency, calculate_accumulated_purchase_reliability, calculate_actual_business_days
from app.visualization.plotter import visualize_accumulated_metrics, plot_prediction_basic, plot_moving_average_analysis
from app.visualization.attention_viz import visualize_attention_weights
from app.utils.date_utils import format_date, is_holiday
from app.utils.file_utils import cleanup_excel_processes, set_seed
from app.utils.serialization import safe_serialize_value, clean_predictions_data, clean_cached_predictions, clean_interval_scores_safe, convert_to_legacy_format
from app.core.state_manager import prediction_state
from app.core.gpu_manager import log_device_usage, check_gpu_availability
from app.models.varmax_model import VARMAXSemiMonthlyForecaster


DEFAULT_DEVICE, CUDA_AVAILABLE = check_gpu_availability()
logger = logging.getLogger(__name__)

def calculate_estimated_time_remaining(start_time, current_progress):
    """
    예측 시작 시간과 현재 진행률을 기반으로 남은 시간을 계산합니다.
    
    Args:
        start_time: 예측 시작 시간 (time.time() 값)
        current_progress: 현재 진행률 (0-100)
    
    Returns:
        dict: {
            'estimated_remaining_seconds': int,
            'estimated_remaining_text': str,
            'elapsed_time_seconds': int,
            'elapsed_time_text': str
        }
    """
    if not start_time or current_progress <= 0:
        return {
            'estimated_remaining_seconds': None,
            'estimated_remaining_text': '계산 중...',
            'elapsed_time_seconds': 0,
            'elapsed_time_text': '0초'
        }
    
    current_time = time.time()
    elapsed_time = current_time - start_time
    
    # 진행률이 100% 이상이면 완료
    if current_progress >= 100:
        return {
            'estimated_remaining_seconds': 0,
            'estimated_remaining_text': '완료',
            'elapsed_time_seconds': int(elapsed_time),
            'elapsed_time_text': format_time_duration(int(elapsed_time))
        }
    
    # 진행률을 기반으로 총 예상 시간 계산
    estimated_total_time = elapsed_time * (100 / current_progress)
    estimated_remaining_time = estimated_total_time - elapsed_time
    
    # 음수가 되지 않도록 보정
    estimated_remaining_time = max(0, estimated_remaining_time)
    
    return {
        'estimated_remaining_seconds': int(estimated_remaining_time),
        'estimated_remaining_text': format_time_duration(int(estimated_remaining_time)),
        'elapsed_time_seconds': int(elapsed_time),
        'elapsed_time_text': format_time_duration(int(elapsed_time))
    }

def format_time_duration(seconds):
    """시간을 사람이 읽기 쉬운 형태로 포맷팅"""
    if seconds < 60:
        return f"{seconds}초"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        if remaining_seconds > 0:
            return f"{minutes}분 {remaining_seconds}초"
        else:
            return f"{minutes}분"
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        if remaining_minutes > 0:
            return f"{hours}시간 {remaining_minutes}분"
        else:
            return f"{hours}시간"
        
def run_accumulated_predictions_with_save(file_path, start_date, end_date=None, save_to_csv=True, use_saved_data=True):
    """
    시작 날짜부터 종료 날짜까지 각 날짜별로 예측을 수행하고 결과를 누적합니다. (최적화됨 - 데이터 한번만 로딩)
    """
    global prediction_state

    try:
        # 상태 초기화
        prediction_state['is_predicting'] = True
        prediction_state['prediction_progress'] = 5
        prediction_state['prediction_start_time'] = time.time()  # 시작 시간 기록
        prediction_state['error'] = None
        prediction_state['accumulated_predictions'] = []
        prediction_state['accumulated_metrics'] = {}
        prediction_state['prediction_dates'] = []
        prediction_state['accumulated_consistency_scores'] = {}
        prediction_state['current_file'] = file_path  # ✅ 현재 파일 경로 설정
        prediction_state['latest_file_path'] = file_path  # 단일 예측과 호환성 유지
        
        logger.info(f"🎯 [ACCUMULATED] Running accumulated predictions from {start_date} to {end_date}")
        logger.info(f"  📁 Data file: {file_path}")
        logger.info(f"  💾 Save to CSV: {save_to_csv}")
        logger.info(f"  🔄 Use saved data: {use_saved_data}")

        # 입력 날짜를 datetime 객체로 변환
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if end_date is not None and isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # 🚀 데이터 로드 (누적 예측용 - LSTM 모델, 2022년 이전 데이터 제거) - 한 번만 수행!
        logger.info("📂 [ACCUMULATED] Loading data once for all predictions...")
        df = load_data(file_path, model_type='lstm')
        prediction_state['current_data'] = df
        prediction_state['prediction_progress'] = 8
        logger.info(f"✅ [ACCUMULATED] Data loaded successfully: {df.shape} (from {df.index.min()} to {df.index.max()})")

        # 저장된 데이터 활용 옵션이 켜져 있으면 먼저 CSV에서 로드 시도
        loaded_predictions = []
        if use_saved_data:
            logger.info("🔍 [CACHE] Attempting to load existing predictions from CSV files...")
            
            # 🔧 인덱스 파일이 없으면 기존 파일들로부터 재생성
            cache_dirs = get_file_cache_dirs(file_path)
            predictions_index_file = cache_dirs['predictions'] / 'predictions_index.cs'
            
            if not predictions_index_file.exists():
                logger.warning("⚠️ [CACHE] predictions_index.cs not found, attempting to rebuild from existing files...")
                if rebuild_predictions_index_from_existing_files():
                    logger.info("✅ [CACHE] Successfully rebuilt predictions index")
                else:
                    logger.warning("⚠️ [CACHE] Failed to rebuild predictions index")
            
            loaded_predictions = load_accumulated_predictions_from_csv(start_date, end_date, file_path=file_path)  # ✅ 파일 경로 추가
            logger.info(f"📦 [CACHE] Successfully loaded {len(loaded_predictions)} predictions from CSV cache")
            if len(loaded_predictions) > 0:
                logger.info(f"💡 [CACHE] Using cached predictions will significantly speed up processing!")

        prediction_state['prediction_progress'] = 10

        # 종료 날짜가 지정되지 않으면 데이터의 마지막 날짜 사용
        if end_date is None:
            end_date = df.index.max()

        # 사용 가능한 날짜 추출 후 정렬
        available_dates = [date for date in df.index if start_date <= date <= end_date]
        available_dates.sort()
        
        if not available_dates:
            raise ValueError(f"지정된 기간 내에 사용 가능한 날짜가 없습니다: {start_date} ~ {end_date}")

        total_dates = len(available_dates)
        logger.info(f"Accumulated prediction: {total_dates} dates from {start_date} to {end_date}")

        # 누적 성능 지표 초기화
        accumulated_metrics = {
            'f1': 0.0,
            'accuracy': 0.0,
            'mape': 0.0,
            'weighted_score': 0.0,
            'total_predictions': 0
        }

        # 이미 로드된 예측 결과들을 날짜별 딕셔너리로 변환
        loaded_by_date = {}
        for pred in loaded_predictions:
            loaded_by_date[pred['date']] = pred

        # ✅ 캐시 활용 통계 초기화
        cache_statistics = {
            'total_dates': 0,
            'cached_dates': 0,
            'new_predictions': 0,
            'cache_hit_rate': 0.0
        }

        all_predictions = []
        accumulated_interval_scores = {}

        # 각 날짜별 예측 수행 또는 로드
        for i, current_date in enumerate(available_dates):
            current_date_str = format_date(current_date)
            cache_statistics['total_dates'] += 1
            
            logger.info(f"Processing date {i+1}/{total_dates}: {current_date_str}")
            
            # 이미 로드된 데이터가 있으면 사용
            if current_date_str in loaded_by_date:
                cache_statistics['cached_dates'] += 1  # ✅ 캐시 사용 시 카운터 증가
                logger.info(f"⚡ [CACHE] Using cached prediction for {current_date_str} (skipping computation)")
                date_result = loaded_by_date[current_date_str]
                
                # 🔧 캐시된 metrics 안전성 처리
                metrics = date_result.get('metrics')
                if not metrics or not isinstance(metrics, dict):
                    logger.warning(f"⚠️ [CACHE] Invalid metrics for {current_date_str}, using defaults")
                    metrics = {'f1': 0.0, 'accuracy': 0.0, 'mape': 0.0, 'weighted_score': 0.0}
                
                # 누적 성능 지표 업데이트
                accumulated_metrics['f1'] += metrics.get('f1', 0.0)
                accumulated_metrics['accuracy'] += metrics.get('accuracy', 0.0)
                accumulated_metrics['mape'] += metrics.get('mape', 0.0)
                accumulated_metrics['weighted_score'] += metrics.get('weighted_score', 0.0)
                accumulated_metrics['total_predictions'] += 1
                
            else:
                # 새로운 예측 수행
                cache_statistics['new_predictions'] += 1
                logger.info(f"🚀 [COMPUTE] Running new prediction for {current_date_str} (not in cache)")
                try:
                    # ✅ 누적 예측에서도 모든 새 예측을 저장하도록 보장
                    results = generate_predictions_with_save(df, current_date, save_to_csv=True, file_path=file_path)
                    
                    # 예측 데이터 타입 안전 확인
                    predictions = results.get('predictions_flat', results.get('predictions', []))
                    
                    # 예측 데이터가 중첩된 딕셔너리 구조인 경우 처리
                    if isinstance(predictions, dict):
                        if 'future' in predictions:
                            predictions = predictions['future']
                        elif 'predictions' in predictions:
                            predictions = predictions['predictions']
                    
                    if not predictions or not isinstance(predictions, list):
                        logger.warning(f"No valid predictions found for {current_date_str}: {type(predictions)}")
                        continue
                        
                    # 실제 예측한 영업일 수 계산 (안전한 방식)
                    actual_business_days = 0
                    try:
                        for p in predictions:
                            # p가 딕셔너리인지 확인
                            if isinstance(p, dict):
                                date_key = p.get('Date') or p.get('date')
                                is_synthetic = p.get('is_synthetic', False)
                                if date_key and not is_synthetic:
                                    actual_business_days += 1
                            else:
                                logger.warning(f"Prediction item is not dict for {current_date_str}: {type(p)}")
                    except Exception as calc_error:
                        logger.error(f"Error calculating business days: {str(calc_error)}")
                        actual_business_days = len(predictions)  # 기본값
                    
                    metrics = results.get('metrics', {})
                    if not metrics:
                        # 메트릭이 없으면 기본값 설정
                        metrics = {'f1': 0.0, 'accuracy': 0.0, 'mape': 0.0, 'weighted_score': 0.0}
                    
                    # 누적 성능 지표 업데이트
                    accumulated_metrics['f1'] += metrics.get('f1', 0.0)
                    accumulated_metrics['accuracy'] += metrics.get('accuracy', 0.0)
                    accumulated_metrics['mape'] += metrics.get('mape', 0.0)
                    accumulated_metrics['weighted_score'] += metrics.get('weighted_score', 0.0)
                    accumulated_metrics['total_predictions'] += 1

                    # 안전한 데이터 구조 생성
                    safe_predictions = predictions if isinstance(predictions, list) else []
                    safe_interval_scores = results.get('interval_scores', {})
                    if not isinstance(safe_interval_scores, dict):
                        safe_interval_scores = {}
                    
                    date_result = {
                        'date': current_date_str,
                        'predictions': safe_predictions,
                        'metrics': metrics,
                        'interval_scores': safe_interval_scores,
                        'actual_business_days': actual_business_days,
                        'next_semimonthly_period': results.get('next_semimonthly_period'),
                        'original_interval_scores': safe_interval_scores,
                        'ma_results': results.get('ma_results', {}),  # 🔑 이동평균 데이터 추가
                        'attention_data': results.get('attention_data', {})  # 🔑 Attention 데이터 추가
                    }
                    
                except Exception as e:
                    logger.error(f"Error in prediction for date {current_date}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue

            # 구간 점수 누적 처리 (안전한 방식)
            interval_scores = date_result.get('interval_scores', {})
            if isinstance(interval_scores, dict):
                for interval in interval_scores.values():
                    if not interval or not isinstance(interval, dict) or 'days' not in interval or interval['days'] is None:
                        continue
                    interval_key = f"{format_date(interval['start_date'])}-{format_date(interval['end_date'])}"
                    if interval_key in accumulated_interval_scores:
                        accumulated_interval_scores[interval_key]['score'] += interval['score']
                        accumulated_interval_scores[interval_key]['count'] += 1
                        accumulated_interval_scores[interval_key]['avg_price'] = (
                            (accumulated_interval_scores[interval_key]['avg_price'] *
                             (accumulated_interval_scores[interval_key]['count'] - 1) +
                             interval['avg_price']) / accumulated_interval_scores[interval_key]['count']
                        )
                    else:
                        accumulated_interval_scores[interval_key] = interval.copy()
                        accumulated_interval_scores[interval_key]['count'] = 1

            all_predictions.append(date_result)
            prediction_state['prediction_progress'] = 10 + int(90 * (i + 1) / total_dates)

        # 평균 성능 지표 계산
        if accumulated_metrics['total_predictions'] > 0:
            count = accumulated_metrics['total_predictions']
            accumulated_metrics['f1'] /= count
            accumulated_metrics['accuracy'] /= count
            accumulated_metrics['mape'] /= count
            accumulated_metrics['weighted_score'] /= count
            
            # 🔧 NaN 값 처리 강화
            for metric_key in ['f1', 'accuracy', 'mape', 'weighted_score']:
                if pd.isna(accumulated_metrics[metric_key]) or np.isnan(accumulated_metrics[metric_key]) or np.isinf(accumulated_metrics[metric_key]):
                    logger.warning(f"⚠️ [METRICS] NaN/Inf detected in {metric_key}, setting to 0.0")
                    accumulated_metrics[metric_key] = 0.0

        # 예측 신뢰도 계산
        logger.info("Calculating prediction consistency scores...")
        unique_periods = set()
        for pred in all_predictions:
            if 'next_semimonthly_period' in pred and pred['next_semimonthly_period']:
                unique_periods.add(pred['next_semimonthly_period'])
        
        accumulated_consistency_scores = {}
        for period in unique_periods:
            try:
                consistency_data = calculate_prediction_consistency(all_predictions, period)
                # 🔧 NaN 값 처리 강화
                if consistency_data and 'consistency_score' in consistency_data:
                    consistency_score = consistency_data['consistency_score']
                    if pd.isna(consistency_score) or np.isnan(consistency_score) or np.isinf(consistency_score):
                        logger.warning(f"⚠️ [CONSISTENCY] NaN/Inf detected for period {period}, setting to 0.0")
                        consistency_data['consistency_score'] = 0.0
                accumulated_consistency_scores[period] = consistency_data
                logger.info(f"Consistency score for {period}: {consistency_data.get('consistency_score', 'N/A')}")
            except Exception as e:
                logger.error(f"Error calculating consistency for period {period}: {str(e)}")

        # accumulated_interval_scores 처리
        accumulated_scores_list = [
            v for v in accumulated_interval_scores.values()
            if v is not None and isinstance(v, dict) and 'days' in v and v['days'] is not None
        ]
        accumulated_scores_list.sort(key=lambda x: x['score'], reverse=True)

        accumulated_purchase_reliability, debug_info = calculate_accumulated_purchase_reliability(all_predictions)
        
        # ✅ 캐시 활용률 계산
        cache_statistics['cache_hit_rate'] = (cache_statistics['cached_dates'] / cache_statistics['total_dates'] * 100) if cache_statistics['total_dates'] > 0 else 0.0
        # 🔧 NaN 값 처리 강화
        if pd.isna(cache_statistics['cache_hit_rate']) or np.isnan(cache_statistics['cache_hit_rate']) or np.isinf(cache_statistics['cache_hit_rate']):
            logger.warning(f"⚠️ [CACHE] NaN/Inf detected in cache_hit_rate, setting to 0.0")
            cache_statistics['cache_hit_rate'] = 0.0
        logger.info(f"🎯 [CACHE] Final statistics: {cache_statistics['cached_dates']}/{cache_statistics['total_dates']} cached ({cache_statistics['cache_hit_rate']:.1f}%), {cache_statistics['new_predictions']} new predictions computed")
        
        # 결과 저장
        prediction_state['accumulated_predictions'] = all_predictions
        prediction_state['accumulated_metrics'] = accumulated_metrics
        prediction_state['prediction_dates'] = [p['date'] for p in all_predictions]
        prediction_state['accumulated_interval_scores'] = accumulated_scores_list
        prediction_state['accumulated_consistency_scores'] = accumulated_consistency_scores
        prediction_state['accumulated_purchase_reliability'] = accumulated_purchase_reliability
        prediction_state['accumulated_purchase_debug'] = debug_info
        prediction_state['cache_statistics'] = cache_statistics  # ✅ 캐시 통계 추가

        if all_predictions:
            latest = all_predictions[-1]
            prediction_state['latest_predictions'] = latest['predictions']
            prediction_state['current_date'] = latest['date']

        prediction_state['is_predicting'] = False
        prediction_state['prediction_progress'] = 100
        logger.info(f"Accumulated prediction completed for {len(all_predictions)} dates")
        
        # 🔧 Excel 프로세스 정리
        cleanup_excel_processes()
        
    except Exception as e:
        logger.error(f"Error in accumulated prediction: {str(e)}")
        logger.error(traceback.format_exc())
        prediction_state['error'] = str(e)
        prediction_state['is_predicting'] = False
        prediction_state['prediction_progress'] = 0
        prediction_state['accumulated_consistency_scores'] = {}
        
        # 🔧 오류 발생 시에도 Excel 프로세스 정리
        cleanup_excel_processes()

def background_accumulated_prediction(file_path, start_date, end_date=None):
    """백그라운드에서 누적 예측을 수행하는 함수"""
    thread = Thread(target=run_accumulated_predictions_with_save, args=(file_path, start_date, end_date))
    thread.daemon = True
    thread.start()
    return thread

def background_prediction_simple_compatible(file_path, current_date, save_to_csv=True, use_cache=True):
    """호환성을 유지하는 백그라운드 예측 함수 - 캐시 우선 사용, JSON 안전성 보장"""
    global prediction_state
    
    try:
        # 일관된 예측 결과를 위한 시드 고정
        set_seed()
        prediction_state['is_predicting'] = True
        prediction_state['prediction_progress'] = 10
        prediction_state['prediction_start_time'] = time.time()  # 시작 시간 기록
        prediction_state['error'] = None
        prediction_state['latest_file_path'] = file_path  # 파일 경로 저장
        prediction_state['current_file'] = file_path  # 캐시 연동용 파일 경로
        
        logger.info(f"🎯 Starting compatible prediction for {current_date}")
        logger.info(f"  🔄 Cache enabled: {use_cache}")
        
        # 데이터 로드 (단일 날짜 예측용 - LSTM 모델, 2022년 이전 데이터 제거)
        df = load_data(file_path, model_type='lstm')
        prediction_state['current_data'] = df
        prediction_state['prediction_progress'] = 20
        
        # 현재 날짜 처리 및 영업일 조정
        if current_date is None:
            current_date = df.index.max()
        else:
            current_date = pd.to_datetime(current_date)
        
        # 🎯 휴일이면 다음 영업일로 조정
        original_date = current_date
        adjusted_date = current_date
        
        # 주말이나 휴일이면 다음 영업일로 이동
        while adjusted_date.weekday() >= 5 or is_holiday(adjusted_date):
            adjusted_date += pd.Timedelta(days=1)
        
        if adjusted_date != original_date:
            logger.info(f"📅 Date adjusted for business day: {original_date.strftime('%Y-%m-%d')} -> {adjusted_date.strftime('%Y-%m-%d')}")
            logger.info(f"  📋 Reason: {'Weekend' if original_date.weekday() >= 5 else 'Holiday'}")
        
        current_date = adjusted_date
        
        # 캐시 확인 - 확장 데이터 스마트 활용
        if use_cache:
            logger.info("🔍 Checking for existing prediction cache...")
            prediction_state['prediction_progress'] = 30
            
            try:
                from app.data.cache_manager import check_existing_prediction, find_compatible_cache_file, load_existing_predictions_for_extension
                # 1. 먼저 특정 날짜의 정확한 캐시 확인
                cached_result = check_existing_prediction(current_date, file_path)
                logger.info(f"  📋 Direct cache check result: {cached_result is not None}")
                
                # 2. 직접 캐시가 없으면 확장 데이터 캐시 활용 검토
                if not cached_result or not cached_result.get('success'):
                    logger.info("🔄 Direct cache not found, checking for extension cache compatibility...")
                    
                    # 현재 파일이 확장된 데이터인지 확인
                    logger.info(f"🔍 [EXTENSION_CHECK] Checking for compatible cache files...")
                    compatibility_info = find_compatible_cache_file(file_path)
                    
                    logger.info(f"📊 [EXTENSION_CHECK] Compatibility check result:")
                    logger.info(f"    Found: {compatibility_info.get('found')}")
                    logger.info(f"    Cache type: {compatibility_info.get('cache_type')}")
                    logger.info(f"    Cache files: {len(compatibility_info.get('cache_files', []))}")
                    
                    if (compatibility_info.get('found') and 
                        compatibility_info.get('cache_type') in ['extension', 'hash_based', 'near_complete']):
                        
                        logger.info("📈 Extension data detected! Attempting smart cache utilization...")
                        
                        # 기존 캐시에서 사용 가능한 예측들 로드
                        existing_predictions = load_existing_predictions_for_extension(
                            file_path, current_date, compatibility_info
                        )
                        
                        if existing_predictions:
                            logger.info(f"✅ Found {len(existing_predictions)} existing predictions to reuse!")
                            
                            # 🔧 개선된 로직: 부분적 캐시 재활용 + 새로운 날짜 추가 예측
                            existing_dates = [pd.to_datetime(p['Date']) for p in existing_predictions]
                            existing_date_range = (min(existing_dates), max(existing_dates))
                            
                            logger.info(f"📅 Existing prediction range: {existing_date_range[0].strftime('%Y-%m-%d')} ~ {existing_date_range[1].strftime('%Y-%m-%d')}")
                            logger.info(f"🎯 Requested date: {current_date.strftime('%Y-%m-%d')}")
                            
                                                                                     # 현재 날짜가 기존 예측에 정확히 있는지 확인
                            if current_date in existing_dates:
                                # Case 1: 정확한 날짜 매치 - 전체 캐시 재활용
                                logger.info("🎯 Current date found in existing predictions - using full cache")
                                cached_result = {
                                    'success': True,
                                    'predictions': existing_predictions,
                                    'metadata': {
                                        'cache_source': 'extension_cache_full',
                                        'reused_predictions': len(existing_predictions),
                                        'target_date': current_date.strftime('%Y-%m-%d'),
                                        'extension_info': compatibility_info
                                    },
                                    'attention_data': None
                                }
                            else:
                                # Case 2: 정확한 날짜 없음 - 부분 재활용 + 새 예측 필요
                                logger.info("🔄 Current date not in existing predictions - attempting smart hybrid approach")
                                
                                # 기존 예측 중 유효한 부분만 재활용
                                useful_predictions = [p for p in existing_predictions 
                                                    if pd.to_datetime(p['Date']) <= existing_date_range[1]]
                                
                                if useful_predictions:
                                    logger.info(f"📊 Will reuse {len(useful_predictions)} existing predictions")
                                    logger.info(f"📅 Need new predictions from {existing_date_range[1].strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')}")
                                    
                                    # 부분 캐시 활용을 위한 특별 플래그 설정
                                    cached_result = {
                                        'success': 'partial',  # 부분 성공 표시
                                        'predictions': useful_predictions,
                                        'metadata': {
                                            'cache_source': 'extension_cache_partial',
                                            'reused_predictions': len(useful_predictions),
                                            'target_date': current_date.strftime('%Y-%m-%d'),
                                            'need_additional_prediction': True,
                                            'additional_start_date': existing_date_range[1].strftime('%Y-%m-%d'),
                                            'extension_info': compatibility_info
                                        },
                                        'attention_data': None
                                    }
                                    logger.info("🚀 Using partial extension cache - will supplement with new predictions")
                
                if cached_result:
                    logger.info(f"  📋 Final cache success status: {cached_result.get('success', False)}")
                else:
                    logger.info("  ❌ No cache result available")
                    
            except Exception as cache_check_error:
                logger.error(f"  ❌ Cache check failed with error: {str(cache_check_error)}")
                logger.error(f"  📝 Error traceback: {traceback.format_exc()}")
                cached_result = None
        else:
            logger.info("🆕 Cache disabled - running new prediction...")
            cached_result = None
            
        # 🔧 개선된 캐시 처리: 완전 캐시와 부분 캐시 모두 처리
        if cached_result and (cached_result.get('success') == True or cached_result.get('success') == 'partial'):
            
            if cached_result.get('success') == 'partial':
                logger.info("🔄 Found partial cache! Will reuse existing predictions and add new ones...")
                prediction_state['prediction_progress'] = 40
                
                # 부분 캐시 활용 로직
                try:
                    # 기존 예측 데이터 정리
                    existing_predictions = cached_result['predictions']
                    metadata = cached_result['metadata']
                    
                    logger.info(f"📊 Reusing {len(existing_predictions)} existing predictions")
                    
                    # 새로운 예측이 필요한 날짜 범위 계산
                    additional_start_date = pd.to_datetime(metadata['additional_start_date']) + pd.Timedelta(days=1)
                    
                    logger.info(f"🚀 Generating new predictions from {additional_start_date.strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')}")
                                        # 새로운 날짜 범위에 대해서만 예측 수행
                    prediction_state['prediction_progress'] = 50
                    
                    # 부분 예측을 위해 현재 날짜가 새로운 범위에 있는지 확인
                    days_gap = (current_date - additional_start_date).days
                    if days_gap > 0 and days_gap <= 23:  # 예측 윈도우 내에 있음
                        logger.info(f"📅 Performing partial prediction for {days_gap} days gap")
                        new_results = generate_predictions_with_save(
                            df, current_date, 
                            save_to_csv=save_to_csv, 
                            file_path=file_path
                        )
                        
                        # 새 예측에서 기존 날짜와 겹치는 부분 제거
                        if new_results and new_results.get('predictions'):
                            new_predictions_raw = new_results['predictions']
                            if isinstance(new_predictions_raw, list):
                                new_predictions = new_predictions_raw
                            else:
                                new_predictions = new_results.get('predictions_flat', [])
                            
                            # 중복 제거: 기존 날짜 이후의 예측만 사용
                            existing_latest_date = max(existing_dates)
                            filtered_new_predictions = []
                            for pred in new_predictions:
                                pred_date = pd.to_datetime(pred.get('date') or pred.get('Date'))
                                if pred_date > existing_latest_date:
                                    filtered_new_predictions.append(pred)
                            
                            logger.info(f"📊 Filtered {len(filtered_new_predictions)} new predictions (removed {len(new_predictions) - len(filtered_new_predictions)} duplicates)")
                            new_results['predictions'] = filtered_new_predictions
                            
                    else:
                        logger.warning(f"⚠️ Date gap too large ({days_gap} days), performing full prediction")
                        new_results = generate_predictions_with_save(
                            df, current_date, 
                            save_to_csv=save_to_csv, 
                            file_path=file_path
                        )
                    
                    prediction_state['prediction_progress'] = 70
                                        # 기존 예측과 새 예측 결합
                    if new_results and new_results.get('predictions'):
                        new_predictions = new_results['predictions']
                        if isinstance(new_predictions, list):
                            combined_predictions = existing_predictions + new_predictions
                        else:
                            combined_predictions = existing_predictions + new_results.get('predictions_flat', [])
                        
                        # 날짜순 정렬 및 중복 제거
                        combined_predictions.sort(key=lambda x: pd.to_datetime(x.get('date') or x.get('Date')))
                        
                        # 최종 중복 확인 (안전장치)
                        seen_dates = set()
                        unique_predictions = []
                        for pred in combined_predictions:
                            pred_date = pd.to_datetime(pred.get('date') or pred.get('Date')).strftime('%Y-%m-%d')
                            if pred_date not in seen_dates:
                                seen_dates.add(pred_date)
                                unique_predictions.append(pred)
                        
                        combined_predictions = unique_predictions
                        logger.info(f"✅ Combined {len(existing_predictions)} existing + {len(new_predictions)} new = {len(combined_predictions)} total predictions (after deduplication)")
                        
                        # 호환성 유지된 형태로 변환
                        compatible_predictions = convert_to_legacy_format(combined_predictions)
                        
                        # 결합된 결과로 상태 업데이트
                        prediction_state['latest_predictions'] = compatible_predictions
                        prediction_state['latest_attention_data'] = new_results.get('attention_data')
                        prediction_state['current_date'] = safe_serialize_value(new_results.get('current_date'))
                        prediction_state['selected_features'] = new_results.get('selected_features', [])
                        prediction_state['semimonthly_period'] = safe_serialize_value(new_results.get('semimonthly_period'))
                        prediction_state['next_semimonthly_period'] = safe_serialize_value(new_results.get('next_semimonthly_period'))
                        prediction_state['latest_interval_scores'] = new_results.get('interval_scores', {})
                        prediction_state['latest_metrics'] = new_results.get('metrics')
                        prediction_state['latest_plots'] = new_results.get('plots', {})
                        prediction_state['latest_ma_results'] = new_results.get('ma_results', {})
                        
                        # feature_importance 설정
                        if new_results.get('attention_data') and 'feature_importance' in new_results['attention_data']:
                            prediction_state['feature_importance'] = new_results['attention_data']['feature_importance']
                        else:
                            prediction_state['feature_importance'] = None
                        
                        prediction_state['prediction_progress'] = 100
                        prediction_state['is_predicting'] = False
                        logger.info("✅ Hybrid prediction (partial cache + new predictions) completed successfully!")
                        return
                    else:
                        logger.warning("⚠️ New prediction failed, falling back to full prediction...")
                        
                except Exception as partial_error:
                    logger.error(f"❌ Partial cache processing failed: {str(partial_error)}")
                    logger.info("🔄 Falling back to full prediction...")
                    
            else:
                logger.info("🎉 Found existing prediction! Loading from cache...")
                prediction_state['prediction_progress'] = 50
            
            try:
                    from app.prediction.metrics import calculate_moving_averages_with_history
                    # 캐시된 데이터 로드 및 정리
                    predictions = cached_result['predictions']
                    metadata = cached_result['metadata']
                    attention_data = cached_result.get('attention_data')
                    
                    # 데이터 정리 (JSON 안전성 보장)
                    cleaned_predictions = clean_cached_predictions(predictions)
                    
                    # 호환성 유지된 형태로 변환
                    compatible_predictions = convert_to_legacy_format(cleaned_predictions)
                    
                    # JSON 직렬화 테스트
                    try:
                        test_json = json.dumps(compatible_predictions[:1] if compatible_predictions else [])
                        logger.info("✅ JSON serialization test passed for cached data")
                    except Exception as json_error:
                        logger.error(f"❌ JSON serialization failed for cached data: {str(json_error)}")
                        raise Exception("Cached data serialization failed")
                    
                    # 구간 점수 처리 (JSON 안전)
                    interval_scores = metadata.get('interval_scores', {})
                    cleaned_interval_scores = {}
                    for key, value in interval_scores.items():
                        if isinstance(value, dict):
                            cleaned_score = {}
                            for k, v in value.items():
                                cleaned_score[k] = safe_serialize_value(v)
                            cleaned_interval_scores[key] = cleaned_score
                        else:
                            cleaned_interval_scores[key] = safe_serialize_value(value)
                    
                    # 이동평균 재계산
                    prediction_state['prediction_progress'] = 60
                    logger.info("Recalculating moving averages from cached data...")
                    historical_data = df[df.index <= current_date].copy()
                    ma_results = calculate_moving_averages_with_history(
                        cleaned_predictions, historical_data, target_col='MOPJ'
                    )
                    
                    # 시각화 재생성
                    prediction_state['prediction_progress'] = 70
                    logger.info("Regenerating visualizations from cached data...")
                    plots = regenerate_visualizations_from_cache(
                        cleaned_predictions, df, current_date, metadata
                    )
                    
                    # 메트릭 정리
                    metrics = metadata.get('metrics')
                    cleaned_metrics = {}
                    if metrics:
                        for key, value in metrics.items():
                            cleaned_metrics[key] = safe_serialize_value(value)
                    
                    # 어텐션 데이터 정리
                    cleaned_attention = None
                    logger.info(f"📊 [CACHE_ATTENTION] Processing attention data: available={bool(attention_data)}")
                    if attention_data:
                        logger.info(f"📊 [CACHE_ATTENTION] Original keys: {list(attention_data.keys())}")
                        
                        cleaned_attention = {}
                        for key, value in attention_data.items():
                            if key == 'image' and value:
                                cleaned_attention[key] = value  # base64 이미지는 그대로
                                logger.info(f"📊 [CACHE_ATTENTION] Image preserved (length: {len(value)})")
                            elif isinstance(value, dict):
                                cleaned_attention[key] = {}
                                for k, v in value.items():
                                    cleaned_attention[key][k] = safe_serialize_value(v)
                                logger.info(f"📊 [CACHE_ATTENTION] Dict '{key}' processed: {len(cleaned_attention[key])} items")
                            else:
                                cleaned_attention[key] = safe_serialize_value(value)
                                logger.info(f"📊 [CACHE_ATTENTION] Value '{key}' processed: {type(value)}")
                        
                        logger.info(f"📊 [CACHE_ATTENTION] Final cleaned keys: {list(cleaned_attention.keys())}")
                    else:
                        logger.warning(f"📊 [CACHE_ATTENTION] No attention data in cache result")
                    
                    # 상태 설정
                    prediction_state['latest_predictions'] = compatible_predictions
                    prediction_state['latest_attention_data'] = cleaned_attention
                    prediction_state['current_date'] = safe_serialize_value(metadata.get('prediction_start_date'))
                    prediction_state['selected_features'] = metadata.get('selected_features', [])
                    prediction_state['semimonthly_period'] = safe_serialize_value(metadata.get('semimonthly_period'))
                    prediction_state['next_semimonthly_period'] = safe_serialize_value(metadata.get('next_semimonthly_period'))
                    prediction_state['latest_interval_scores'] = cleaned_interval_scores
                    prediction_state['latest_metrics'] = cleaned_metrics
                    prediction_state['latest_plots'] = plots
                    prediction_state['latest_ma_results'] = ma_results
                    
                    # feature_importance 설정
                    if cleaned_attention and 'feature_importance' in cleaned_attention:
                        prediction_state['feature_importance'] = cleaned_attention['feature_importance']
                    else:
                        prediction_state['feature_importance'] = None
                    
                    prediction_state['prediction_progress'] = 100
                    prediction_state['is_predicting'] = False
                    logger.info("✅ Cache prediction completed successfully!")
                    return
                    
            except Exception as cache_error:
                logger.warning(f"⚠️  Cache processing failed: {str(cache_error)}")
                logger.info("🔄 Falling back to new prediction...")
        else:
            logger.info("  📋 No usable cache found - proceeding with new prediction")
        
        # 새로운 예측 수행
        logger.info(f"🤖 Running new prediction...")
        prediction_state['prediction_progress'] = 40
        
        # 예측 수행
        results = generate_predictions_with_save(df, current_date, save_to_csv=save_to_csv, file_path=file_path)
        prediction_state['prediction_progress'] = 80
        
        # 새로운 예측 결과 정리 (JSON 안전성 보장)
        if isinstance(results.get('predictions'), list):
            raw_predictions = results['predictions']
        else:
            raw_predictions = results.get('predictions_flat', [])
        
        # 호환성 유지된 형태로 변환
        compatible_predictions = convert_to_legacy_format(raw_predictions)
        
        # JSON 직렬화 테스트
        try:
            test_json = json.dumps(compatible_predictions[:1] if compatible_predictions else [])
            logger.info("✅ JSON serialization test passed for new prediction")
        except Exception as json_error:
            logger.error(f"❌ JSON serialization failed for new prediction: {str(json_error)}")
            # 데이터 추가 정리 시도
            for pred in compatible_predictions:
                for key, value in pred.items():
                    pred[key] = safe_serialize_value(value)
        
        # 상태 설정
        prediction_state['latest_predictions'] = compatible_predictions
        prediction_state['latest_attention_data'] = results.get('attention_data')
        prediction_state['current_date'] = safe_serialize_value(results.get('current_date'))
        prediction_state['selected_features'] = results.get('selected_features', [])
        prediction_state['semimonthly_period'] = safe_serialize_value(results.get('semimonthly_period'))
        prediction_state['next_semimonthly_period'] = safe_serialize_value(results.get('next_semimonthly_period'))
        prediction_state['latest_interval_scores'] = results.get('interval_scores', {})
        prediction_state['latest_metrics'] = results.get('metrics')
        prediction_state['latest_plots'] = results.get('plots', {})
        prediction_state['latest_ma_results'] = results.get('ma_results', {})
        
        # feature_importance 설정
        if results.get('attention_data') and 'feature_importance' in results['attention_data']:
            prediction_state['feature_importance'] = results['attention_data']['feature_importance']
        else:
            prediction_state['feature_importance'] = None
        
        # 저장
        if save_to_csv:
            from app.data.cache_manager import save_prediction_simple
            logger.info("💾 Saving prediction to cache...")
            save_result = save_prediction_simple(results, current_date)
            if save_result['success']:
                logger.info(f"✅ Cache saved successfully: {save_result.get('prediction_start_date')}")
            else:
                logger.warning(f"⚠️  Cache save failed: {save_result.get('error')}")
        
        prediction_state['prediction_progress'] = 100
        prediction_state['is_predicting'] = False
        logger.info("✅ New prediction completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error in compatible prediction: {str(e)}")
        logger.error(traceback.format_exc())
        prediction_state['error'] = str(e)
        prediction_state['is_predicting'] = False
        prediction_state['prediction_progress'] = 0

def generate_visualizations_realtime(predictions, df, current_date, metadata):
    """실시간으로 시각화 생성 (저장하지 않음)"""
    try:
        # DataFrame으로 변환
        sequence_df = pd.DataFrame(predictions)
        if 'Date' in sequence_df.columns:
            sequence_df['Date'] = pd.to_datetime(sequence_df['Date'])
        
        # 시작값 계산
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
        
        start_day_value = df.loc[current_date, 'MOPJ'] if current_date in df.index else None
        
        if start_day_value is not None:
            # 성능 메트릭 계산
            from app.prediction.metrics import compute_performance_metrics_improved
            from app.visualization.plotter import plot_prediction_basic, plot_moving_average_analysis
            from app.prediction.metrics import calculate_moving_averages_with_history
            metrics = compute_performance_metrics_improved(sequence_df, start_day_value)
            
            # 기본 그래프 생성 (메모리에만)
            _, basic_plot_img = plot_prediction_basic(
                sequence_df, 
                metadata.get('prediction_start_date', current_date),
                start_day_value,
                metrics['f1'],
                metrics['accuracy'], 
                metrics['mape'],
                metrics['weighted_score'],
                save_prefix=None  # 파일별 캐시 디렉토리 자동 사용
                )
                
            # 이동평균 계산 및 시각화
            historical_data = df[df.index <= current_date].copy()
            ma_results = calculate_moving_averages_with_history(predictions, historical_data, target_col='MOPJ')
            
            _, ma_plot_img = plot_moving_average_analysis(
                ma_results,
                metadata.get('prediction_start_date', current_date),
                save_prefix=None  # 파일별 캐시 디렉토리 자동 사용
            )
            
            # 상태에 저장
            prediction_state['latest_plots'] = {
                'basic_plot': {'file': None, 'image': basic_plot_img},
                'ma_plot': {'file': None, 'image': ma_plot_img}
            }
            prediction_state['latest_ma_results'] = ma_results
            prediction_state['latest_metrics'] = metrics
            
        else:
            logger.warning("Cannot generate visualizations: start day value not available")
            prediction_state['latest_plots'] = {
                'basic_plot': {'file': None, 'image': None},
                'ma_plot': {'file': None, 'image': None}
            }
            prediction_state['latest_ma_results'] = {}
            prediction_state['latest_metrics'] = {}
            
    except Exception as e:
        logger.error(f"Error generating realtime visualizations: {str(e)}")
        prediction_state['latest_plots'] = {
            'basic_plot': {'file': None, 'image': None},
            'ma_plot': {'file': None, 'image': None}
        }
        prediction_state['latest_ma_results'] = {}
        prediction_state['latest_metrics'] = {}

def regenerate_visualizations_from_cache(predictions, df, current_date, metadata):
    """
    캐시된 데이터로부터 시각화를 재생성하는 함수
    🔑 current_date를 전달하여 과거/미래 구분 시각화 생성
    """
    try:
        logger.info("🎨 Regenerating visualizations from cached data...")
        
        # DataFrame으로 변환 (안전한 방식)
        temp_df_for_plot = pd.DataFrame([
            {
                'Date': pd.to_datetime(item.get('Date') or item.get('date')),
                'Prediction': safe_serialize_value(item.get('Prediction') or item.get('prediction')),
                'Actual': safe_serialize_value(item.get('Actual') or item.get('actual'))
            } for item in predictions if item.get('Date') or item.get('date')
        ])
        
        logger.info(f"  📊 Plot data prepared: {len(temp_df_for_plot)} predictions")
        
        # current_date 처리
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
        
        # 시작값 계산
        start_day_value = None
        if current_date in df.index and not pd.isna(df.loc[current_date, 'MOPJ']):
            start_day_value = df.loc[current_date, 'MOPJ']
            logger.info(f"  📈 Start day value: {start_day_value:.2f}")
        else:
            logger.warning(f"  ⚠️  Start day value not available for {current_date}")
        
        # 메타데이터에서 메트릭 가져오기 (안전한 방식)
        metrics = metadata.get('metrics')
        if metrics:
            f1_score = safe_serialize_value(metrics.get('f1', 0.0))
            accuracy = safe_serialize_value(metrics.get('accuracy', 0.0))
            mape = safe_serialize_value(metrics.get('mape', 0.0))
            weighted_score = safe_serialize_value(metrics.get('weighted_score', 0.0))
            logger.info(f"  📊 Metrics loaded - F1: {f1_score:.3f}, Acc: {accuracy:.1f}%, MAPE: {mape:.1f}%")
        else:
            f1_score = accuracy = mape = weighted_score = 0.0
            logger.info("  ℹ️  No metrics available - using default values")
        
        plots = {
            'basic_plot': {'file': None, 'image': None},
            'ma_plot': {'file': None, 'image': None}
        }
        
        # 시각화 생성 (데이터가 충분한 경우만)
        if start_day_value is not None and not temp_df_for_plot.empty:
            from app.prediction.metrics import calculate_moving_averages_with_history
            logger.info("  🎨 Generating basic prediction plot...")
            
            # 예측 시작일 계산
            prediction_start_date = metadata.get('prediction_start_date')
            if isinstance(prediction_start_date, str):
                prediction_start_date = pd.to_datetime(prediction_start_date)
            elif prediction_start_date is None:
                # 메타데이터에 없으면 current_date 다음 영업일로 계산
                prediction_start_date = current_date + pd.Timedelta(days=1)
                while prediction_start_date.weekday() >= 5 or is_holiday(prediction_start_date):
                    prediction_start_date += pd.Timedelta(days=1)
                logger.info(f"  📅 Calculated prediction start date: {prediction_start_date}")
            
            # ✅ 핵심 수정: current_date 전달하여 과거/미래 구분 시각화
            basic_plot_file, basic_plot_img = plot_prediction_basic(
                temp_df_for_plot,
                prediction_start_date,
                start_day_value,
                f1_score,
                accuracy,
                mape,
                weighted_score,
                current_date=current_date,  # 🔑 핵심 수정: current_date 전달
                save_prefix=None,  # 파일별 캐시 디렉토리 자동 사용
                title_prefix="Cached Prediction Analysis"
            )
            
            if basic_plot_file:
                logger.info(f"  ✅ Basic plot generated: {basic_plot_file}")
            else:
                logger.warning("  ❌ Basic plot generation failed")
            
            # 이동평균 계산 및 시각화
            logger.info("  📈 Calculating moving averages...")
            historical_data = df[df.index <= current_date].copy()
            
            # 캐시된 예측 데이터를 이동평균 계산용으로 변환
            ma_input_data = []
            for pred in predictions:
                try:
                    ma_item = {
                        'Date': pd.to_datetime(pred.get('Date') or pred.get('date')),
                        'Prediction': safe_serialize_value(pred.get('Prediction') or pred.get('prediction')),
                        'Actual': safe_serialize_value(pred.get('Actual') or pred.get('actual'))
                    }
                    ma_input_data.append(ma_item)
                except Exception as e:
                    logger.warning(f"  ⚠️  Error processing MA data item: {str(e)}")
                    continue
            
            ma_results = calculate_moving_averages_with_history(
                ma_input_data, historical_data, target_col='MOPJ'
            )
            
            if ma_results:
                logger.info(f"  📊 MA calculated for {len(ma_results)} windows")
                
                # 이동평균 시각화
                ma_plot_file, ma_plot_img = plot_moving_average_analysis(
                    ma_results,
                    prediction_start_date,
                    save_prefix=None,  # 파일별 캐시 디렉토리 자동 사용
                    title_prefix="Cached Moving Average Analysis"
                )
                
                if ma_plot_file:
                    logger.info(f"  ✅ MA plot generated: {ma_plot_file}")
                else:
                    logger.warning("  ❌ MA plot generation failed")
            else:
                logger.warning("  ⚠️  Moving average calculation failed")
                ma_plot_file, ma_plot_img = None, None
            
            plots = {
                'basic_plot': {'file': basic_plot_file, 'image': basic_plot_img},
                'ma_plot': {'file': ma_plot_file, 'image': ma_plot_img}
            }
            
            logger.info("  ✅ Visualizations regenerated from cache successfully")
        else:
            if start_day_value is None:
                logger.warning("  ❌ Cannot regenerate visualizations: start day value not available")
            if temp_df_for_plot.empty:
                logger.warning("  ❌ Cannot regenerate visualizations: no prediction data")
        
        return plots
        
    except Exception as e:
        logger.error(f"❌ Error regenerating visualizations from cache: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'basic_plot': {'file': None, 'image': None},
            'ma_plot': {'file': None, 'image': None}
        }
    
def generate_accumulated_report():
    from app.prediction.metrics import decide_purchase_interval
    """누적 예측 결과 보고서 생성"""
    global prediction_state
    
    if not prediction_state['accumulated_predictions']:
        return None
    
    try:
        metrics = prediction_state['accumulated_metrics']
        all_preds = prediction_state['accumulated_predictions']
        
        # 보고서 파일 이름 생성 - 파일별 캐시 디렉토리 사용
        start_date = all_preds[0]['date']
        end_date = all_preds[-1]['date']
        try:
            cache_dirs = get_file_cache_dirs()
            report_dir = cache_dirs['predictions']
            report_filename = os.path.join(report_dir, f"accumulated_report_{start_date}_to_{end_date}.txt")
        except Exception as e:
            logger.warning(f"Could not get cache directories for accumulated report: {str(e)}")
            report_filename = f"accumulated_report_{start_date}_to_{end_date}.txt"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(f"===== Accumulated Prediction Report =====\n")
            f.write(f"Period: {start_date} to {end_date}\n")
            f.write(f"Total Predictions: {metrics['total_predictions']}\n\n")
            
            # 누적 성능 지표
            f.write("Average Performance Metrics:\n")
            f.write(f"- F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"- Direction Accuracy: {metrics['accuracy']:.2f}%\n")
            f.write(f"- MAPE: {metrics['mape']:.2f}%\n")
            f.write(f"- Weighted Score: {metrics['weighted_score']:.2f}%\n\n")
            
            # 날짜별 상세 정보
            f.write("Performance By Date:\n")
            for pred in all_preds:
                date = pred['date']
                m = pred['metrics']
                f.write(f"\n* {date}:\n")
                f.write(f"  - F1 Score: {m['f1']:.4f}\n")
                f.write(f"  - Accuracy: {m['accuracy']:.2f}%\n")
                f.write(f"  - MAPE: {m['mape']:.2f}%\n")
                f.write(f"  - Weighted Score: {m['weighted_score']:.2f}%\n")
                
                # 구매 구간 정보
                if pred['interval_scores']:
                    best_interval = decide_purchase_interval(pred['interval_scores'])
                    f.write("Best Purchase Interval:\n")
                    f.write(f"- Start Date: {best_interval['start_date']}\n")
                    f.write(f"- End Date: {best_interval['end_date']}\n")
                    f.write(f"- Duration: {best_interval['days']} days\n")
                    f.write(f"- Average Price: {best_interval['avg_price']:.2f}\n")
                    f.write(f"- Score: {best_interval['score']}\n")
                    f.write(f"- Selection Reason: {best_interval.get('selection_reason', '')}\n\n")
        
        return report_filename
    
    except Exception as e:
        logger.error(f"Error generating accumulated report: {str(e)}")
        return None
    
def background_varmax_prediction(file_path, current_date, pred_days, use_cache=True):
    """백그라운드에서 VARMAX 예측 작업을 수행하는 함수"""
    global prediction_state
    
    try:
        from app.utils.file_utils import set_seed
        from app.data.cache_manager import load_varmax_prediction, save_varmax_prediction
        from app.visualization.plotter import create_varmax_visualizations
        # 일관된 예측 결과를 위한 시드 고정
        set_seed()
        # 현재 파일 상태 업데이트
        prediction_state['current_file'] = file_path
        
        # 🔍 기존 저장된 예측 확인 (use_cache=True인 경우)
        if use_cache:
            logger.info(f"🔍 [VARMAX_CACHE] Checking for existing prediction for date: {current_date}")
            existing_prediction = load_varmax_prediction(current_date)
            
            if existing_prediction:
                logger.info(f"✅ [VARMAX_CACHE] Found existing VARMAX prediction for {current_date}")
                logger.info(f"🔍 [VARMAX_CACHE] Cached data keys: {list(existing_prediction.keys())}")
                logger.info(f"🔍 [VARMAX_CACHE] MA results available: {bool(existing_prediction.get('ma_results'))}")
                ma_results = existing_prediction.get('ma_results')
                if ma_results:
                    logger.info(f"🔍 [VARMAX_CACHE] MA results type: {type(ma_results)}")
                    if isinstance(ma_results, dict):
                        logger.info(f"🔍 [VARMAX_CACHE] MA results keys: {list(ma_results.keys())}")
                    else:
                        logger.warning(f"⚠️ [VARMAX_CACHE] MA results is not a dict: {type(ma_results)}")
                
                # 🔑 상태 복원 (순차적으로)
                logger.info(f"🔄 [VARMAX_CACHE] Restoring state from cached prediction...")
                
                # 기존 예측 결과를 상태에 로드 (안전한 타입 검사)
                prediction_state['varmax_predictions'] = existing_prediction.get('predictions', [])
                prediction_state['varmax_half_month_averages'] = existing_prediction.get('half_month_averages', [])
                prediction_state['varmax_metrics'] = existing_prediction.get('metrics', {})
                
                # MA results 안전한 로드
                ma_results = existing_prediction.get('ma_results', {})
                if isinstance(ma_results, dict):
                    prediction_state['varmax_ma_results'] = ma_results
                else:
                    logger.warning(f"⚠️ [VARMAX_CACHE] Invalid ma_results type: {type(ma_results)}, setting empty dict")
                    prediction_state['varmax_ma_results'] = {}
                
                prediction_state['varmax_selected_features'] = existing_prediction.get('selected_features', [])
                prediction_state['varmax_current_date'] = existing_prediction.get('current_date', current_date)
                prediction_state['varmax_model_info'] = existing_prediction.get('model_info', {})
                prediction_state['varmax_plots'] = existing_prediction.get('plots', {})
                
                # 즉시 완료 상태로 설정
                prediction_state['varmax_is_predicting'] = False
                prediction_state['varmax_prediction_progress'] = 100
                prediction_state['varmax_error'] = None
                
                logger.info(f"✅ [VARMAX_CACHE] State restoration completed")
                
                logger.info(f"✅ [VARMAX_CACHE] Successfully loaded existing prediction for {current_date}")
                logger.info(f"🔍 [VARMAX_CACHE] State restored - predictions: {len(prediction_state['varmax_predictions'])}, MA results: {len(prediction_state['varmax_ma_results'])}")
                
                # 🔍 최종 검증
                logger.info(f"🔍 [VARMAX_CACHE] Final verification - is_predicting: {prediction_state.get('varmax_is_predicting')}")
                logger.info(f"🔍 [VARMAX_CACHE] Final verification - predictions count: {len(prediction_state.get('varmax_predictions', []))}")
                logger.info(f"🔍 [VARMAX_CACHE] Final verification - ma_results count: {len(prediction_state.get('varmax_ma_results', {}))}")
                
                # 🛡️ 상태 안정화를 위한 짧은 대기
                time.sleep(1.0)
                
                logger.info(f"🎯 [VARMAX_CACHE] Cache loading process completed for {current_date}")
                return
        
        # 🚀 새로운 예측 수행
        logger.info(f"🚀 [VARMAX_NEW] Starting new VARMAX prediction for {current_date}")
        forecaster = VARMAXSemiMonthlyForecaster(file_path, pred_days=pred_days)
        prediction_state['varmax_is_predicting'] = True
        prediction_state['varmax_prediction_progress'] = 10
        prediction_state['varmax_prediction_start_time'] = time.time()  # VARMAX 시작 시간 기록
        prediction_state['varmax_error'] = None
        
        # VARMAX 예측 수행
        prediction_state['varmax_prediction_progress'] = 30
        logger.info(f"🔄 [VARMAX_NEW] Starting prediction generation (30% complete)")
        
        try:
            min_index = 1 # 임시 인덱스
            logger.info(f"🔄 [VARMAX_NEW] Calling generate_predictions_varmax with current_date={current_date}, var_num={min_index+2}")
            
            # 예측 진행률을 30%로 설정 (모델 초기화 완료)
            prediction_state['varmax_prediction_progress'] = 30

            mape_list=[]
            valid_indices = []
            for var_num in range(2,8):
                mape_value = forecaster.generate_variables_varmax(current_date, var_num)
                mape_list.append(mape_value)
                if mape_value is not None:
                    valid_indices.append(var_num - 2)  # 인덱스 조정
                logger.info(f"Var {var_num} model MAPE: {mape_value}")
            
            # None 값 필터링
            valid_mape_values = [mape for mape in mape_list if mape is not None]
            
            if not valid_mape_values:
                raise Exception("All VARMAX variable models failed to generate valid MAPE values")
            
            # 최소 MAPE 값의 인덱스 찾기
            min_mape = min(valid_mape_values)
            min_index = None
            for i, mape in enumerate(mape_list):
                if mape == min_mape:
                    min_index = i
                    break
            
            logger.info(f"Var {min_index+2} model is selected, MAPE:{mape_list[min_index]}%")
            logger.info(f"Valid models: {len(valid_mape_values)}/{len(mape_list)}")
            
            results = forecaster.generate_predictions_varmax(current_date, min_index+2)
            logger.info(f"✅ [VARMAX_NEW] Prediction generation completed successfully")
            
            # 최종 진행률 95%로 설정 (시각화 생성 전)
            prediction_state['varmax_prediction_progress'] = 95
            
        except Exception as prediction_error:
            logger.error(f"❌ [VARMAX_NEW] Error during prediction generation: {str(prediction_error)}")
            logger.error(f"❌ [VARMAX_NEW] Prediction error traceback: {traceback.format_exc()}")
            
            # 예측 실패 상태로 설정
            prediction_state['varmax_error'] = f"Prediction generation failed: {str(prediction_error)}"
            prediction_state['varmax_is_predicting'] = False
            prediction_state['varmax_prediction_progress'] = 0
            logger.error(f"❌ [VARMAX_NEW] Prediction state reset due to error")
            return
        
        if results['success']:
            logger.info(f"🔄 [VARMAX_NEW] Updating state with new prediction results...")
            
            # 상태에 결과 저장 (기존 LSTM 결과와 분리)
            prediction_state['varmax_predictions'] = results['predictions']
            prediction_state['varmax_half_month_averages'] = results.get('half_month_averages', [])
            prediction_state['varmax_metrics'] = results['metrics']
            prediction_state['varmax_ma_results'] = results['ma_results']
            prediction_state['varmax_selected_features'] = results['selected_features']
            prediction_state['varmax_current_date'] = results['current_date']
            prediction_state['varmax_model_info'] = results['model_info']
            
            # 시각화 생성 (기존 app.py 방식 활용)
            plots_info = create_varmax_visualizations(results)
            prediction_state['varmax_plots'] = plots_info
            
            prediction_state['varmax_prediction_progress'] = 100
            prediction_state['varmax_is_predicting'] = False
            prediction_state['varmax_error'] = None
            
            # VARMAX 예측 결과 저장
            save_varmax_prediction(results, current_date)
            
            logger.info("✅ [VARMAX_NEW] Prediction completed successfully")
            logger.info(f"🔍 [VARMAX_NEW] Final state - predictions: {len(prediction_state['varmax_predictions'])}")
        else:
            prediction_state['varmax_error'] = results['error']
            prediction_state['varmax_is_predicting'] = False
            prediction_state['varmax_prediction_progress'] = 0
            
    except Exception as e:
        logger.error(f"❌ [VARMAX_BG] Error in background VARMAX prediction: {str(e)}")
        logger.error(f"❌ [VARMAX_BG] Full traceback: {traceback.format_exc()}")
        
        # 에러 상태로 설정하고 자세한 로깅
        prediction_state['varmax_error'] = f"Background prediction failed: {str(e)}"
        prediction_state['varmax_is_predicting'] = False
        prediction_state['varmax_prediction_progress'] = 0
        
        logger.error(f"❌ [VARMAX_BG] VARMAX prediction failed completely. Current state reset.")
        logger.error(f"❌ [VARMAX_BG] Error type: {type(e).__name__}")
        logger.error(f"❌ [VARMAX_BG] Error details: {str(e)}")
        
        # 에러 발생 시 모든 VARMAX 관련 상태 초기화
        prediction_state['varmax_predictions'] = []
        prediction_state['varmax_metrics'] = {}
        prediction_state['varmax_ma_results'] = {}
        prediction_state['varmax_selected_features'] = []
        prediction_state['varmax_current_date'] = None
        prediction_state['varmax_model_info'] = {}
        prediction_state['varmax_plots'] = {}
        prediction_state['varmax_half_month_averages'] = []