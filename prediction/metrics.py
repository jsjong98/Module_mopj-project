import numpy as np
import pandas as pd
import logging
import traceback
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

from app.utils.date_utils import format_date

logger = logging.getLogger(__name__)

def calculate_interval_averages_and_scores(predictions, business_days, min_window_size=5):
    """
    다음 반월 기간에 대해 다양한 크기의 구간별 평균 가격을 계산하고 점수를 부여하는 함수
    - 반월 전체 영업일 수에 맞춰 윈도우 크기 범위 조정
    - global_rank 방식: 모든 구간을 비교해 전역적으로 가장 저렴한 구간에 점수 부여
    
    Parameters:
    -----------
    predictions : list
        날짜별 예측 가격 정보 (딕셔너리 리스트)
    business_days : list
        다음 반월의 영업일 목록
    min_window_size : int
        최소 고려할 윈도우 크기 (기본값: 3)
    
    Returns:
    -----------
    tuple
        (구간별 평균 가격 정보, 구간별 점수 정보, 분석 추가 정보)
    """
    import numpy as np
    
    # 예측 데이터를 날짜별로 정리
    predictions_dict = {pred['Date']: pred['Prediction'] for pred in predictions if pred['Date'] in business_days}
    
    # 날짜 순으로 정렬된 영업일 목록
    sorted_days = sorted(business_days)
    
    # 다음 반월 총 영업일 수 계산
    total_days = len(sorted_days)
    
    # 최소 윈도우 크기와 최대 윈도우 크기 설정 (최대는 반월 전체 일수)
    max_window_size = total_days
    
    # 고려할 모든 윈도우 크기 범위 생성
    window_sizes = range(min_window_size, max_window_size + 1)
    
    print(f"다음 반월 영업일: {total_days}일, 고려할 윈도우 크기: {list(window_sizes)}")
    
    # 각 윈도우 크기별 결과 저장
    interval_averages = {}
    
    # 모든 구간을 저장할 리스트
    all_intervals = []
    
    # 각 윈도우 크기에 대해 모든 가능한 구간 계산
    for window_size in window_sizes:
        window_results = []
        
        # 가능한 모든 시작점에 대해 윈도우 평균 계산
        for i in range(len(sorted_days) - window_size + 1):
            interval_days = sorted_days[i:i+window_size]
            
            # 모든 날짜에 예측 가격이 있는지 확인
            if all(day in predictions_dict for day in interval_days):
                avg_price = np.mean([predictions_dict[day] for day in interval_days])
                
                interval_info = {
                    'start_date': interval_days[0],
                    'end_date': interval_days[-1],
                    'days': window_size,
                    'avg_price': avg_price,
                    'dates': interval_days.copy()
                }
                
                window_results.append(interval_info)
                all_intervals.append(interval_info)  # 모든 구간 목록에도 추가
        
        # 해당 윈도우 크기에 대한 결과 저장 (참고용)
        if window_results:
            # 평균 가격 기준으로 정렬
            window_results.sort(key=lambda x: x['avg_price'])
            interval_averages[window_size] = window_results
    
    # 구간 점수 계산을 위한 딕셔너리
    interval_scores = {}
    
    # Global Rank 전략: 모든 구간을 통합하여 가격 기준으로 정렬
    all_intervals.sort(key=lambda x: x['avg_price'])
    
    # 상위 3개 구간에만 점수 부여 (전체 중에서)
    for i, interval in enumerate(all_intervals[:min(3, len(all_intervals))]):
        score = 3 - i  # 1등: 3점, 2등: 2점, 3등: 1점
        
        # 구간 식별을 위한 키 생성 (문자열 키로 변경)
        interval_key = f"{format_date(interval['start_date'])}-{format_date(interval['end_date'])}"
        
        # 점수 정보 저장
        interval_scores[interval_key] = {
            'start_date': format_date(interval['start_date']),  # 형식 적용
            'end_date': format_date(interval['end_date']),      # 형식 적용
            'days': interval['days'],
            'avg_price': interval['avg_price'],
            'dates': [format_date(d) for d in interval['dates']],  # 날짜 목록도 형식 적용
            'score': score,
            'rank': i + 1
        }
    
    # 분석 정보 추가
    analysis_info = {
        'total_days': total_days,
        'window_sizes': list(window_sizes),
        'total_intervals': len(all_intervals),
        'min_avg_price': min([interval['avg_price'] for interval in all_intervals]) if all_intervals else None,
        'max_avg_price': max([interval['avg_price'] for interval in all_intervals]) if all_intervals else None
    }
    
    # 결과 출력 (참고용)
    if interval_scores:
        top_interval = max(interval_scores.values(), key=lambda x: x['score'])
        print(f"\n최고 점수 구간: {top_interval['days']}일 구간 ({format_date(top_interval['start_date'])} ~ {format_date(top_interval['end_date'])})")
        print(f"점수: {top_interval['score']}, 순위: {top_interval['rank']}, 평균가: {top_interval['avg_price']:.2f}")
    
    return interval_averages, interval_scores, analysis_info

def decide_purchase_interval(interval_scores):
    """
    점수가 부여된 구간들 중에서 최종 구매 구간을 결정하는 함수
    - 점수가 가장 높은 구간 선택
    - 동점인 경우 평균 가격이 더 낮은 구간 선택
    
    Parameters:
    -----------
    interval_scores : dict
        구간별 점수 정보
    
    Returns:
    -----------
    dict
        최종 선택된 구매 구간 정보
    """
    if not interval_scores:
        return None
    
    # 점수가 가장 높은 구간 선택
    max_score = max(interval['score'] for interval in interval_scores.values())
    
    # 최고 점수를 가진 모든 구간 찾기
    top_intervals = [interval for interval in interval_scores.values() 
                    if interval['score'] == max_score]
    
    # 동점이 있는 경우, 평균 가격이 더 낮은 구간 선택
    if len(top_intervals) > 1:
        best_interval = min(top_intervals, key=lambda x: x['avg_price'])
        best_interval['selection_reason'] = "최고 점수 중 최저 평균가 구간"
    else:
        best_interval = top_intervals[0]
        best_interval['selection_reason'] = "최고 점수 구간"
    
    return best_interval

def compute_performance_metrics_improved(validation_data, start_day_value):
    """
    검증 데이터만을 사용한 성능 지표 계산
    """
    try:
        # ✅ start_day_value가 비어있는 경우(None, NaN)를 먼저 확인합니다.
        if start_day_value is None or pd.isna(start_day_value):
            logger.warning("start_day_value가 유효하지 않아 메트릭 계산을 건너뜁니다.")
            return None # None을 반환하여 오류를 방지합니다.

        if not validation_data or len(validation_data) < 1:
            logger.info("No validation data available - this is normal for pure future predictions")
            return None
        
        # start_day_value 안전하게 처리
        if hasattr(start_day_value, 'iloc'):  # pandas Series/DataFrame인 경우
            start_val = float(start_day_value.iloc[0] if len(start_day_value) > 0 else start_day_value)
        elif hasattr(start_day_value, 'item'):  # numpy scalar인 경우
            start_val = float(start_day_value.item())
        else:
            start_val = float(start_day_value)
        
        # 검증 데이터에서 값 추출 (DataFrame/Series를 numpy로 안전하게 변환)
        actual_vals = [start_val]
        pred_vals = [start_val]
        
        for item in validation_data:
            # actual 값 안전하게 추출
            actual_val = item['actual']
            if hasattr(actual_val, 'iloc'):  # pandas Series/DataFrame인 경우
                actual_val = float(actual_val.iloc[0] if len(actual_val) > 0 else actual_val)
            elif hasattr(actual_val, 'item'):  # numpy scalar인 경우
                actual_val = float(actual_val.item())
            else:
                actual_val = float(actual_val)
            actual_vals.append(actual_val)
            
            # prediction 값 안전하게 추출
            pred_val = item['prediction']
            if hasattr(pred_val, 'iloc'):  # pandas Series/DataFrame인 경우
                pred_val = float(pred_val.iloc[0] if len(pred_val) > 0 else pred_val)
            elif hasattr(pred_val, 'item'):  # numpy scalar인 경우
                pred_val = float(pred_val.item())
            else:
                pred_val = float(pred_val)
            pred_vals.append(pred_val)
        
        # F1 점수 계산 (각 단계별 로깅 추가)
        try:
            f1, f1_report = calculate_f1_score(actual_vals, pred_vals)
        except Exception as e:
            logger.error(f"Error in F1 score calculation: {str(e)}")
            f1, f1_report = 0.0, "Error in F1 calculation"
            
        try:
            direction_accuracy = calculate_direction_accuracy(actual_vals, pred_vals)
        except Exception as e:
            logger.error(f"Error in direction accuracy calculation: {str(e)}")
            direction_accuracy = 0.0
            
        try:
            weighted_score, max_score = calculate_direction_weighted_score(actual_vals[1:], pred_vals[1:])
            weighted_score_pct = (weighted_score / max_score) * 100 if max_score > 0 else 0.0
        except Exception as e:
            logger.error(f"Error in weighted score calculation: {str(e)}")
            weighted_score_pct = 0.0
            
        try:
            mape = calculate_mape(actual_vals[1:], pred_vals[1:])
        except Exception as e:
            logger.error(f"Error in MAPE calculation: {str(e)}")
            mape = 0.0
        
        # 코사인 유사도
        cosine_similarity = None
        try:
            if len(actual_vals) > 1:
                # numpy 배열로 변환하여 안전하게 처리
                actual_vals_arr = np.array(actual_vals, dtype=float)
                pred_vals_arr = np.array(pred_vals, dtype=float)
                
                diff_actual = np.diff(actual_vals_arr)
                diff_pred = np.diff(pred_vals_arr)
                norm_actual = np.linalg.norm(diff_actual)
                norm_pred = np.linalg.norm(diff_pred)
                if norm_actual > 0 and norm_pred > 0:
                    cosine_similarity = np.dot(diff_actual, diff_pred) / (norm_actual * norm_pred)
        except Exception as e:
            logger.error(f"Error in cosine similarity calculation: {str(e)}")
            cosine_similarity = None
        
        return {
            'f1': float(f1),
            'accuracy': float(direction_accuracy),
            'mape': float(mape),
            'weighted_score': float(weighted_score_pct),
            'cosine_similarity': float(cosine_similarity) if cosine_similarity is not None else None,
            'f1_report': f1_report,
            'validation_points': len(validation_data)
        }
        
    except Exception as e:
        logger.error(f"Error computing improved metrics: {str(e)}")
        return None

def calculate_f1_score(actual, predicted):
    """방향성 예측의 F1 점수 계산"""
    # 입력을 numpy 배열로 변환
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    actual_directions = np.sign(np.diff(actual))
    predicted_directions = np.sign(np.diff(predicted))

    if len(actual_directions) < 2:
        return 0.0, "Insufficient data for classification report"
        
    try:
        # zero_division=0 파라미터 추가
        f1 = f1_score(actual_directions, predicted_directions, average='macro', zero_division=0)
        report = classification_report(actual_directions, predicted_directions, 
                                    digits=2, zero_division=0)
    except Exception as e:
        logger.error(f"Error in calculating F1 score: {str(e)}")
        return 0.0, "Error in calculation"
        
    return f1, report

def calculate_direction_accuracy(actual, predicted):
    """등락 방향 예측의 정확도 계산"""
    if len(actual) <= 1:
        return 0.0

    try:
        # 입력을 numpy 배열로 변환
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        actual_directions = np.sign(np.diff(actual))
        predicted_directions = np.sign(np.diff(predicted))
        
        correct_predictions = np.sum(actual_directions == predicted_directions)
        total_predictions = len(actual_directions)
        
        accuracy = (correct_predictions / total_predictions) * 100
        return accuracy
    except Exception as e:
        logger.error(f"Error in calculating direction accuracy: {str(e)}")
        return 0.0
    
def calculate_direction_weighted_score(actual, predicted):
    """변화율 기반의 가중 점수 계산"""
    if len(actual) <= 1:
        return 0.0, 1.0
        
    try:
        # 입력을 numpy 배열로 변환
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        actual_changes = 100 * np.diff(actual) / actual[:-1]
        predicted_changes = 100 * np.diff(predicted) / predicted[:-1]

        def assign_class(change):
            if change > 6:
                return 1
            elif 4 < change <= 6:
                return 2
            elif 2 < change <= 4:
                return 3
            elif -2 <= change <= 2:
                return 4
            elif -4 <= change < -2:
                return 5
            elif -6 <= change < -4:
                return 6
            else:
                return 7

        actual_classes = np.array([assign_class(x) for x in actual_changes])
        predicted_classes = np.array([assign_class(x) for x in predicted_changes])

        score = 0
        for ac, pc in zip(actual_classes, predicted_classes):
            diff = abs(ac - pc)
            score += max(0, 3 - diff)

        max_score = 3 * len(actual_classes)
        return score, max_score
    except Exception as e:
        logger.error(f"Error in calculating weighted score: {str(e)}")
        return 0.0, 1.0

def calculate_mape(actual, predicted):
    """MAPE 계산 함수"""
    try:
        # 입력을 numpy 배열로 변환
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        if len(actual) == 0:
            return 0.0
        # inf 방지를 위해 0이 아닌 값만 사용
        mask = actual != 0
        if not np.any(mask):  # any() 대신 np.any() 사용
            return 0.0
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    except Exception as e:
        logger.error(f"Error in MAPE calculation: {str(e)}")
        return 0.0

def calculate_moving_averages_with_history(predictions, historical_data, target_col='MOPJ', windows=[5, 10, 23]):
    """예측 데이터와 과거 데이터를 모두 활용한 이동평균 계산"""
    try:
        # 입력 데이터 검증
        if not predictions or len(predictions) == 0:
            logger.warning("No predictions provided for moving average calculation")
            return {}
            
        if historical_data is None or historical_data.empty:
            logger.warning("No historical data provided for moving average calculation")
            return {}
            
        if target_col not in historical_data.columns:
            logger.warning(f"Target column {target_col} not found in historical data")
            return {}
        
        results = {}
        
        # 예측 데이터를 DataFrame으로 변환 및 정렬
        try:
            pred_df = pd.DataFrame(predictions) if not isinstance(predictions, pd.DataFrame) else predictions.copy()
            
            # Date 컬럼 검증 (대소문자 모두 지원)
            date_col = None
            if 'Date' in pred_df.columns:
                date_col = 'Date'
            elif 'date' in pred_df.columns:
                date_col = 'date'
            else:
                logger.error("Date column not found in predictions (checked both 'Date' and 'date')")
                return {}
                
            pred_df['Date'] = pd.to_datetime(pred_df[date_col])
            pred_df = pred_df.sort_values('Date')
            
            # Prediction 컬럼 검증 (대소문자 모두 지원)
            prediction_col = None
            if 'Prediction' in pred_df.columns:
                prediction_col = 'Prediction'
            elif 'prediction' in pred_df.columns:
                prediction_col = 'prediction'
            else:
                logger.error("Prediction column not found in predictions (checked both 'Prediction' and 'prediction')")
                return {}
                
            # Prediction 컬럼을 표준화
            if prediction_col != 'Prediction':
                pred_df['Prediction'] = pred_df[prediction_col]
                
        except Exception as e:
            logger.error(f"Error processing prediction data: {str(e)}")
            return {}
        
        # 예측 시작일 확인
        prediction_start_date = pred_df['Date'].min()
        logger.info(f"MA calculation - prediction start date: {prediction_start_date}")
        
        # 과거 데이터에서 타겟 열 추출 (예측 시작일 이전)
        historical_series = pd.Series(
            data=historical_data.loc[historical_data.index < prediction_start_date, target_col],
            index=historical_data.loc[historical_data.index < prediction_start_date].index
        )
        
        # 최근 30일만 사용 (이동평균 계산에 충분)
        historical_series = historical_series.sort_index().tail(30)
        
        # 예측 데이터에서 시리즈 생성
        prediction_series = pd.Series(
            data=pred_df['Prediction'].values,
            index=pred_df['Date']
        )
        
        # 과거와 예측 데이터 결합
        combined_series = pd.concat([historical_series, prediction_series])
        combined_series = combined_series[~combined_series.index.duplicated(keep='last')]
        combined_series = combined_series.sort_index()
        
        logger.info(f"Combined series for MA: {len(combined_series)} data points "
                   f"({len(historical_series)} historical, {len(prediction_series)} predicted)")
        
        # 각 윈도우 크기별 이동평균 계산
        for window in windows:
            # 전체 데이터에 대해 이동평균 계산
            rolling_avg = combined_series.rolling(window=window, min_periods=1).mean()
            
            # 예측 기간에 해당하는 부분만 추출
            window_results = []
            
            for i, date in enumerate(pred_df['Date']):
                # 해당 날짜의 예측 및 실제값
                pred_value = pred_df['Prediction'].iloc[i]
                actual_value = pred_df['Actual'].iloc[i] if 'Actual' in pred_df.columns else None
                
                # 해당 날짜의 이동평균 값
                ma_value = rolling_avg.loc[date] if date in rolling_avg.index else None
                
                # NaN 값 처리
                if pd.isna(pred_value) or np.isnan(pred_value) or np.isinf(pred_value):
                    pred_value = None
                if pd.isna(actual_value) or np.isnan(actual_value) or np.isinf(actual_value):
                    actual_value = None
                if pd.isna(ma_value) or np.isnan(ma_value) or np.isinf(ma_value):
                    ma_value = None
                
                window_results.append({
                    'date': date,
                    'prediction': pred_value,
                    'actual': actual_value,
                    'ma': ma_value
                })
            
            results[f'ma{window}'] = window_results
            logger.info(f"MA{window} calculated: {len(window_results)} data points")
        
        logger.info(f"Moving average calculation completed with {len(results)} windows")
        return results
        
    except Exception as e:
        logger.error(f"Error calculating moving averages with history: {str(e)}")
        logger.error(traceback.format_exc())
        return {}
    
def calculate_prediction_consistency(accumulated_predictions, target_period):
    """
    다음 반월에 대한 여러 날짜의 예측 일관성을 계산
    
    Parameters:
    -----------
    accumulated_predictions: list
        여러 날짜에 수행한 예측 결과 목록
    target_period: str
        다음 반월 기간 (예: "2025-01-SM1")
    
    Returns:
    -----------
    dict: 일관성 점수와 관련 메트릭
    """
    import numpy as np
    
    # 날짜별 예측 데이터 추출
    period_predictions = {}
    
    for prediction in accumulated_predictions:
        # 안전한 데이터 접근
        if not isinstance(prediction, dict):
            continue
            
        prediction_date = prediction.get('date')
        next_period = prediction.get('next_semimonthly_period')
        predictions_list = prediction.get('predictions', [])
        
        if next_period != target_period:
            continue
            
        if prediction_date not in period_predictions:
            period_predictions[prediction_date] = []
        
        # predictions_list가 배열인지 확인
        if not isinstance(predictions_list, list):
            logger.warning(f"predictions_list is not a list for {prediction_date}: {type(predictions_list)}")
            continue
            
        for pred in predictions_list:
            # pred가 딕셔너리인지 확인
            if not isinstance(pred, dict):
                logger.warning(f"Prediction item is not a dict for {prediction_date}: {type(pred)}")
                continue
                
            pred_date = pred.get('Date') or pred.get('date')
            pred_value = pred.get('Prediction') or pred.get('prediction')
            
            # 값이 유효한지 확인
            if pred_date and pred_value is not None:
                period_predictions[prediction_date].append({
                    'date': pred_date,
                    'value': pred_value
                })
    
    # 날짜별로 정렬
    prediction_dates = sorted(period_predictions.keys())
    
    if len(prediction_dates) < 2:
        return {
            "consistency_score": None,
            "message": "Insufficient prediction data (min 2 required)",
            "period": target_period,
            "dates_count": len(prediction_dates)
        }
    
    # 일관성 분석을 위한 날짜 매핑
    date_predictions = {}
    
    for pred_date in prediction_dates:
        for p in period_predictions[pred_date]:
            target_date = p['date']
            if target_date not in date_predictions:
                date_predictions[target_date] = []
            
            date_predictions[target_date].append({
                'prediction_date': pred_date,
                'value': p['value']
            })
    
    # 각 타겟 날짜별 예측값 변동성 계산
    overall_variations = []
    
    for target_date, predictions in date_predictions.items():
        if len(predictions) >= 2:
            # 예측값 추출 (None 값 필터링)
            values = [p['value'] for p in predictions if p['value'] is not None]
            
            if len(values) < 2:
                continue
                
            # 값이 모두 같은 경우 CV를 0으로 처리
            if all(v == values[0] for v in values):
                cv = 0.0
                overall_variations.append(cv)
                continue
            
            mean_value = np.mean(values)
            std_value = np.std(values)
            
            # 변동 계수 (Coefficient of Variation)
            cv = std_value / abs(mean_value) if mean_value != 0 else float('inf')
            overall_variations.append(cv)
    
    # 전체 일관성 점수 계산 (변동 계수 평균을 0-100 점수로 변환)
    if overall_variations:
        avg_cv = np.mean(overall_variations)
        consistency_score = max(0, min(100, 100 - (avg_cv * 100)))
    else:
        consistency_score = None
    
    # 신뢰도 등급 부여
    if consistency_score is not None:
        if consistency_score >= 90:
            grade = "Very High"
        elif consistency_score >= 75:
            grade = "High"
        elif consistency_score >= 60:
            grade = "Medium"
        elif consistency_score >= 40:
            grade = "Low"
        else:
            grade = "Very Low"
    else:
        grade = "Unable to determine"
    
    return {
        "consistency_score": consistency_score,
        "consistency_grade": grade,
        "target_period": target_period,
        "prediction_count": len(prediction_dates),
        "average_variation": avg_cv * 100 if overall_variations else None,
        "message": f"Consistency for period {target_period} based on {len(prediction_dates)} predictions"
    }

# 누적 예측의 구매 신뢰도 계산 함수 (올바른 버전)
def calculate_accumulated_purchase_reliability(accumulated_predictions):
    """
    누적 예측의 구매 신뢰도 계산 (올바른 방식)
    
    각 예측마다 상위 3개 구간(1등:3점, 2등:2점, 3등:1점)을 선정하고,
    같은 구간이 여러 예측에서 선택되면 점수를 누적하여,
    최고 누적 점수 구간의 점수 / (예측 횟수 × 3점) × 100%로 계산
    
    Returns:
        tuple: (reliability_percentage, debug_info)
    """
    print(f"🔍 [RELIABILITY] Function called with {len(accumulated_predictions) if accumulated_predictions else 0} predictions")
    
    if not accumulated_predictions or not isinstance(accumulated_predictions, list):
        print(f"⚠️ [RELIABILITY] Invalid input: accumulated_predictions is empty or not a list")
        return 0.0, {}
    
    try:
        prediction_count = len(accumulated_predictions)
        print(f"📊 [RELIABILITY] Processing {prediction_count} predictions...")
        
        # 🔑 구간별 누적 점수를 저장할 딕셔너리
        interval_accumulated_scores = {}
        
        for i, pred in enumerate(accumulated_predictions):
            if not isinstance(pred, dict):
                continue
                
            interval_scores = pred.get('interval_scores', {})
            pred_date = pred.get('date')
            
            if interval_scores and isinstance(interval_scores, dict):
                # 모든 구간을 평균 가격 순으로 정렬 (가격이 낮을수록 좋음)
                valid_intervals = []
                for score_data in interval_scores.values():
                    if isinstance(score_data, dict) and 'avg_price' in score_data:
                        # 🔧 NaN 값 처리 강화
                        avg_price = score_data.get('avg_price', 0)
                        if pd.isna(avg_price) or np.isnan(avg_price) or np.isinf(avg_price):
                            avg_price = float('inf')  # NaN인 경우 최후순위로 설정
                            score_data['avg_price'] = avg_price
                        valid_intervals.append(score_data)
                
                if valid_intervals:
                    # 평균 가격 기준으로 정렬 (낮은 가격이 우선)
                    valid_intervals.sort(key=lambda x: x.get('avg_price', float('inf')))
                    
                    # 상위 3개 구간에 점수 부여
                    for rank, interval in enumerate(valid_intervals[:3]):
                        score = 3 - rank  # 1등: 3점, 2등: 2점, 3등: 1점
                        
                        # 구간 식별키 생성 (시작일-종료일)
                        interval_key = f"{interval.get('start_date')} ~ {interval.get('end_date')} ({interval.get('days')}일)"
                        
                        # 누적 점수 계산
                        if interval_key not in interval_accumulated_scores:
                            interval_accumulated_scores[interval_key] = {
                                'total_score': 0,
                                'appearances': 0,
                                'details': [],
                                'avg_price': interval.get('avg_price', 0),
                                'days': interval.get('days', 0)
                            }
                        
                        interval_accumulated_scores[interval_key]['total_score'] += score
                        interval_accumulated_scores[interval_key]['appearances'] += 1
                        interval_accumulated_scores[interval_key]['details'].append({
                            'date': pred_date,
                            'rank': rank + 1,
                            'score': score,
                            'avg_price': interval.get('avg_price', 0)
                        })
                        
                        print(f"📊 [RELIABILITY] 날짜 {pred_date}: {rank+1}등 {interval_key} → {score}점 (평균가: {interval.get('avg_price', 0):.2f})")
        
        # 최고 누적 점수 구간 찾기
        if interval_accumulated_scores:
            best_interval_key = max(interval_accumulated_scores.keys(), 
                                  key=lambda k: interval_accumulated_scores[k]['total_score'])
            best_total_score = interval_accumulated_scores[best_interval_key]['total_score']
            
            # 만점 계산 (각 예측마다 최대 3점씩)
            max_possible_total_score = prediction_count * 3
            
            # 구매 신뢰도 계산
            reliability_percentage = (best_total_score / max_possible_total_score) * 100 if max_possible_total_score > 0 else 0.0
            
            # 🔧 NaN 값 처리 강화
            if pd.isna(reliability_percentage) or np.isnan(reliability_percentage) or np.isinf(reliability_percentage):
                print(f"⚠️ [RELIABILITY] NaN/Inf detected in reliability calculation, setting to 0.0")
                reliability_percentage = 0.0
            
            print(f"\n🎯 [RELIABILITY] === 구간별 누적 점수 분석 ===")
            print(f"📊 예측 횟수: {prediction_count}개")
            print(f"📊 구간별 누적 점수:")
            
            # 누적 점수 순으로 정렬하여 표시
            sorted_intervals = sorted(interval_accumulated_scores.items(), 
                                    key=lambda x: x[1]['total_score'], reverse=True)
            
            for interval_key, data in sorted_intervals[:5]:  # 상위 5개만 표시
                print(f"   - {interval_key}: {data['total_score']}점 ({data['appearances']}회 선택)")
            
            print(f"\n🏆 최고 점수 구간: {best_interval_key}")
            print(f"🏆 최고 누적 점수: {best_total_score}점")
            print(f"🏆 구간 신뢰도: {best_total_score}/{max_possible_total_score} = {reliability_percentage:.1f}%")
            
            # 디버그 정보 생성
            debug_info = {
                'prediction_count': prediction_count,
                'interval_accumulated_scores': interval_accumulated_scores,
                'best_interval_key': best_interval_key,
                'best_total_score': best_total_score,
                'max_possible_total_score': max_possible_total_score,
                'reliability_percentage': reliability_percentage
            }
            
            return reliability_percentage, debug_info
        else:
            print(f"⚠️ [RELIABILITY] No valid interval scores found")
            return 0.0, {}
            
    except Exception as e:
        print(f"Error calculating accumulated purchase reliability: {str(e)}")
        return 0.0, {'error': str(e)} 

def calculate_accumulated_purchase_reliability_with_debug(accumulated_predictions):
    """
    디버그 정보와 함께 누적 예측의 구매 신뢰도 계산
    """
    if not accumulated_predictions or not isinstance(accumulated_predictions, list):
        return 0.0, {}
    
    debug_info = {
        'prediction_count': len(accumulated_predictions),
        'individual_scores': [],
        'total_best_score': 0,
        'max_possible_total_score': 0
    }
    
    try:
        total_best_score = 0
        prediction_count = len(accumulated_predictions)
        
        for i, pred in enumerate(accumulated_predictions):
            if not isinstance(pred, dict):
                continue
                
            pred_date = pred.get('date')
            interval_scores = pred.get('interval_scores', {})
            
            best_score = 0
            capped_score = 0  # ✅ 초기화 추가
            valid_scores = []  # ✅ valid_scores도 외부에서 초기화
            
            if interval_scores and isinstance(interval_scores, dict):
                # 유효한 interval score 찾기
                for score_data in interval_scores.values():
                    if isinstance(score_data, dict) and 'score' in score_data:
                        score_value = score_data.get('score', 0)
                        if isinstance(score_value, (int, float)):
                            valid_scores.append(score_value)
                
                if valid_scores:
                    best_score = max(valid_scores)
                    # 점수를 3점으로 제한 (각 예측의 최대 점수)
                    capped_score = min(best_score, 3.0)
                    total_best_score += capped_score
            
            debug_info['individual_scores'].append({
                'date': pred_date,
                'original_best_score': best_score,
                'actual_score_used': capped_score if valid_scores else 0,
                'max_score_per_prediction': 3,
                'has_valid_scores': len(valid_scores) > 0
            })
        
        # 전체 계산 - 3점이 만점
        max_possible_total_score = prediction_count * 3
        reliability_percentage = (total_best_score / max_possible_total_score) * 100 if max_possible_total_score > 0 else 0.0
        
        debug_info['total_best_score'] = total_best_score
        debug_info['max_possible_total_score'] = max_possible_total_score
        debug_info['reliability_percentage'] = reliability_percentage
        
        logger.info(f"🎯 올바른 누적 구매 신뢰도 계산:")
        logger.info(f"  - 예측 횟수: {prediction_count}회")
        
        # 🔍 개별 점수 디버깅 정보 출력
        for score_info in debug_info['individual_scores']:
            original = score_info.get('original_best_score', 0)
            actual = score_info.get('actual_score_used', 0)
            logger.info(f"📊 날짜 {score_info['date']}: 원본점수={original:.1f}, 적용점수={actual:.1f}, 유효점수있음={score_info['has_valid_scores']}")
        
        logger.info(f"  - 총 획득 점수: {total_best_score:.1f}점")
        logger.info(f"  - 최대 가능 점수: {max_possible_total_score}점 ({prediction_count} × 3)")
        logger.info(f"  - 구매 신뢰도: {reliability_percentage:.1f}%")
        
        # ✅ 추가 검증 로깅
        if reliability_percentage == 100.0:
            logger.warning("⚠️ [RELIABILITY] 구매 신뢰도가 100%입니다. 계산 검증:")
            logger.warning(f"   - total_best_score: {total_best_score}")
            logger.warning(f"   - max_possible_total_score: {max_possible_total_score}")
            logger.warning(f"   - prediction_count: {prediction_count}")
            for i, score_info in enumerate(debug_info['individual_scores']):
                logger.warning(f"   - 예측 {i+1}: {score_info}")
        elif reliability_percentage == 0.0:
            logger.warning("⚠️ [RELIABILITY] 구매 신뢰도가 0%입니다. 계산 검증:")
            logger.warning(f"   - total_best_score: {total_best_score}")
            logger.warning(f"   - max_possible_total_score: {max_possible_total_score}")
            logger.warning(f"   - prediction_count: {prediction_count}")
        
        return reliability_percentage, debug_info
            
    except Exception as e:
        logger.error(f"Error calculating accumulated purchase reliability: {str(e)}")
        return 0.0, {'error': str(e)}

def calculate_actual_business_days(predictions):
    """
    예측 결과에서 실제 영업일 수를 계산하는 헬퍼 함수
    """
    if not predictions:
        return 0
    
    try:
        actual_days = len([p for p in predictions 
                          if p.get('Date') and not p.get('is_synthetic', False)])
        return actual_days
    except Exception as e:
        logger.error(f"Error calculating actual business days: {str(e)}")
        return 0

def get_previous_semimonthly_period(semimonthly_period):
    """
    주어진 반월 기간의 이전 반월 기간을 계산하는 함수
    
    Parameters:
    -----------
    semimonthly_period : str
        "YYYY-MM-SM1" 또는 "YYYY-MM-SM2" 형식의 반월 기간
    
    Returns:
    --------
    str
        이전 반월 기간
    """
    parts = semimonthly_period.split('-')
    year = int(parts[0])
    month = int(parts[1])
    half = parts[2]
    
    if half == "SM1":
        # 상반월인 경우 이전 월의 하반월로
        if month == 1:
            return f"{year-1}-12-SM2"
        else:
            return f"{year}-{month-1:02d}-SM2"
    else:
        # 하반월인 경우 같은 월의 상반월로
        return f"{year}-{month:02d}-SM1"