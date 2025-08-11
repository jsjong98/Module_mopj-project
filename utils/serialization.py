import json
import numpy as np
import pandas as pd
import logging
import traceback

logger = logging.getLogger(__name__)

def safe_serialize_value(value):
    """값을 JSON 안전하게 직렬화 (NaN/Infinity 처리 강화)"""
    if value is None:
        return None
    
    # 🔧 CRITICAL: 모든 NaN 케이스를 먼저 체크
    try:
        # 1. 문자열로 변환된 NaN 체크
        if isinstance(value, str):
            value_lower = value.lower().strip()
            if value_lower in ['nan', 'inf', '-inf', 'infinity', '-infinity', 'null', 'none', '']:
                return None
            # 문자열이 숫자로 변환 가능한지 체크하여 NaN 탐지
            try:
                float_val = float(value)
                if np.isnan(float_val) or np.isinf(float_val):
                    return None
                return value  # 정상 문자열
            except (ValueError, TypeError):
                return value  # 숫자가 아닌 정상 문자열
        
        # 2. pandas isna 체크 (가장 포괄적)
        if pd.isna(value):
            return None
            
        # 3. NumPy NaN/Infinity 체크
        if isinstance(value, (int, float, np.number)):
            if np.isnan(value) or np.isinf(value):
                return None
            # 정상 숫자값인 경우
            if isinstance(value, (np.floating, float)):
                return float(value)
            elif isinstance(value, (np.integer, int)):
                return int(value)
        
        # 4. 파이썬 기본 타입 체크
        if isinstance(value, float):
            if value != value:  # NaN 체크 (NaN != NaN)
                return None
            if value == float('inf') or value == float('-inf'):
                return None
    except (TypeError, ValueError, OverflowError):
        pass
    
    # numpy/pandas 배열 타입 체크
    if isinstance(value, (np.ndarray, pd.Series, list)):
        if len(value) == 0:
            return []
        elif len(value) == 1:
            # 단일 원소 배열인 경우 스칼라로 처리
            return safe_serialize_value(value[0])
        else:
            # 다중 원소 배열인 경우 리스트로 변환
            try:
                return [safe_serialize_value(item) for item in value]
            except:
                return [str(item) for item in value]
    
    # 날짜 객체 처리
    if hasattr(value, 'isoformat'):  # datetime/Timestamp
        try:
            return value.strftime('%Y-%m-%d')
        except:
            return str(value)
    elif hasattr(value, 'strftime'):  # 기타 날짜 객체
        try:
            return value.strftime('%Y-%m-%d')
        except:
            return str(value)
    
    # 🔧 최종 JSON 직렬화 테스트 및 안전 처리
    try:
        # 먼저 JSON 직렬화 테스트
        json.dumps(value)
        return value
    except (TypeError, ValueError) as e:
        # JSON 직렬화 실패 시 안전한 문자열로 변환
        try:
            if hasattr(value, 'item'):  # numpy scalar
                return safe_serialize_value(value.item())
            elif hasattr(value, 'tolist'):  # numpy array
                return safe_serialize_value(value.tolist())
            else:
                return str(value)
        except:
            return str(value)

def clean_predictions_data(predictions):
    """예측 데이터를 JSON 안전하게 정리 - 강화된 NaN 처리"""
    if not predictions:
        return []
    
    cleaned = []
    for pred in predictions:
        cleaned_pred = {}
        for key, value in pred.items():
            if key in ['date', 'prediction_from']:
                # 날짜 필드
                if hasattr(value, 'strftime'):
                    cleaned_pred[key] = value.strftime('%Y-%m-%d')
                else:
                    cleaned_pred[key] = str(value)
            elif key in ['prediction', 'actual', 'error', 'error_pct']:
                # 🔧 CRITICAL: 숫자 필드 - actual 값 NaN 처리 강화
                safe_value = safe_serialize_value(value)
                if safe_value is None and key == 'actual':
                    # actual 값이 None이면 명시적으로 null로 설정
                    cleaned_pred[key] = None
                else:
                    cleaned_pred[key] = safe_value
            else:
                # 기타 필드
                cleaned_pred[key] = safe_serialize_value(value)
        
        # 🔧 JSON 직렬화 테스트 (각 예측 항목별로)
        try:
            json.dumps(cleaned_pred)
        except Exception as e:
            logger.warning(f"⚠️ JSON serialization failed for prediction item: {e}")
            # 추가 정리 시도
            for k, v in cleaned_pred.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    cleaned_pred[k] = None
                elif isinstance(v, str) and v.lower() in ['nan', 'inf', '-inf']:
                    cleaned_pred[k] = None
        
        cleaned.append(cleaned_pred)
    
    return cleaned

def clean_cached_predictions(predictions):
    """캐시에서 로드된 예측 데이터를 정리하는 함수"""
    cleaned_predictions = []
    
    for pred in predictions:
        try:
            # 모든 필드를 안전하게 처리
            cleaned_pred = {}
            for key, value in pred.items():
                if key in ['Date', 'date']:
                    # 날짜 필드 특별 처리
                    if pd.notna(value):
                        if hasattr(value, 'strftime'):
                            cleaned_pred[key] = value.strftime('%Y-%m-%d')
                        else:
                            cleaned_pred[key] = str(value)[:10]
                    else:
                        cleaned_pred[key] = None
                elif key in ['Prediction', 'prediction', 'Actual', 'actual']:
                    # 숫자 필드 처리
                    cleaned_pred[key] = safe_serialize_value(value)
                else:
                    # 기타 필드
                    cleaned_pred[key] = safe_serialize_value(value)
            
            cleaned_predictions.append(cleaned_pred)
            
        except Exception as e:
            logger.warning(f"Error cleaning prediction item: {str(e)}")
            continue
    
    return cleaned_predictions

def clean_interval_scores_safe(interval_scores):
    """구간 점수를 안전하게 정리하는 함수 - 강화된 오류 처리"""
    cleaned_interval_scores = []
    
    try:
        # 입력값 검증
        if interval_scores is None:
            logger.info("📋 interval_scores is None, returning empty list")
            return []
        
        if not isinstance(interval_scores, (dict, list)):
            logger.warning(f"⚠️ interval_scores is not dict or list: {type(interval_scores)}")
            return []
        
        if isinstance(interval_scores, dict):
            if not interval_scores:  # 빈 dict
                logger.info("📋 interval_scores is empty dict, returning empty list")
                return []
                
            for key, value in interval_scores.items():
                try:
                    if isinstance(value, dict):
                        cleaned_score = {}
                        for k, v in value.items():
                            try:
                                # 배열이나 복잡한 타입은 특별 처리
                                if isinstance(v, (np.ndarray, pd.Series, list)):
                                    if hasattr(v, '__len__') and len(v) == 1:
                                        cleaned_score[k] = safe_serialize_value(v[0])
                                    elif hasattr(v, '__len__') and len(v) == 0:
                                        cleaned_score[k] = None
                                    else:
                                        # 다중 원소 배열은 문자열로 변환
                                        cleaned_score[k] = str(v)
                                else:
                                    cleaned_score[k] = safe_serialize_value(v)
                            except Exception as inner_e:
                                logger.warning(f"⚠️ Error processing key {k}: {str(inner_e)}")
                                cleaned_score[k] = None
                        cleaned_interval_scores.append(cleaned_score)
                    else:
                        # dict가 아닌 경우 안전하게 처리
                        cleaned_interval_scores.append(safe_serialize_value(value))
                except Exception as value_e:
                    logger.warning(f"⚠️ Error processing interval_scores key {key}: {str(value_e)}")
                    continue
                    
        elif isinstance(interval_scores, list):
            if not interval_scores:  # 빈 list
                logger.info("📋 interval_scores is empty list, returning empty list")
                return []
                
            for i, score in enumerate(interval_scores):
                try:
                    if isinstance(score, dict):
                        cleaned_score = {}
                        for k, v in score.items():
                            try:
                                # 배열이나 복잡한 타입은 특별 처리
                                if isinstance(v, (np.ndarray, pd.Series, list)):
                                    if hasattr(v, '__len__') and len(v) == 1:
                                        cleaned_score[k] = safe_serialize_value(v[0])
                                    elif hasattr(v, '__len__') and len(v) == 0:
                                        cleaned_score[k] = None
                                    else:
                                        cleaned_score[k] = str(v)
                                else:
                                    cleaned_score[k] = safe_serialize_value(v)
                            except Exception as inner_e:
                                logger.warning(f"⚠️ Error processing score[{i}].{k}: {str(inner_e)}")
                                cleaned_score[k] = None
                        cleaned_interval_scores.append(cleaned_score)
                    else:
                        cleaned_interval_scores.append(safe_serialize_value(score))
                except Exception as score_e:
                    logger.warning(f"⚠️ Error processing interval_scores[{i}]: {str(score_e)}")
                    continue
        
        logger.info(f"✅ Successfully cleaned {len(cleaned_interval_scores)} interval scores")
        return cleaned_interval_scores
        
    except Exception as e:
        logger.error(f"❌ Critical error cleaning interval scores: {str(e)}")
        logger.error(traceback.format_exc())
        return []
    
def convert_to_legacy_format(predictions_data):
    """
    새·옛 구조를 모두 받아 프론트엔드(대문자) + 백엔드(소문자) 키를 동시 보존.
    JSON 직렬화 안전성 보장
    """
    if not predictions_data:
        return []
    
    legacy_out = []
    actual_values_found = 0  # 실제값이 발견된 수 카운트
    
    for i, pred in enumerate(predictions_data):
        try:
            # 날짜 필드 안전 처리
            date_value = pred.get("date") or pred.get("Date")
            if hasattr(date_value, 'strftime'):
                date_str = date_value.strftime('%Y-%m-%d')
            elif isinstance(date_value, str):
                date_str = date_value[:10] if len(date_value) > 10 else date_value
            else:
                date_str = str(date_value) if date_value is not None else None
            
            # 예측값 안전 처리
            prediction_value = pred.get("prediction") or pred.get("Prediction")
            prediction_safe = safe_serialize_value(prediction_value)
            
            # 실제값 안전 처리 - 다양한 필드명 확인
            actual_value = (pred.get("actual") or 
                          pred.get("Actual") or 
                          pred.get("actual_value") or 
                          pred.get("Actual_Value"))
            
            # 실제값이 있는지 확인
            if actual_value is not None and actual_value != 'None' and not (
                isinstance(actual_value, float) and (np.isnan(actual_value) or np.isinf(actual_value))
            ):
                actual_safe = safe_serialize_value(actual_value)
                actual_values_found += 1
                if i < 5:  # 처음 5개만 로깅
                    logger.debug(f"  📊 [LEGACY_FORMAT] Found actual value for {date_str}: {actual_safe}")
            else:
                actual_safe = None
            
            # 기타 필드들 안전 처리
            prediction_from = pred.get("prediction_from") or pred.get("Prediction_From")
            if hasattr(prediction_from, 'strftime'):
                prediction_from = prediction_from.strftime('%Y-%m-%d')
            elif prediction_from:
                prediction_from = str(prediction_from)
            
            legacy_item = {
                # ── 프론트엔드 호환 대문자 키 (JSON 안전) ───────────────
                "Date": date_str,
                "Prediction": prediction_safe,
                "Actual": actual_safe,

                # ── 백엔드 후속 함수(소문자 'date' 참조)용 ──
                "date": date_str,
                "prediction": prediction_safe,
                "actual": actual_safe,

                # 기타 필드 안전 처리
                "Prediction_From": prediction_from,
                "SemimonthlyPeriod": safe_serialize_value(pred.get("semimonthly_period")),
                "NextSemimonthlyPeriod": safe_serialize_value(pred.get("next_semimonthly_period")),
                "is_synthetic": bool(pred.get("is_synthetic", False)),
                
                # 추가 메타데이터 (있는 경우)
                "day_offset": safe_serialize_value(pred.get("day_offset")),
                "is_business_day": bool(pred.get("is_business_day", True)),
                "error": safe_serialize_value(pred.get("error")),
                "error_pct": safe_serialize_value(pred.get("error_pct"))
            }
            
            legacy_out.append(legacy_item)
            
        except Exception as e:
            logger.warning(f"Error converting prediction item {i}: {str(e)}")
            continue
    
    # 실제값 통계 로깅
    total_predictions = len(legacy_out)
    logger.info(f"  📊 [LEGACY_FORMAT] Converted {total_predictions} predictions, {actual_values_found} with actual values")
    
    return legacy_out