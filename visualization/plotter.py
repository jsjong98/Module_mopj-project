import matplotlib
matplotlib.use('Agg') # 서버에서 GUI 백엔드 사용 안 함
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64
import os
import logging
from datetime import timedelta
import traceback
import logging

from app.data.cache_manager import get_file_cache_dirs
from app.utils.date_utils import format_date, is_holiday
from app.utils.serialization import safe_serialize_value

logger = logging.getLogger(__name__)

def get_global_y_range(original_df, test_dates, predict_window):
    """
    테스트 구간의 모든 MOPJ 값을 기반으로 전역 y축 범위를 계산합니다.
    
    Args:
        original_df: 원본 데이터프레임
        test_dates: 테스트 날짜 배열
        predict_window: 예측 기간
    
    Returns:
        tuple: (y_min, y_max) 전역 범위 값
    """
    # 테스트 구간 데이터 추출
    test_values = []
    
    # 테스트 데이터의 실제 값 수집
    for date in test_dates:
        if date in original_df.index and not pd.isna(original_df.loc[date, 'MOPJ']):
            test_values.append(original_df.loc[date, 'MOPJ'])
    
    # 안전장치: 데이터가 없으면 None 반환
    if not test_values:
        return None, None
    
    # 최소/최대 계산 (약간의 마진 추가)
    y_min = min(test_values) * 0.95
    y_max = max(test_values) * 1.05
    
    return y_min, y_max

def plot_prediction_basic(sequence_df, prediction_start_date, start_day_value, 
                         f1, accuracy, mape, weighted_score_pct, 
                         current_date=None,  # 🔑 추가: 데이터 컷오프 날짜
                         save_prefix=None, title_prefix="Basic Prediction Graph",
                         y_min=None, y_max=None, file_path=None):
    """
    기본 예측 그래프 시각화 - 과거/미래 명확 구분
    🔑 current_date 이후는 미래 예측으로만 표시 (데이터 누출 방지)
    """
    
    fig = None
    
    try:
        logger.info(f"Creating prediction graph for prediction starting {format_date(prediction_start_date)}")
        
        # 📁 저장 디렉토리 설정 (파일별 캐시 디렉토리 사용)
        if save_prefix is None:
            try:
                cache_dirs = get_file_cache_dirs(file_path)
                save_dir = cache_dirs['plots']
            except Exception as e:
                logger.warning(f"Could not get cache directories for plots: {str(e)}")
                save_dir = Path("temp_plots")
                save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = Path(save_prefix)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # DataFrame의 날짜 열이 문자열인 경우 날짜 객체로 변환
        if 'Date' in sequence_df.columns and isinstance(sequence_df['Date'].iloc[0], str):
            sequence_df['Date'] = pd.to_datetime(sequence_df['Date'])
        
        # ✅ current_date 기준으로 과거/미래 분할
        if current_date is not None:
            current_date = pd.to_datetime(current_date)
            
            # 과거 데이터 (current_date 이전): 실제값과 예측값 모두 표시 가능
            past_df = sequence_df[sequence_df['Date'] <= current_date].copy()
            # 미래 데이터 (current_date 이후): 예측값만 표시
            future_df = sequence_df[sequence_df['Date'] > current_date].copy()
            
            # 과거 데이터에서 실제값이 있는 것만 검증용으로 사용
            valid_df = past_df.dropna(subset=['Actual']) if 'Actual' in past_df.columns else pd.DataFrame()
            
            logger.info(f"  📊 Data split - Past: {len(past_df)}, Future: {len(future_df)}, Validation: {len(valid_df)}")
        else:
            # current_date가 없으면 기존 방식 사용 (하위 호환성)
            valid_df = sequence_df.dropna(subset=['Actual']) if 'Actual' in sequence_df.columns else pd.DataFrame()
            future_df = sequence_df
            past_df = valid_df
        
        pred_df = sequence_df.dropna(subset=['Prediction'])
        
        # 그래프 생성
        fig = plt.figure(figsize=(16, 9))
        plt.subplots_adjust(top=0.85)
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3, figure=fig)
        
        # 그래프 타이틀과 서브타이틀
        if isinstance(prediction_start_date, str):
            main_title = f"{title_prefix} - Start: {prediction_start_date}"
        else:
            main_title = f"{title_prefix} - Start: {prediction_start_date.strftime('%Y-%m-%d')}"
        
        # ✅ 과거/미래 구분 정보가 포함된 서브타이틀
        if current_date is not None:
            validation_count = len(valid_df)
            future_count = len(future_df)
            subtitle = f"Data Cutoff: {current_date.strftime('%Y-%m-%d')} | Validation: {validation_count} pts | Future: {future_count} pts"
            if validation_count > 0:
                subtitle += f" | F1: {f1:.3f}, Acc: {accuracy:.1f}%, MAPE: {mape:.1f}%"
        else:
            # 기존 방식
            if f1 == 0 and accuracy == 0 and mape == 0 and weighted_score_pct == 0:
                subtitle = "Future Prediction Only (No Validation Data Available)"
            else:
                subtitle = f"F1: {f1:.2f}, Acc: {accuracy:.2f}%, MAPE: {mape:.2f}%, Score: {weighted_score_pct:.2f}%"

        fig.text(0.5, 0.94, main_title, ha='center', fontsize=14, weight='bold')
        fig.text(0.5, 0.90, subtitle, ha='center', fontsize=12)
        
        # (1) 상단: 가격 예측 그래프
        ax1 = fig.add_subplot(gs[0])
        ax1.set_title("Price Prediction: Past Validation vs Future Forecast", fontsize=13)
        ax1.grid(True, linestyle='--', alpha=0.5)

        if y_min is not None and y_max is not None:
            ax1.set_ylim(y_min, y_max)
        
        # 예측 시작 날짜 처리
        if isinstance(prediction_start_date, str):
            start_date = pd.to_datetime(prediction_start_date)
        else:
            start_date = prediction_start_date
        
        # 시작일 이전 날짜 계산 (연결점용)
        prev_date = start_date - pd.Timedelta(days=1)
        while prev_date.weekday() >= 5 or is_holiday(prev_date):
            prev_date -= pd.Timedelta(days=1)
        
        # ✅ 1. 과거 실제값 (파란색 실선) - 가장 중요한 기준선
        if not valid_df.empty:
            real_dates = [prev_date] + valid_df['Date'].tolist()
            real_values = [start_day_value] + valid_df['Actual'].tolist()
            ax1.plot(real_dates, real_values, marker='o', color='blue', 
                    label='Actual (Past)', linewidth=2.5, markersize=5, zorder=3)
        
        # ✅ 2. 과거 예측값 (회색 점선) - 모델 성능 확인용
        if not valid_df.empty:
            past_pred_dates = [prev_date] + valid_df['Date'].tolist()
            past_pred_values = [start_day_value] + valid_df['Prediction'].tolist()
            ax1.plot(past_pred_dates, past_pred_values, marker='x', color='gray', 
                    label='Predicted (Past)', linewidth=1.5, linestyle=':', markersize=4, alpha=0.8, zorder=2)
        
        # ✅ 3. 미래 예측값 (빨간색 점선) - 핵심 예측
        if not future_df.empty:
            future_dates = future_df['Date'].tolist()
            future_values = future_df['Prediction'].tolist()
            
            # 연결선 (마지막 실제값 → 첫 미래 예측값)
            if not valid_df.empty and future_dates:
                # 마지막 검증 데이터의 실제값에서 첫 미래 예측으로 연결
                connection_x = [valid_df['Date'].iloc[-1], future_dates[0]]
                connection_y = [valid_df['Actual'].iloc[-1], future_values[0]]
                ax1.plot(connection_x, connection_y, '--', color='orange', alpha=0.6, linewidth=1.5, zorder=1)
            elif start_day_value is not None and future_dates:
                # 검증 데이터가 없으면 시작값에서 연결
                connection_x = [prev_date, future_dates[0]]
                connection_y = [start_day_value, future_values[0]]
                ax1.plot(connection_x, connection_y, '--', color='orange', alpha=0.6, linewidth=1.5, zorder=1)
            
            ax1.plot(future_dates, future_values, marker='o', color='red', 
                    label='Predicted (Future)', linewidth=2.5, linestyle='--', markersize=5, zorder=3)
        
        # ✅ 4. 데이터 컷오프 라인 (초록색 세로선)
        if current_date is not None:
            ax1.axvline(x=current_date, color='green', linestyle='-', alpha=0.8, 
                       linewidth=2.5, label=f'Data Cutoff', zorder=4)
            
            # 컷오프 날짜 텍스트 추가
            ax1.text(current_date, ax1.get_ylim()[1] * 0.95, 
                    f'{current_date.strftime("%m/%d")}', 
                    ha='center', va='top', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        else:
            # 예측 시작점에 수직선 표시 (기존 방식)
            ax1.axvline(x=start_date, color='green', linestyle='--', alpha=0.7, 
                       linewidth=2, label='Prediction Start', zorder=4)
        
        # ✅ 5. 배경 색칠 (방향성 일치 여부) - 검증 데이터만
        if not valid_df.empty and len(valid_df) > 1:
            for i in range(len(valid_df) - 1):
                curr_date = valid_df['Date'].iloc[i]
                next_date = valid_df['Date'].iloc[i + 1]
                
                curr_actual = valid_df['Actual'].iloc[i]
                next_actual = valid_df['Actual'].iloc[i + 1]
                curr_pred = valid_df['Prediction'].iloc[i]
                next_pred = valid_df['Prediction'].iloc[i + 1]
                
                # 방향 계산
                actual_dir = np.sign(next_actual - curr_actual)
                pred_dir = np.sign(next_pred - curr_pred)
                
                # 방향 일치 여부에 따른 색상
                color = 'lightblue' if actual_dir == pred_dir else 'lightcoral'
                ax1.axvspan(curr_date, next_date, color=color, alpha=0.15, zorder=0)
        
        ax1.set_xlabel("")
        ax1.set_ylabel("Price (USD/MT)", fontsize=11)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # ✅ (2) 하단: 오차 분석 - 검증 데이터만 또는 변화량
        ax2 = fig.add_subplot(gs[1])
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        if not valid_df.empty and len(valid_df) > 0:
            # 검증 데이터의 절대 오차
            error_dates = valid_df['Date'].tolist()
            error_values = [abs(row['Actual'] - row['Prediction']) for _, row in valid_df.iterrows()]
            
            if error_dates and error_values:
                bars = ax2.bar(error_dates, error_values, width=0.6, color='salmon', alpha=0.7, edgecolor='darkred', linewidth=0.5)
                ax2.set_title(f"Prediction Error - Validation Period ({len(error_dates)} points)", fontsize=11)
                
                # 평균 오차 라인
                avg_error = np.mean(error_values)
                ax2.axhline(y=avg_error, color='red', linestyle='--', alpha=0.8, 
                           label=f'Avg Error: {avg_error:.2f}')
                ax2.legend(fontsize=9)
            else:
                ax2.text(0.5, 0.5, "No validation errors to display", 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=10)
                ax2.set_title("Error Analysis")
        else:
            # 실제값이 없는 경우: 미래 예측의 일일 변화량 표시
            if not future_df.empty and len(future_df) > 1:
                change_dates = future_df['Date'].iloc[1:].tolist()
                change_values = np.diff(future_df['Prediction'].values)
                
                # 상승/하락에 따른 색상 구분
                colors = ['green' if change >= 0 else 'red' for change in change_values]
                
                bars = ax2.bar(change_dates, change_values, width=0.6, color=colors, alpha=0.7)
                ax2.set_title("Daily Price Changes - Future Predictions", fontsize=11)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
                
                # 범례 추가
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='green', alpha=0.7, label='Price Up'),
                                 Patch(facecolor='red', alpha=0.7, label='Price Down')]
                ax2.legend(handles=legend_elements, fontsize=9)
            else:
                ax2.text(0.5, 0.5, "Insufficient data for change analysis", 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=10)
                ax2.set_title("Change Analysis")
        
        ax2.set_xlabel("Date", fontsize=11)
        ax2.set_ylabel("Value", fontsize=11)
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 파일 경로 생성
        if isinstance(prediction_start_date, str):
            date_str = pd.to_datetime(prediction_start_date).strftime('%Y%m%d')
        else:
            date_str = prediction_start_date.strftime('%Y%m%d')
        
        filename = f"prediction_start_{date_str}.png"
        full_path = save_dir / filename
        
        # 이미지를 메모리에 저장
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
        img_buf.seek(0)
        
        # Base64로 인코딩
        img_str = base64.b64encode(img_buf.read()).decode('utf-8')
        
        # 파일로 저장
        plt.savefig(str(full_path), dpi=300, bbox_inches='tight')
        
        # 메모리 정리
        plt.close(fig)
        plt.clf()
        img_buf.close()
        
        logger.info(f"Enhanced prediction graph saved: {full_path}")
        logger.info(f"  - Past validation points: {len(valid_df) if not valid_df.empty else 0}")
        logger.info(f"  - Future prediction points: {len(future_df) if not future_df.empty else 0}")
        
        return str(full_path), img_str
        
    except Exception as e:
        if fig is not None:
            plt.close(fig)
        plt.close('all')
        plt.clf()
        
        logger.error(f"Error in enhanced graph creation: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None
    
def plot_moving_average_analysis(ma_results, sequence_start_date, save_prefix=None,
                               title_prefix="Moving Average Analysis", y_min=None, y_max=None, file_path=None):
    """이동평균 분석 시각화"""
    try:
        # 입력 데이터 검증
        if not ma_results or len(ma_results) == 0:
            logger.warning("No moving average results to plot")
            return None, None
            
        # ma_results 형식: {'ma5': [{'date': '...', 'prediction': X, 'actual': Y, 'ma': Z}, ...], 'ma10': [...]}
        windows = sorted(ma_results.keys())
        
        if len(windows) == 0:
            logger.warning("No moving average windows found")
            return None, None
        
        # 유효한 윈도우 필터링
        valid_windows = []
        for window_key in windows:
            if window_key in ma_results and ma_results[window_key] and len(ma_results[window_key]) > 0:
                valid_windows.append(window_key)
        
        if len(valid_windows) == 0:
            logger.warning("No valid moving average data found")
            return None, None
        
        fig = plt.figure(figsize=(12, max(4, 4 * len(valid_windows))))
        
        if isinstance(sequence_start_date, str):
            title = f"{title_prefix} Starting {sequence_start_date}"
        else:
            title = f"{title_prefix} Starting {sequence_start_date.strftime('%Y-%m-%d')}"
            
        fig.suptitle(title, fontsize=16)
        
        for idx, window_key in enumerate(valid_windows):
            window_num = window_key.replace('ma', '')
            ax = fig.add_subplot(len(valid_windows), 1, idx+1)
            
            window_data = ma_results[window_key]
            
            # 데이터 검증
            if not window_data or len(window_data) == 0:
                ax.text(0.5, 0.5, f"No data for {window_key}", 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # 날짜, 예측, 실제값, MA 추출
            dates = []
            predictions = []
            actuals = []
            ma_preds = []
            
            for item in window_data:
                try:
                    # 안전한 데이터 추출
                    if isinstance(item['date'], str):
                        dates.append(pd.to_datetime(item['date']))
                    else:
                        dates.append(item['date'])
                    
                    # None 값 처리
                    predictions.append(item.get('prediction', 0))
                    actuals.append(item.get('actual', None))
                    ma_preds.append(item.get('ma', None))
                except Exception as e:
                    logger.warning(f"Error processing MA data item: {str(e)}")
                    continue
            
            if len(dates) == 0:
                ax.text(0.5, 0.5, f"No valid data for {window_key}", 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # y축 범위 설정
            if y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)
            
            # 원본 실제값 vs 예측값 (옅게)
            ax.plot(dates, actuals, marker='o', color='blue', alpha=0.3, label='Actual')
            ax.plot(dates, predictions, marker='o', color='red', alpha=0.3, label='Predicted')
            
            # 이동평균
            # 실제값(actuals)과 이동평균(ma_preds) 모두 None이 아닌 인덱스를 선택
            valid_indices = [
                i for i in range(len(ma_preds))
                if (ma_preds[i] is not None and actuals[i] is not None)
            ]

            if valid_indices:
                valid_dates = [dates[i] for i in valid_indices]
                valid_ma = [ma_preds[i] for i in valid_indices]
                valid_actuals = [actuals[i] for i in valid_indices]
                
                # 배열로 변환
                valid_actuals_arr = np.array(valid_actuals)
                valid_ma_arr = np.array(valid_ma)
                
                # 실제값이 0인 항목은 제외하여 MAPE 계산
                non_zero_mask = valid_actuals_arr != 0
                if np.sum(non_zero_mask) > 0:
                    ma_mape = np.mean(np.abs((valid_actuals_arr[non_zero_mask] - valid_ma_arr[non_zero_mask]) /
                                           valid_actuals_arr[non_zero_mask])) * 100
                else:
                    ma_mape = 0.0
                
                ax.set_title(f"MA-{window_num} Analysis (MAPE: {ma_mape:.2f}%, Count: {len(valid_indices)})")
            else:
                ax.set_title(f"MA-{window_num} Analysis (Insufficient data)")
            
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
        
        plt.tight_layout()
        
        # 📁 저장 디렉토리 설정 (파일별 캐시 디렉토리 사용)
        if save_prefix is None:
            try:
                cache_dirs = get_file_cache_dirs(file_path)
                save_dir = cache_dirs['ma_plots']
            except Exception as e:
                logger.warning(f"Could not get cache directories for MA plots: {str(e)}")
                save_dir = Path("temp_ma_plots")
                save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = Path(save_prefix)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        if isinstance(sequence_start_date, str):
            date_str = pd.to_datetime(sequence_start_date).strftime('%Y%m%d')
        else:
            date_str = sequence_start_date.strftime('%Y%m%d')
            
        filename = save_dir / f"ma_analysis_{date_str}.png"
        
        # 이미지를 메모리에 저장
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=300)
        img_buf.seek(0)
        
        # Base64로 인코딩
        img_str = base64.b64encode(img_buf.read()).decode('utf-8')
        
        # 파일 저장
        plt.savefig(str(filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Moving Average graph saved: {filename}")
        return str(filename), img_str
        
    except Exception as e:
        logger.error(f"Error in moving average visualization: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None
    
def visualize_accumulated_metrics():
    """누적 예측 결과 시각화"""
    global prediction_state
    
    if not prediction_state['accumulated_predictions']:
        return None, None
    
    try:
        # 데이터 준비
        dates = []
        f1_scores = []
        accuracies = []
        mapes = []
        weighted_scores = []
        
        for pred in prediction_state['accumulated_predictions']:
            dates.append(pred['date'])
            m = pred['metrics']
            f1_scores.append(m['f1'])
            accuracies.append(m['accuracy'])
            mapes.append(m['mape'])
            weighted_scores.append(m['weighted_score'])
        
        # 날짜를 datetime으로 변환
        dates = [pd.to_datetime(d) for d in dates]
        
        # 그래프 생성
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Accumulated Prediction Metrics', fontsize=16)
        
        # F1 Score
        axs[0, 0].plot(dates, f1_scores, marker='o', color='blue')
        axs[0, 0].set_title('F1 Score')
        axs[0, 0].set_ylim(0, 1)
        axs[0, 0].grid(True)
        plt.setp(axs[0, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # Accuracy
        axs[0, 1].plot(dates, accuracies, marker='o', color='green')
        axs[0, 1].set_title('Direction Accuracy (%)')
        axs[0, 1].set_ylim(0, 100)
        axs[0, 1].grid(True)
        plt.setp(axs[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # MAPE
        axs[1, 0].plot(dates, mapes, marker='o', color='red')
        axs[1, 0].set_title('MAPE (%)')
        axs[1, 0].set_ylim(0, max(mapes) * 1.2)
        axs[1, 0].grid(True)
        plt.setp(axs[1, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # Weighted Score
        axs[1, 1].plot(dates, weighted_scores, marker='o', color='purple')
        axs[1, 1].set_title('Weighted Score (%)')
        axs[1, 1].set_ylim(0, 100)
        axs[1, 1].grid(True)
        plt.setp(axs[1, 1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # 이미지 저장
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=300)
        img_buf.seek(0)
        
        # Base64로 인코딩
        img_str = base64.b64encode(img_buf.read()).decode('utf-8')
        
        # 파일로 저장 - 파일별 캐시 디렉토리 사용
        try:
            cache_dirs = get_file_cache_dirs()
            plots_dir = cache_dirs['plots']
            filename = os.path.join(plots_dir, 'accumulated_metrics.png')
        except Exception as e:
            logger.warning(f"Could not get cache directories for accumulated metrics: {str(e)}")
            filename = 'accumulated_metrics.png'
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename, img_str
        
    except Exception as e:
        logger.error(f"Error visualizing accumulated metrics: {str(e)}")
        return None, None
    
def plot_varmax_prediction_basic(sequence_df, sequence_start_date, start_day_value, 
                                f1, accuracy, mape, weighted_score, 
                                save_prefix=None, title_prefix="VARMAX Semi-monthly Prediction", file_path=None):
    """VARMAX 기본 예측 그래프 시각화 (기존 plot_prediction_basic과 동일한 스타일)"""
    try:
        logger.info(f"Creating VARMAX prediction graph for {sequence_start_date}")
        
        # 파일별 캐시 디렉토리 가져오기
        if save_prefix is None:
            cache_dirs = get_file_cache_dirs(file_path)
            save_prefix = cache_dirs['plots']
        
        # 예측값만 있는 데이터 처리
        pred_df = sequence_df.dropna(subset=['Prediction'])
        valid_df = sequence_df.dropna(subset=['Actual']) if 'Actual' in sequence_df.columns else pd.DataFrame()
        
        # 그래프 생성
        fig = plt.figure(figsize=(16, 9))
        plt.subplots_adjust(top=0.85)
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3, figure=fig)
        
        # 제목 설정
        main_title = f"{title_prefix} - {sequence_start_date}"
        subtitle = f"F1: {f1:.2f}, Acc: {accuracy:.2f}%, MAPE: {mape:.2f}%, Score: {weighted_score:.2f}%"
        
        fig.text(0.5, 0.94, main_title, ha='center', fontsize=14, weight='bold')
        fig.text(0.5, 0.90, subtitle, ha='center', fontsize=12)
        
        # 상단: 예측 vs 실제 (있는 경우)
        ax1 = fig.add_subplot(gs[0])
        ax1.set_title("VARMAX Long-term Prediction")
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # 예측값 플롯
        ax1.plot(pred_df['Date'], pred_df['Prediction'],
                marker='o', color='red', label='VARMAX Predicted', linewidth=2)
        
        # 실제값 플롯 (있는 경우)
        if len(valid_df) > 0:
            ax1.plot(valid_df['Date'], valid_df['Actual'],
                    marker='o', color='blue', label='Actual', linewidth=2)
            
            # 방향성 일치 여부 배경 색칠
            for i in range(1, len(valid_df)):
                if i < len(pred_df):
                    actual_dir = np.sign(valid_df['Actual'].iloc[i] - valid_df['Actual'].iloc[i-1])
                    pred_dir = np.sign(pred_df['Prediction'].iloc[i] - pred_df['Prediction'].iloc[i-1])
                    color = 'blue' if actual_dir == pred_dir else 'red'
                    ax1.axvspan(valid_df['Date'].iloc[i-1], valid_df['Date'].iloc[i], alpha=0.1, color=color)
        
        ax1.set_xlabel("Date")
        ax1.set_ylabel("MOPJ Price")
        ax1.legend()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # 하단: 오차 (실제값이 있는 경우만)
        ax2 = fig.add_subplot(gs[1])
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        if len(valid_df) > 0:
            # 오차 계산 및 플롯
            errors = valid_df['Actual'] - valid_df['Prediction']
            ax2.bar(valid_df['Date'], errors, alpha=0.7, color='orange', width=0.8)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8)
            ax2.set_title(f"Prediction Error (MAE: {abs(errors).mean():.2f})")
        else:
            ax2.text(0.5, 0.5, 'No actual data for error calculation', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title("Prediction Error (No validation data)")
        
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Error")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # 파일 저장
        os.makedirs(save_prefix, exist_ok=True)
        filename = f"varmax_prediction_{sequence_start_date}.png"
        filepath = os.path.join(save_prefix, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"VARMAX prediction graph saved: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error creating VARMAX prediction graph: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def create_varmax_visualizations(results):
    """VARMAX 결과에 대한 시각화 생성"""
    try:
        # 기본 예측 그래프
        sequence_df = pd.DataFrame(results['predictions'])
        sequence_df['Date'] = pd.to_datetime(sequence_df['Date'])
        
        metrics = results['metrics']
        current_date = results['current_date']
        start_day_value = sequence_df['Prediction'].iloc[0] if len(sequence_df) > 0 else 0
        
        # 기본 그래프
        basic_plot = plot_varmax_prediction_basic(
            sequence_df, current_date, start_day_value,
            metrics['f1'], metrics['accuracy'], metrics['mape'], metrics['weighted_score']
        )
        
        # 이동평균 그래프
        ma_plot = plot_varmax_moving_average_analysis(
            results['ma_results'], current_date
        )
        
        plots_info = {
            'basic_plot': basic_plot,
            'ma_plot': ma_plot
        }
        
        logger.info("VARMAX visualizations created successfully")
        return plots_info
        
    except Exception as e:
        logger.error(f"Error creating VARMAX visualizations: {str(e)}")
        logger.error(traceback.format_exc())
        return {}

def plot_varmax_moving_average_analysis(ma_results, sequence_start_date, save_prefix=None,
                                        title_prefix="VARMAX Moving Average Analysis", file_path=None):
    """VARMAX 이동평균 분석 그래프"""
    try:
        logger.info(f"Creating VARMAX moving average analysis for {sequence_start_date}")
        
        # 파일별 캐시 디렉토리 가져오기
        if save_prefix is None:
            cache_dirs = get_file_cache_dirs(file_path)
            save_prefix = cache_dirs['ma_plots']
        
        if not ma_results:
            logger.warning("No moving average results to plot")
            return None
        
        windows = list(ma_results.keys())
        n_windows = len(windows)
        
        if n_windows == 0:
            logger.warning("No moving average windows found")
            return None
        
        # 그래프 생성 (2x2 그리드로 최대 4개 윈도우 표시)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"{title_prefix} - {sequence_start_date}", fontsize=16, weight='bold')
        axes = axes.flatten()
        
        for i, window in enumerate(windows[:4]):  # 최대 4개까지만
            ax = axes[i]
            ma_data = ma_results[window]
            
            if not ma_data:
                ax.text(0.5, 0.5, f'No data for {window}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{window} (No Data)")
                continue
            
            # 데이터프레임 변환
            df = pd.DataFrame(ma_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # 예측값과 이동평균 플롯
            ax.plot(df['date'], df['prediction'], marker='o', color='red', 
                   label='Prediction', linewidth=2, markersize=4)
            ax.plot(df['date'], df['ma'], marker='s', color='blue', 
                   label=f'MA-{window.replace("ma", "")}', linewidth=2, markersize=4)
            
            # 실제값 플롯 (있는 경우)
            actual_data = df.dropna(subset=['actual'])
            if len(actual_data) > 0:
                ax.plot(actual_data['date'], actual_data['actual'], 
                       marker='^', color='green', label='Actual', linewidth=2, markersize=4)
            
            ax.set_title(f"{window.upper()} Moving Average")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 빈 subplot 숨기기
        for i in range(n_windows, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # 파일 저장
        os.makedirs(save_prefix, exist_ok=True)
        filename = f"varmax_ma_analysis_{sequence_start_date}.png"
        filepath = os.path.join(save_prefix, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"VARMAX moving average analysis saved: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error creating VARMAX moving average analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return None