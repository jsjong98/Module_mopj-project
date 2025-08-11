import logging
import pandas as pd
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score

import traceback
from statsmodels.tsa.statespace.varmax import VARMAX
from app.core.state_manager import prediction_state

logger = logging.getLogger(__name__)

# VARMAX 관련 import (선택적 가져오기)
try:
    from statsmodels.tsa.statespace.varmax import VARMAX
    VARMAX_AVAILABLE = True
except ImportError:
    VARMAX_AVAILABLE = False
    logger.warning("VARMAX not available. Please install statsmodels.")

SEED = 42

def set_seed(seed=SEED):
    """
    모든 라이브러리의 시드를 고정하여 일관된 예측 결과 보장
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # PyTorch의 deterministic 동작 강제
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Optuna 시드 설정 (하이퍼파라미터 최적화용)
    try:
        import optuna
        # Optuna 2.x 버전 호환
        if hasattr(optuna.samplers, 'RandomSampler'):
            optuna.samplers.RandomSampler(seed=seed)
        # 레거시 지원
        if hasattr(optuna.samplers, '_random'):
            optuna.samplers._random.seed(seed)
    except Exception as e:
        logger.debug(f"Optuna 시드 설정 실패: {e}")
    
    logger.debug(f"🎯 랜덤 시드 {seed}로 고정됨")

# 날짜 포맷팅 유틸리티 함수
def format_date(date_obj, format_str='%Y-%m-%d'):
    """날짜 객체를 문자열로 안전하게 변환"""
    try:
        # pandas Timestamp 또는 datetime.datetime
        if hasattr(date_obj, 'strftime'):
            return date_obj.strftime(format_str)
        
        # numpy.datetime64
        elif isinstance(date_obj, np.datetime64):
            # 날짜 포맷이 'YYYY-MM-DD'인 경우
            return str(date_obj)[:10]
        
        # 문자열인 경우 이미 날짜 형식이라면 추가 처리
        elif isinstance(date_obj, str):
            # GMT 형식이면 파싱하여 변환
            if 'GMT' in date_obj:
                parsed_date = datetime.strptime(date_obj, '%a, %d %b %Y %H:%M:%S GMT')
                return parsed_date.strftime(format_str)
            return date_obj[:10] if len(date_obj) > 10 else date_obj
        
        # 그 외 경우
        else:
            return str(date_obj)
    
    except Exception as e:
        logger.warning(f"날짜 포맷팅 오류: {str(e)}")
        return str(date_obj)

#######################################################################
# VARMAX 관련 클래스 및 함수
#######################################################################

class VARMAXSemiMonthlyForecaster:
    """VARMAX 기반 반월별 시계열 예측 클래스 - 세 번째 탭용"""
    
    def __init__(self, file_path, result_var='MOPJ', pred_days=50):
        # 일관된 예측 결과를 위한 시드 고정
        set_seed()
        
        self.file_path = file_path
        self.result_var = result_var
        self.pred_days = pred_days
        self.df1 = None
        self.df_origin = None
        self.target_df = None
        self.df_train = None
        self.df_test = None
        self.ts_exchange = None
        self.exogenous_data = None
        self.varx_result = None
        self.pred_df = None
        self.final_forecast_var = None
        self.final_value = None
        self.final_index = None
        self.filtered_vars = None
        self.var_num = None  # 기본값
        self.r2_train = None
        self.r2_test = None
        self.pred_index = None
        self.selected_vars = []
        self.mape_value = None

    def load_data(self):
        """데이터 로드 (VARMAX 모델용 - 모든 데이터 사용, 최근 800개로 제한)"""
        try:
            # VARMAX 모델은 장기예측이므로 모든 데이터 사용 (2022년 이전 포함)
            from app.data.loader import load_data as data_loader
            df_full = data_loader(self.file_path, model_type='varmax')
            # 기존 로직 유지: 최근 800개 데이터만 사용
            self.df_origin = df_full.iloc[-800:]
            logger.info(f"VARMAX data loaded: {self.df_origin.shape} (last 800 records from full dataset)")
            logger.info(f"Date range: {self.df_origin.index.min()} to {self.df_origin.index.max()}")
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise e

    def select_variables(self, current_date=None):
        """변수 선택 - 현재 날짜까지의 데이터만 사용하여 데이터 누출 방지"""
        try:
            # 🔑 수정: 현재 날짜까지의 데이터만 사용
            if current_date is not None:
                if isinstance(current_date, str):
                    current_date = pd.to_datetime(current_date)
                recent_data = self.df_origin[self.df_origin.index <= current_date]
                logger.info(f"🔧 Variable selection using data up to {current_date.strftime('%Y-%m-%d')} ({len(recent_data)} records)")
            else:
                recent_data = self.df_origin
                logger.info(f"🔧 Variable selection using all available data ({len(recent_data)} records)")
            
            correlations = recent_data.corr()[self.result_var]
            correlations = correlations.drop(self.result_var)
            correlations = correlations.sort_values(ascending=False)
            select = correlations.index.tolist()
            self.selected_vars = select
            
            # 변수 그룹 정의 (원본 코드와 동일)
            variable_groups = {
                'crude_oil': ['WTI', 'Brent', 'Dubai'],
                'gasoline': ['Gasoline_92RON', 'Gasoline_95RON', 'Europe_M.G_10ppm', 'RBOB (NYMEX)_M1'],
                'naphtha': ['MOPAG', 'MOPS', 'Europe_CIF NWE'],
                'lpg': ['C3_LPG', 'C4_LPG'],
                'product': ['EL_CRF NEA', 'EL_CRF SEA', 'PL_FOB Korea', 'BZ_FOB Korea', 'BZ_FOB SEA', 'BZ_FOB US M1', 'BZ_FOB US M2', 'TL_FOB Korea', 'TL_FOB US M1', 'TL_FOB US M2',
                'MX_FOB Korea', 'PX_FOB Korea', 'SM_FOB Korea', 'RPG Value_FOB PG', 'FO_HSFO 180 CST', 'MTBE_FOB Singapore'],
                'spread': ['Monthly Spread','BZ_H2-TIME SPREAD', 'Brent_WTI', 'MOPJ_MOPAG', 'MOPJ_MOPS', 'Naphtha_Spread', 'MG92_E Nap', 'C3_MOPJ', 'C4_MOPJ', 'Nap_Dubai',
                'MG92_Nap_MOPS', '95R_92R_Asia', 'M1_M2_RBOB', 'RBOB_Brent_m1', 'RBOB_Brent_m2', 'EL_MOPJ', 'PL_MOPJ', 'BZ_MOPJ', 'TL_MOPJ', 'PX_MOPJ', 'HD_EL', 'LD_EL', 'LLD_EL', 'PP_PL',
                'SM_EL+BZ', 'US_FOBK_BZ', 'NAP_HSFO_180', 'MTBE_MOPJ'],
                'economics': ['Dow_Jones', 'Euro', 'Gold'],
                'freight': ['Freight_55_PG', 'Freight_55_Maili', 'Freight_55_Yosu', 'Freight_55_Daes', 'Freight_55_Chiba',
                'Freight_75_PG', 'Freight_75_Maili', 'Freight_75_Yosu', 'Freight_75_Daes', 'Freight_75_Chiba', 'Flat Rate_PG', 'Flat Rate_Maili', 'Flat Rate_Yosu', 'Flat Rate_Daes',
                'Flat Rate_Chiba']
            }
            
            # 그룹별 최적 변수 선택
            self.filtered_vars = []
            for group, variables in variable_groups.items():
                filtered_group_vars = [var for var in variables if var in self.selected_vars]
                if filtered_group_vars:
                    best_var = max(filtered_group_vars, key=lambda x: abs(correlations[x]))
                    self.filtered_vars.append(best_var)
            
            self.selected_vars = sorted(self.filtered_vars, key=lambda x: abs(correlations[x]), reverse=True)
            logger.info(f"Selected {len(self.selected_vars)} variables for VARMAX prediction")
            logger.info(f"Top 5 selected variables: {self.selected_vars[:5]}")
            
        except Exception as e:
            logger.error(f"Variable selection failed: {str(e)}")
            raise e

    def prepare_data_for_prediction(self, current_date):
        """예측용 데이터 준비"""
        try:
            # 현재 날짜까지의 데이터만 사용
            if isinstance(current_date, str):
                current_date = pd.to_datetime(current_date)
            
            # 현재 날짜까지의 데이터 필터링
            historical_data = self.df_origin[self.df_origin.index <= current_date]
            
            filtered_values = self.selected_vars
            input_columns = filtered_values[:self.var_num]
            output_column = [self.result_var]
            
            self.final_value = historical_data.iloc[-1][self.result_var]
            self.final_index = historical_data.index[-1]

            self.target_df = historical_data[input_columns + output_column]
            
            self.df_train = self.target_df
            
            # 외생변수 (환율) 설정
            if 'Exchange' in self.df_origin.columns:
                self.ts_exchange = historical_data['Exchange']
                self.exogenous_data = pd.DataFrame(self.ts_exchange, index=self.ts_exchange.index)
            else:
                self.exogenous_data = None
                
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise e

    def fit_varmax_model(self):
        """VARMAX 모델 학습"""
        try:
            if not VARMAX_AVAILABLE:
                raise ImportError("VARMAX dependencies not available")
                
            logger.info("🔄 [VARMAX_FIT] Starting VARMAX model fitting...")
            logger.info(f"🔄 [VARMAX_FIT] Training data shape: {self.df_train.shape}")
            logger.info(f"🔄 [VARMAX_FIT] Exogenous data available: {self.exogenous_data is not None}")
            
            best_p = 7
            best_q = 0
            
            logger.info(f"🔄 [VARMAX_FIT] Creating VARMAX model with order=({best_p}, {best_q})")
            varx_model = VARMAX(endog=self.df_train, exog=self.exogenous_data, order=(best_p, best_q))
            
            logger.info("🔄 [VARMAX_FIT] Starting model fitting (this may take a while)...")
            
            # 🔑 진행률 업데이트
            prediction_state['varmax_prediction_progress'] = 50
            
            self.varx_result = varx_model.fit(disp=False, maxiter=1000)
            
            if hasattr(self.varx_result, 'converged') and not self.varx_result.converged:
                logger.warning("⚠️ [VARMAX_FIT] VARMAX model did not converge (res.converged=False)")
            else:
                logger.info("✅ [VARMAX_FIT] VARMAX model converged successfully")
                
            logger.info("✅ [VARMAX_FIT] VARMAX model fitted successfully")
            
            # 🔑 진행률 업데이트
            prediction_state['varmax_prediction_progress'] = 60
            
        except Exception as e:
            logger.error(f"❌ [VARMAX_FIT] VARMAX fitting failed: {str(e)}")
            logger.error(f"❌ [VARMAX_FIT] Fitting error traceback: {traceback.format_exc()}")
            
            # 🔑 에러 상태 업데이트
            prediction_state['varmax_error'] = f"Model fitting failed: {str(e)}"
            prediction_state['varmax_is_predicting'] = False
            prediction_state['varmax_prediction_progress'] = 0
            
            raise e

    def forecast_varmax(self):
        """VARMAX 예측 수행"""
        try:
            # 미래 외생변수 준비
            if self.exogenous_data is not None:
                # 마지막 값을 예측 기간만큼 반복
                last_exog_value = self.ts_exchange.iloc[-1]
                future_dates = pd.bdate_range(start=self.final_index + pd.Timedelta(days=1), periods=self.pred_days)
                exog_future = pd.DataFrame([last_exog_value] * self.pred_days, 
                                         index=future_dates, 
                                         columns=self.exogenous_data.columns)
            else:
                exog_future = None
                future_dates = pd.bdate_range(start=self.final_index + pd.Timedelta(days=1), periods=self.pred_days)
                
            # VARMAX 예측
            varx_forecast = self.varx_result.forecast(steps=self.pred_days, exog=exog_future)
            self.pred_index = future_dates
            self.pred_df = pd.DataFrame(varx_forecast.values, index=self.pred_index, columns=self.df_train.columns)
            logger.info(f"VARMAX forecast completed for {self.pred_days} days")
            
        except Exception as e:
            logger.error(f"VARMAX forecasting failed: {str(e)}")
            raise e

    def residual_correction(self):
        """랜덤포레스트를 이용한 잔차 보정"""
        try:
            if not VARMAX_AVAILABLE:
                logger.warning("VARMAX not available, skipping residual correction")
                self.final_forecast_var = self.pred_df[[self.result_var]]
                self.r2_train = 0.0
                self.r2_test = 0.0
                return
                
            # 잔차 계산
            residuals_origin = self.df_train - self.varx_result.fittedvalues
            residuals_real = residuals_origin.iloc[1:]
            X = residuals_real.iloc[:, :-1]
            y = residuals_real.iloc[:, -1]
            
            # 테스트 크기 계산
            test_size_value = min(0.3, (self.pred_days + 1) / len(self.target_df))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_value, shuffle=False)
            
            # 랜덤포레스트 모델 학습
            rfr_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rfr_model.fit(X_train, y_train)
            
            # 성능 평가
            y_train_pred = rfr_model.predict(X_train)
            y_test_pred = rfr_model.predict(X_test)
            self.r2_train = r2_score(y_train, y_train_pred)
            self.r2_test = r2_score(y_test, y_test_pred)
            
            # 예측에 잔차 보정 적용
            var_predictions = self.pred_df[[self.result_var]]
            
            # 최근 잔차 데이터로 예측
            recent_residuals = residuals_real.iloc[-self.pred_days:, :-1]
            if len(recent_residuals) < self.pred_days:
                # 데이터가 부족하면 마지막 행을 반복
                last_residual = residuals_real.iloc[-1:, :-1]
                additional_rows = self.pred_days - len(recent_residuals)
                repeated_residuals = pd.concat([last_residual] * additional_rows, ignore_index=True)
                recent_residuals = pd.concat([recent_residuals, repeated_residuals])[:self.pred_days]
            
            rfr_predictions = rfr_model.predict(recent_residuals.iloc[:len(var_predictions)])
            rfr_pred_df = pd.DataFrame(rfr_predictions, 
                                     index=var_predictions.index, 
                                     columns=var_predictions.columns)
            
            # 최종 예측값 = VARMAX 예측 + 잔차 보정
            self.final_forecast_var = var_predictions.add(rfr_pred_df)
            
            logger.info(f"Residual correction completed. Train R2: {self.r2_train:.4f}, Test R2: {self.r2_test:.4f}")
            
        except Exception as e:
            logger.error(f"Residual correction failed: {str(e)}")
            # 보정 실패 시 원본 VARMAX 예측값 사용
            self.final_forecast_var = self.pred_df[[self.result_var]]
            self.r2_train = 0.0
            self.r2_test = 0.0

    def calculate_performance_metrics(self, actual_data=None):
        """성능 지표 계산 (실제 데이터가 있는 경우)"""
        if actual_data is None:
            return {
                'f1': 0.0,
                'accuracy': 0.0,
                'mape': 0.0,
                'weighted_score': 0.0,
                'r2_train': self.r2_train or 0.0,
                'r2_test': self.r2_test or 0.0
            }
        
        try:
            # 방향성 예측 성능
            pred_series = self.final_forecast_var[self.result_var]
            actual_series = actual_data
            
            pred_trend = (pred_series.diff() > 0).astype(int)[1:]
            actual_trend = (actual_series.diff() > 0).astype(int)[1:]
            
            # 공통 인덱스로 맞춤
            common_idx = pred_trend.index.intersection(actual_trend.index)
            if len(common_idx) > 0:
                pred_trend_common = pred_trend.loc[common_idx]
                actual_trend_common = actual_trend.loc[common_idx]
                
                if VARMAX_AVAILABLE:
                    precision = precision_score(actual_trend_common, pred_trend_common, zero_division=0)
                    recall = recall_score(actual_trend_common, pred_trend_common, zero_division=0)
                    f1 = f1_score(actual_trend_common, pred_trend_common, zero_division=0)
                    accuracy = (actual_trend_common == pred_trend_common).mean() * 100
                else:
                    precision = recall = f1 = accuracy = 0.0
            else:
                precision = recall = f1 = accuracy = 0.0
            
            # MAPE 계산
            common_values_pred = pred_series.loc[common_idx] if len(common_idx) > 0 else pred_series
            common_values_actual = actual_series.loc[common_idx] if len(common_idx) > 0 else actual_series
            
            mask = common_values_actual != 0
            if mask.any():
                mape = np.mean(np.abs((common_values_actual[mask] - common_values_pred[mask]) / common_values_actual[mask])) * 100
            else:
                mape = 0.0
            
            return {
                'f1': f1,
                'accuracy': accuracy,
                'mape': mape,
                'weighted_score': f1 * 100,  # F1 점수를 가중 점수로 사용
                'r2_train': self.r2_train or 0.0,
                'r2_test': self.r2_test or 0.0,
                'precision': precision,
                'recall': recall
            }
            
        except Exception as e:
            logger.error(f"Performance calculation failed: {str(e)}")
            return {
                'f1': 0.0,
                'accuracy': 0.0,
                'mape': 0.0,
                'weighted_score': 0.0,
                'r2_train': self.r2_train or 0.0,
                'r2_test': self.r2_test or 0.0
            }

    def calculate_moving_averages(self, predictions, current_date, windows=[5, 10, 23]):
        """이동평균 계산 (기존 app.py 방식과 동일)"""
        try:
            results = {}
            
            # 예측 데이터를 DataFrame으로 변환
            pred_df = pd.DataFrame(predictions)
            pred_df['Date'] = pd.to_datetime(pred_df['Date'])
            pred_df = pred_df.sort_values('Date')
            
            # 과거 데이터 추가 (이동평균 계산용)
            historical_data = self.df_origin[self.df_origin.index <= current_date]
            historical_series = historical_data[self.result_var].tail(30)  # 최근 30일
            
            # 예측 시리즈 생성
            prediction_series = pd.Series(
                data=pred_df['Prediction'].values,
                index=pred_df['Date']
            )
            
            # 과거와 예측 데이터 결합
            combined_series = pd.concat([historical_series, prediction_series])
            combined_series = combined_series[~combined_series.index.duplicated(keep='last')]
            combined_series = combined_series.sort_index()
            
            # 각 윈도우별 이동평균 계산
            for window in windows:
                rolling_avg = combined_series.rolling(window=window, min_periods=1).mean()
                
                window_results = []
                for i, row in pred_df.iterrows():
                    date = row['Date']
                    pred_value = row['Prediction']
                    actual_value = row['Actual']
                    
                    # 해당 날짜의 이동평균
                    ma_value = rolling_avg.loc[date] if date in rolling_avg.index else None
                    
                    window_results.append({
                        'date': date,
                        'prediction': pred_value,
                        'actual': actual_value,
                        'ma': ma_value
                    })
                
                results[f'ma{window}'] = window_results
            
            return results
            
        except Exception as e:
            logger.error(f"Moving average calculation failed: {str(e)}")
            return {}

    def calculate_moving_averages_varmax(self, predictions, current_date, windows=[5, 10, 20, 30]):
        try:
            results = {}
            
            # 예측 데이터를 DataFrame으로 변환
            pred_df = pd.DataFrame(predictions)
            pred_df['Date'] = pd.to_datetime(pred_df['Date'])
            pred_df = pred_df.sort_values('Date')
            
            # 과거 데이터 추가 (이동평균 계산용)
            historical_data = self.df_origin[self.df_origin.index <= current_date]
            historical_series = historical_data[self.result_var].tail(30)  # 최근 30일
            
            # 예측 시리즈 생성
            prediction_series = pd.Series(
                data=pred_df['Prediction'].values,
                index=pred_df['Date']
            )
            
            # 과거와 예측 데이터 결합
            combined_series = pd.concat([historical_series, prediction_series])
            combined_series = combined_series[~combined_series.index.duplicated(keep='last')]
            combined_series = combined_series.sort_index()
            
            # 각 윈도우별 이동평균 계산
            for window in windows:
                rolling_avg = combined_series.rolling(window=window, min_periods=1).mean()
                
                window_results = []
                for i, row in pred_df.iterrows():
                    date = row['Date']
                    pred_value = row['Prediction']
                    actual_value = row['Actual']
                    
                    # 해당 날짜의 이동평균
                    ma_value = rolling_avg.loc[date] if date in rolling_avg.index else None
                    
                    window_results.append({
                        'date': date,
                        'prediction': pred_value,
                        'actual': actual_value,
                        'ma': ma_value
                    })
                
                results[f'ma{window}'] = window_results
            
            return results
            
        except Exception as e:
            logger.error(f"Moving average calculation failed: {str(e)}")
            return {}

    def calculate_half_month_averages(self, predictions, current_date):
        """VarmaxResult 컴포넌트용 반월 평균 데이터 계산"""
        try:
            # 예측 데이터를 DataFrame으로 변환
            pred_df = pd.DataFrame(predictions)
            pred_df['Date'] = pd.to_datetime(pred_df['Date'])
            pred_df = pred_df.sort_values('Date')
            
            # 반월 기간별로 그룹화
            half_month_groups = {}
            
            for _, row in pred_df.iterrows():
                date = row['Date']
                
                # 반월 라벨 생성 (예: 25_05_1 = 2025년 5월 상반기)
                year = date.year % 100  # 연도 마지막 두 자리
                month = date.month
                half = 1 if date.day <= 15 else 2
                
                half_month_label = f"{year:02d}_{month:02d}_{half}"
                
                if half_month_label not in half_month_groups:
                    half_month_groups[half_month_label] = []
                
                half_month_groups[half_month_label].append(row['Prediction'])
            
            # 각 반월 기간의 평균 계산
            half_month_data = []
            for label, values in half_month_groups.items():
                avg_value = np.mean(values)
                half_month_data.append({
                    'half_month_label': label,
                    'half_month_avg': float(avg_value),
                    'count': len(values)
                })
            
            # 라벨순으로 정렬
            half_month_data.sort(key=lambda x: x['half_month_label'])
            
            logger.info(f"반월 평균 데이터 계산 완료: {len(half_month_data)}개 기간")
            
            return half_month_data
            
        except Exception as e:
            logger.error(f"Half month averages calculation failed: {str(e)}")
            return []

    def prepare_variable_for_prediction(self, current_date):
        """예측용 데이터 준비"""
        try:
            # 현재 날짜까지의 데이터만 사용
            if isinstance(current_date, str):
                current_date = pd.to_datetime(current_date)
            
            # 현재 날짜까지의 데이터 필터링
            historical_data = self.df_origin[self.df_origin.index <= current_date]
            
            filtered_values = self.selected_vars
            input_columns = filtered_values[:self.var_num]
            output_column = [self.result_var]
            
            self.final_value = historical_data.iloc[-1-self.pred_days][self.result_var]
            self.final_index = historical_data.index[-1-self.pred_days]

            self.target_df = historical_data[input_columns + output_column]
            
            self.df_train = self.target_df[:-self.pred_days]
            
            # 외생변수 (환율) 설정
            if 'Exchange' in self.df_origin.columns:
                self.ts_exchange = historical_data['Exchange']
                self.ts_exchange = self.ts_exchange[:-self.pred_days]
                self.exogenous_data = pd.DataFrame(self.ts_exchange, index=self.ts_exchange.index)
            else:
                self.exogenous_data = None
                
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise e

    def calculate_mape(self, predicted, actual):
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        return mape 

    def generate_variables_varmax(self, current_date, var_num):
        """변수 수 예측 프로세스 실행"""
        try:
            self.var_num = var_num
            self.load_data()
            self.select_variables(current_date)
            self.prepare_variable_for_prediction(current_date)
            self.fit_varmax_model()
            logger.info("VARMAX 변수 선정 모델 학습 완료")

            self.forecast_varmax()
            logger.info("VARMAX 변수 선정 모델 예측 완료")

            self.residual_correction()
            logger.info(f"잔차 보정 완료 (R2 train={self.r2_train:.3f}, test={self.r2_test:.3f})")
            historical_data = self.df_origin[self.df_origin.index <= current_date]
            test_data = historical_data[-self.pred_days:]
            self.final_forecast_var.index = test_data.index
            self.mape_value = self.calculate_mape(self.final_forecast_var[self.result_var], test_data[self.result_var])
            
            return self.mape_value

        except Exception as e:
            logger.error(f"VARMAX variables generation failed: {str(e)}")
            return None

    def generate_predictions_varmax(self, current_date, var_num):
        """VARMAX 예측 수행"""
        try:
            logger.info(f"🔄 [VARMAX_GEN] Starting VARMAX prediction generation")
            logger.info(f"🔄 [VARMAX_GEN] Parameters: current_date={current_date}, var_num={var_num}")
            
            self.var_num = var_num
            logger.info(f"🔄 [VARMAX_GEN] Step 1: Loading data...")
            prediction_state['varmax_prediction_progress'] = 35
            self.load_data()
            
            logger.info(f"🔄 [VARMAX_GEN] Step 2: Selecting variables...")
            prediction_state['varmax_prediction_progress'] = 40
            self.select_variables(current_date)
            
            logger.info(f"🔄 [VARMAX_GEN] Step 3: Preparing data for prediction...")
            prediction_state['varmax_prediction_progress'] = 45
            self.prepare_data_for_prediction(current_date)
            
            logger.info(f"🔄 [VARMAX_GEN] Step 4: Fitting VARMAX model...")
            # fit_varmax_model 내에서 50→60으로 업데이트됨
            self.fit_varmax_model()
            
            logger.info(f"🔄 [VARMAX_GEN] Step 5: Forecasting...")
            prediction_state['varmax_prediction_progress'] = 65
            self.forecast_varmax()
            
            logger.info(f"🔄 [VARMAX_GEN] Step 6: Residual correction...")
            prediction_state['varmax_prediction_progress'] = 70
            self.residual_correction()
            
            logger.info(f"🔄 [VARMAX_GEN] Step 7: Converting results to standard format...")
            prediction_state['varmax_prediction_progress'] = 75
            # 예측 결과를 표준 형식으로 변환
            predictions = []
            for date, value in self.final_forecast_var.iterrows():
                predictions.append({
                    'Date': format_date(date),
                    'Prediction': float(value[self.result_var]),
                    'Actual': None  # 실제값은 미래이므로 None
                })
            logger.info(f"🔄 [VARMAX_GEN] Converted {len(predictions)} predictions")
            
            logger.info(f"🔄 [VARMAX_GEN] Step 8: Calculating performance metrics...")
            prediction_state['varmax_prediction_progress'] = 80
            # 성능 지표 계산
            metrics = self.calculate_performance_metrics()
            
            logger.info(f"🔄 [VARMAX_GEN] Step 9: Calculating moving averages...")
            prediction_state['varmax_prediction_progress'] = 85
            # 이동평균 계산 (VARMAX용)
            ma_results = self.calculate_moving_averages_varmax(predictions, current_date)
            
            logger.info(f"🔄 [VARMAX_GEN] Step 10: Calculating half-month averages...")
            prediction_state['varmax_prediction_progress'] = 90
            # 반월 평균 데이터 계산 (VarmaxResult 컴포넌트용)
            half_month_data = self.calculate_half_month_averages(predictions, current_date)
            
            logger.info(f"✅ [VARMAX_GEN] All steps completed successfully!")
            logger.info(f"✅ [VARMAX_GEN] Final results: {len(predictions)} predictions, {len(ma_results)} MA windows")
            
            return {
                'success': True,
                'predictions': predictions,  # 원래 예측 데이터 (차트용)
                'half_month_averages': half_month_data,  # 반월 평균 데이터 (VarmaxResult 컴포넌트용)
                'metrics': metrics,
                'ma_results': ma_results,
                'selected_features': self.selected_vars[:var_num],
                'current_date': format_date(current_date),
                'model_info': {
                    'model_type': 'VARMAX',
                    'variables_used': var_num,
                    'prediction_days': self.pred_days,
                    'r2_train': self.r2_train,
                    'r2_test': self.r2_test
                }
            }
            
        except Exception as e:
            logger.error(f"❌ [VARMAX_GEN] VARMAX prediction failed: {str(e)}")
            logger.error(f"❌ [VARMAX_GEN] Full traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e)
            }
        
#######################################################################
# VARMAX 관련 유틸리티 함수
#######################################################################

def varmax_decision(file_path):
    """Varmax 의사결정 관련"""
    fp = pd.read_csv(file_path)
    df = pd.DataFrame(fp, columns=fp.columns)
    col = df.columns
    # 1) 분석에 사용할 변수 리스트
    vars_pct = ['max_pct2', 'min_pct2', 'mean_pct2', 'max_pct3', 'min_pct3', 'mean_pct3']
    logger.info(f'데이터프레임{df}')
    rename_dict = {
    'max_pct2': '[현 반월 최대 증가율]',
    'min_pct2': '[현 반월 최대 감소율]',
    'mean_pct2': '[현 반월 평균 변동률]',
    'max_pct3': '[이전 반월 최대 증가율]',
    'min_pct3': '[이전 반월 최대 감소율]',
    'mean_pct3': '[이전 반월 평균 변동률]'
    }
    rename_col = list(rename_dict.values())
    df = df.rename(columns=rename_dict)
    logger.info(f'열{col}')
    # 2) Case 정의
    case1 = df['saving_rate'] < 0
    abs_thresh = df['saving_rate'].abs().quantile(0.9)
    case2 = df['saving_rate'].abs() >= abs_thresh

    # 3) 최적 조건 탐색 함수
    def find_best_condition(df, case_mask, var):
        best = None
        for direction in ['greater', 'less']:
            for p in np.linspace(0.1, 0.9, 9):
                th = df[var].quantile(p)
                if direction == 'greater':
                    mask = df[var] > th
                else:
                    mask = df[var] < th
                # 샘플 수가 너무 적은 경우 제외
                if mask.sum() < 5:
                    continue
                prop = case_mask[mask].mean()
                if best is None or prop > best[4]:
                    best = (direction, p, th, mask.sum(), prop)
        return best

    # 5) 각 변수별 최적 조건 찾기
    results_case1 = {var: find_best_condition(df, case1, var) for var in rename_col}
    results_case2 = {var: find_best_condition(df, case2, var) for var in rename_col}

    from itertools import combinations
    # 6) 두 변수 조합을 사용하여 saving_rate < 0 분류 성능 평가 (샘플 수 ≥ 30)
    combi_results_case1 = []

    for var1, var2 in combinations(rename_col, 2):
        for d1 in ['greater', 'less']:
            for d2 in ['greater', 'less']:
                for p1 in np.linspace(0.1, 0.9, 9):
                    for p2 in np.linspace(0.1, 0.9, 9):
                        th1 = df[var1].quantile(p1)
                        th2 = df[var2].quantile(p2)
                        mask1 = df[var1] > th1 if d1 == 'greater' else df[var1] < th1
                        mask2 = df[var2] > th2 if d2 == 'greater' else df[var2] < th2
                        mask = mask1 & mask2
                        n = mask.sum()
                        if n < 30:
                            continue
                        rate = case1[mask].mean()
                        combi_results_case1.append({
                            "조건1": f"{var1} { '>' if d1 == 'greater' else '<' } {round(th1, 3)}%",
                            "조건2": f"{var2} { '>' if d2 == 'greater' else '<' } {round(th2, 3)}%",
                            "샘플 수": n,
                            "음수 비율 [%]": round(rate*100, 3)
                        })
    column_order1 = ["조건1", "조건2", "샘플 수", "음수 비율 [%]"]
    combi_df_case1 = pd.DataFrame(combi_results_case1).sort_values(by="음수 비율 [%]", ascending=False)
    combi_df_case1 = combi_df_case1.reindex(columns=column_order1)

    # 7) 두 변수 조합을 사용하여 절댓값 상위 10% 분류 성능 평가
    combi_results_case2 = []

    for var1, var2 in combinations(rename_col, 2):
        for d1 in ['greater', 'less']:
            for d2 in ['greater', 'less']:
                for p1 in np.linspace(0.1, 0.9, 9):
                    for p2 in np.linspace(0.1, 0.9, 9):
                        th1 = df[var1].quantile(p1)
                        th2 = df[var2].quantile(p2)
                        mask1 = df[var1] > th1 if d1 == 'greater' else df[var1] < th1
                        mask2 = df[var2] > th2 if d2 == 'greater' else df[var2] < th2
                        mask = mask1 & mask2
                        n = mask.sum()
                        if n < 30:
                            continue
                        rate = case2[mask].mean()
                        combi_results_case2.append({
                            "조건1": f"{var1} { '>' if d1 == 'greater' else '<' } {round(th1, 3)}%",
                            "조건2": f"{var2} { '>' if d2 == 'greater' else '<' } {round(th2, 3)}%",
                            "샘플 수": n,
                            "상위 변동성 확률 [%]": round(rate*100, 3)
                        })
    column_order2 = ["조건1", "조건2", "샘플 수", "상위 변동성 확률 [%]"]
    combi_df_case2 = pd.DataFrame(combi_results_case2).sort_values(by="상위 변동성 확률 [%]", ascending=False)
    combi_df_case2 = combi_df_case2.reindex(columns=column_order2)
    return {
        'case_1': combi_df_case1.to_dict(orient='records'),
        'case_2': combi_df_case2.to_dict(orient='records')
    }

# VARMAX 관련 캐시 함수 import (호환성을 위해)
try:
    from app.data.cache_manager import load_varmax_prediction, save_varmax_prediction
    logger.info("✅ [VARMAX_MODEL] Successfully imported cache functions")
except ImportError as e:
    logger.warning(f"⚠️ [VARMAX_MODEL] Failed to import cache functions: {e}")
    
    # Fallback 함수 정의
    def load_varmax_prediction(prediction_date):
        """Fallback load_varmax_prediction function"""
        logger.warning("⚠️ [VARMAX_MODEL] Using fallback load_varmax_prediction")
        return None
    
    def save_varmax_prediction(prediction_data, prediction_date):
        """Fallback save_varmax_prediction function"""
        logger.warning("⚠️ [VARMAX_MODEL] Using fallback save_varmax_prediction")
        return False