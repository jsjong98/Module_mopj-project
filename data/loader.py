import pandas as pd
import os
import logging
import warnings
import numpy as np
import shutil
import logging
from pathlib import Path
import time
import traceback
import json

logger = logging.getLogger(__name__)

# 필요한 함수들 import (순환 참조 방지를 위해 조건부)
def get_file_cache_dirs(file_path):
    """
    파일별 캐시 디렉토리를 가져오는 함수 (순환 참조 방지를 위해 local import)
    """
    try:
        from app.data.cache_manager import get_file_cache_dirs as _get_file_cache_dirs
        return _get_file_cache_dirs(file_path)
    except ImportError:
        # cache_manager가 없는 경우 기본 캐시 구조 생성
        import hashlib
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:12]
        file_name = Path(file_path).stem
        cache_dir_name = f"{file_hash}_{file_name}"
        cache_root = Path(CACHE_ROOT_DIR) / cache_dir_name
        
        # 기본 구조 반환
        return {
            'root': cache_root,
            'models': cache_root / 'models',
            'predictions': cache_root / 'predictions',
            'plots': cache_root / 'static' / 'plots',
            'ma_plots': cache_root / 'static' / 'ma_plots',
            'accumulated': cache_root / 'accumulated'
        }

_dataframe_cache = {}
_cache_expiry_seconds = 3600

# 무한 루프 방지를 위한 캐시 생성 플래그
_cache_creation_in_progress = False

# xlwings 관련 전역 변수 처리 (app_rev.py에서 가져옴)
try:
    import xlwings as xw
    XLWINGS_AVAILABLE = True
except ImportError:
    XLWINGS_AVAILABLE = False
    logging.warning("xlwings not available - falling back to pandas only")

# 전역 변수
from app.config import CACHE_ROOT_DIR, UPLOAD_FOLDER
from app.data.cache_manager import get_data_content_hash # load_data에서 사용

def safe_read_excel(file_path, **kwargs):
    """
    Excel 파일을 안전하게 읽는 헬퍼 함수
    다양한 엔진을 시도하여 호환성 문제를 해결합니다.
    
    Args:
        file_path (str): Excel 파일 경로
        **kwargs: pandas.read_excel에 전달할 추가 인자들
    
    Returns:
        pd.DataFrame: 읽어온 데이터프레임
    
    Raises:
        ValueError: 모든 엔진으로 읽기에 실패한 경우
    """
    engines = ['openpyxl', 'xlrd']  # 시도할 엔진 순서
    last_error = None
    
    for engine in engines:
        try:
            logger.info(f"📖 Excel 파일 읽기 시도 (엔진: {engine}): {os.path.basename(file_path)}")
            df = pd.read_excel(file_path, engine=engine, **kwargs)
            logger.info(f"✅ Excel 파일 읽기 성공 (엔진: {engine})")
            return df
        except Exception as e:
            last_error = e
            logger.warning(f"⚠️ 엔진 {engine}으로 읽기 실패: {str(e)}")
            continue
    
    # 모든 엔진이 실패한 경우
    error_msg = f"모든 엔진으로 Excel 파일 읽기 실패. 마지막 오류: {str(last_error)}"
    logger.error(f"❌ {error_msg}")
    raise ValueError(error_msg)

def load_data_with_xlwings(file_path, model_type=None):
    """
    xlwings를 사용하여 DRM 보호 파일 및 보안 확장자 파일을 Excel 프로세스 경유로 읽는 함수
    🔑 핵심: Excel.exe를 중계자로 사용하여 DRM 인증 및 보안 정책 우회
    
    Args:
        file_path (str): Excel 파일 경로 (DRM 보호 또는 보안 확장자 포함)
        model_type (str): 모델 타입 ('lstm', 'varmax', None)
    
    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    if not XLWINGS_AVAILABLE:
        raise ImportError("xlwings is not available. Please install it with: pip install xlwings")
    
    # 🔒 보안 확장자 확인
    actual_file_type, is_security_file = normalize_security_extension(file_path)
    
    if is_security_file:
        logger.info(f"🔓 [XLWINGS_SECURITY] Security extension detected: {os.path.basename(file_path)} (.{os.path.splitext(file_path)[1][1:]} -> {actual_file_type})")
        logger.info(f"🔓 [XLWINGS_SECURITY] Using Excel process to bypass security policy")
    else:
        logger.info(f"🔓 [XLWINGS_DRM] DRM 우회: Excel 프로세스 경유로 파일 로딩")
    
    logger.info(f"📁 [XLWINGS] Target: {os.path.basename(file_path)}")
    
    app = None
    wb = None
    
    try:
        # 🔑 핵심: Excel 애플리케이션을 실제 Excel.exe로 시작
        # DRM은 Excel.exe를 신뢰하므로 이를 통해 우회 가능
        app = xw.App(visible=False, add_book=False)
        app.display_alerts = False  # DRM 관련 경고도 무시
        app.screen_updating = False 
        
        logger.info(f"📱 [XLWINGS_DRM] Excel.exe process started (PID: {app.pid})")
        logger.info(f"🔐 [XLWINGS_DRM] DRM will recognize this as trusted Excel access")
        
        # 🔒 DRM 보호 파일 열기 시도
        try:
            # read_only=True로 DRM 경고 최소화
            # update_links=False로 외부 링크 업데이트 방지 (보안 이슈 회피)
            wb = app.books.open(file_path, read_only=True, update_links=False, password=None)
            logger.info(f"✅ [XLWINGS_DRM] DRM 보호 파일 성공적으로 열림: {wb.name}")
        except Exception as open_error:
            logger.error(f"❌ [XLWINGS_DRM] DRM 파일 열기 실패: {str(open_error)}")
            # 비밀번호가 필요한 경우나 다른 DRM 이슈
            if "password" in str(open_error).lower():
                raise ValueError("🔐 DRM 파일에 비밀번호가 필요합니다. 파일 제공업체에 비밀번호를 요청하세요.")
            elif "permission" in str(open_error).lower() or "access" in str(open_error).lower():
                raise ValueError("🚫 DRM 접근 권한이 없습니다. IT 관리자에게 권한 요청을 하거나 인증된 환경에서 파일을 여세요.")
            else:
                raise ValueError(f"🔒 DRM 보호로 인해 파일을 열 수 없습니다: {str(open_error)}")
        
        # 워크시트 정보 확인
        sheet_names = [sheet.name for sheet in wb.sheets]
        logger.info(f"📋 [XLWINGS_DRM] Available sheets: {sheet_names}")
        
        # 적절한 시트 선택
        target_sheet_name = '29 Nov 2010 till todate'
        if target_sheet_name in sheet_names:
            sheet = wb.sheets[target_sheet_name]
            logger.info(f"🎯 [XLWINGS_DRM] Using target sheet: {target_sheet_name}")
        else:
            sheet = wb.sheets[0]  # 첫 번째 시트 사용
            logger.info(f"🎯 [XLWINGS_DRM] Using first sheet: {sheet.name}")
        
        # 🔓 DRM 우회 데이터 추출
        try:
            # 사용된 범위 확인
            used_range = sheet.used_range
            if used_range is None:
                raise ValueError("Sheet appears to be empty")
            
            logger.info(f"📏 [XLWINGS_DRM] Used range: {used_range.address}")
            
            # 🚀 핵심: Excel 프로세스를 통해 데이터를 메모리로 복사
            # 이 과정에서 DRM 보호가 해제됨
            df_raw = sheet['A1'].options(pd.DataFrame, index=False, expand='table').value
            
            logger.info(f"📊 [XLWINGS_DRM] Raw DRM 데이터 추출: {df_raw.shape}")
            logger.info(f"📋 [XLWINGS_DRM] Raw columns: {list(df_raw.columns)}")
            
            # 🧹 Excel DRM 텍스트 필터링 및 데이터 정제
            df = clean_drm_from_excel(df_raw)
            
            if df.empty:
                logger.warning(f"⚠️ [XLWINGS_DRM] DRM 정제 후 데이터가 비어있음")
                raise ValueError("🚫 DRM 정제 후 유효한 데이터가 없습니다")
            
            logger.info(f"✅ [XLWINGS_DRM] DRM Excel 정제 완료: {df.shape}")
            logger.info(f"📋 [XLWINGS_DRM] Cleaned columns: {list(df.columns)}")
            
        except Exception as extract_error:
            logger.error(f"❌ [XLWINGS_DRM] 데이터 추출 실패: {str(extract_error)}")
            raise ValueError(f"🔒 DRM 보호로 인해 데이터를 추출할 수 없습니다: {str(extract_error)}")
        
        # 데이터 검증
        if df is None or df.empty:
            raise ValueError("🚫 DRM 파일에서 데이터를 찾을 수 없습니다")
        
        # Date 컬럼 처리
        if 'Date' not in df.columns:
            # 첫 번째 컬럼이 날짜일 가능성 확인
            first_col = df.columns[0]
            if 'date' in first_col.lower() or df[first_col].dtype == 'datetime64[ns]':
                df = df.rename(columns={first_col: 'Date'})
                logger.info(f"🔄 [XLWINGS_DRM] Renamed '{first_col}' to 'Date'")
            else:
                # Date 컬럼이 없어도 진행 (DRM 파일 구조가 다를 수 있음)
                logger.warning(f"⚠️ [XLWINGS_DRM] Date column not found, using data as-is")
                return df
        
        # Date 컬럼을 datetime으로 변환
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            logger.info(f"📅 [XLWINGS_DRM] Date range: {df.index.min()} to {df.index.max()}")
        except Exception as date_error:
            logger.warning(f"⚠️ [XLWINGS_DRM] Date conversion failed: {str(date_error)}")
            # Date 변환에 실패해도 원본 데이터 반환
            return df
        
        # 모델 타입별 데이터 필터링
        if model_type == 'lstm':
            cutoff_date = pd.to_datetime('2022-01-01')
            original_shape = df.shape
            df = df[df.index >= cutoff_date]
            logger.info(f"🔍 [XLWINGS_DRM] LSTM filter: {original_shape[0]} -> {df.shape[0]} records")
            
            if df.empty:
                raise ValueError("No data available after 2022-01-01 filter for LSTM model")
        
        # 기본 데이터 정제
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()
        
        logger.info(f"✅ [XLWINGS_DRM] DRM 우회 완료 - 데이터 성공적으로 로딩: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"❌ [XLWINGS_DRM] DRM 우회 실패: {str(e)}")
        # DRM 관련 구체적 에러 메시지 제공
        if "password" in str(e).lower():
            raise ValueError("🔐 DRM 파일에 비밀번호 보호가 설정되어 있습니다. 파일 제공업체에 비밀번호를 요청하세요.")
        elif "permission" in str(e).lower() or "access" in str(e).lower():
            raise ValueError("🚫 현재 사용자 계정에 DRM 파일 접근 권한이 없습니다. IT 관리자에게 권한을 요청하세요.")
        else:
            raise e
        
    finally:
        # 🧹 Excel 프로세스 정리 (중요: DRM 세션 종료)
        try:
            if wb is not None:
                wb.close()
                logger.info("📖 [XLWINGS_DRM] DRM workbook closed")
        except:
            pass
        
        try:
            if app is not None:
                app.quit()
                logger.info("📱 [XLWINGS_DRM] Excel.exe process terminated")
        except:
            pass

def load_data_safe_holidays(file_path):
    """
    휴일 파일 전용 xlwings 로딩 함수 - 보안프로그램 우회
    
    Args:
        file_path (str): 휴일 Excel 파일 경로
    
    Returns:
        pd.DataFrame: 휴일 데이터프레임 (date, description 컬럼)
    """
    if not XLWINGS_AVAILABLE:
        raise ImportError("xlwings is not available for holiday file loading")
    
    logger.info(f"🔓 [HOLIDAYS_XLWINGS] Loading holiday file with security bypass: {os.path.basename(file_path)}")
    
    app = None
    wb = None
    
    try:
        # Excel 애플리케이션을 백그라운드에서 시작
        app = xw.App(visible=False, add_book=False)
        app.display_alerts = False
        app.screen_updating = False
        
        logger.info(f"📱 [HOLIDAYS_XLWINGS] Excel app started for holidays")
        
        # Excel 파일 열기
        wb = app.books.open(file_path, read_only=True, update_links=False)
        logger.info(f"📖 [HOLIDAYS_XLWINGS] Holiday workbook opened: {wb.name}")
        
        # 첫 번째 시트 사용 (휴일 파일은 보통 단순 구조)
        sheet = wb.sheets[0]
        logger.info(f"🎯 [HOLIDAYS_XLWINGS] Using sheet: {sheet.name}")
        
        # 사용된 범위 확인
        used_range = sheet.used_range
        if used_range is None:
            raise ValueError("Holiday sheet appears to be empty")
        
        logger.info(f"📏 [HOLIDAYS_XLWINGS] Used range: {used_range.address}")
        
        # 데이터를 DataFrame으로 읽기 (헤더 포함)
        df = sheet['A1'].options(pd.DataFrame, index=False, expand='table').value
        
        logger.info(f"📊 [HOLIDAYS_XLWINGS] Holiday data loaded: {df.shape}")
        logger.info(f"📋 [HOLIDAYS_XLWINGS] Columns: {list(df.columns)}")
        
        # 데이터 검증
        if df is None or df.empty:
            raise ValueError("No holiday data found in the Excel file")
        
        # 컬럼명 정규화 (case-insensitive)
        df.columns = df.columns.str.lower()
        
        # 필수 컬럼 확인
        if 'date' not in df.columns:
            # 첫 번째 컬럼을 날짜로 가정
            first_col = df.columns[0]
            df = df.rename(columns={first_col: 'date'})
            logger.info(f"🔄 [HOLIDAYS_XLWINGS] Renamed '{first_col}' to 'date'")
        
        # description 컬럼이 없으면 추가
        if 'description' not in df.columns:
            df['description'] = 'Holiday'
            logger.info(f"➕ [HOLIDAYS_XLWINGS] Added default 'description' column")
        
        logger.info(f"✅ [HOLIDAYS_XLWINGS] Holiday data loaded successfully: {len(df)} holidays")
        return df
        
    except Exception as e:
        logger.error(f"❌ [HOLIDAYS_XLWINGS] Error loading holiday file: {str(e)}")
        raise e
        
    finally:
        # 리소스 정리
        try:
            if wb is not None:
                wb.close()
                logger.info("📖 [HOLIDAYS_XLWINGS] Holiday workbook closed")
        except:
            pass
        
        try:
            if app is not None:
                app.quit()
                logger.info("📱 [HOLIDAYS_XLWINGS] Excel app closed")
        except:
            pass

def load_data_safe(file_path, model_type=None, use_cache=True, use_xlwings_fallback=True):
    """
    안전한 데이터 로딩 함수 - 보안 확장자 지원 및 xlwings를 우선 시도하고 보안 문제 시 pandas로 자동 전환
    
    Args:
        file_path (str): 데이터 파일 경로
        model_type (str): 모델 타입 ('lstm', 'varmax', None)
        use_cache (bool): 메모리 캐시 사용 여부
        use_xlwings_fallback (bool): 이 함수에서는 사용되지 않음 (하위 호환성 유지)
    
    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    # 🔒 보안 파일 처리 (확장자 물리적 변경 포함)
    processed_file_path, actual_ext, is_security_file = process_security_file_in_loader(file_path)
    
    # 처리된 파일 경로로 업데이트
    file_path = processed_file_path
    
    # 파일 타입 재확인
    actual_file_type, _ = normalize_security_extension(file_path)
    
    if is_security_file:
        logger.info(f"🔒 [SECURITY_SAFE] Security file processed: {os.path.basename(file_path)} -> {actual_file_type}")
    
    # ✅ xlwings 우선 시도 (Excel 타입인 경우)
    if XLWINGS_AVAILABLE and actual_file_type in ['xlsx', 'xls', 'excel']:
        try:
            logger.info("🔓 [SECURITY_BYPASS] Attempting xlwings bypass first...")
            return load_data_with_xlwings(file_path, model_type)
        except Exception as xlwings_error:
            logger.warning(f"⚠️ [SECURITY_BYPASS] xlwings failed: {str(xlwings_error)}")
            logger.info("🔄 Falling back to standard pandas loading...")
            # xlwings 실패 시 아래의 표준 load_data로 넘어감
    
    # xlwings를 사용하지 않거나 실패한 경우, 표준 load_data 함수 시도
    try:
        return load_data(file_path, model_type, use_cache)
    except Exception as e:
        logger.error(f"❌ Both xlwings and standard loading failed: {str(e)}")
        raise e

def load_csv_with_xlwings(csv_path, max_retries=3, retry_delay=1):
    """
    xlwings를 사용하여 CSV 파일을 읽는 함수 - 보안프로그램 우회 (재시도 로직 포함)
    
    Args:
        csv_path (str): CSV 파일 경로
        max_retries (int): 최대 재시도 횟수
        retry_delay (int): 재시도 간격 (초)
    
    Returns:
        pd.DataFrame: CSV 데이터프레임
    """
    if not XLWINGS_AVAILABLE:
        raise ImportError("xlwings is not available for CSV loading")
    
    logger.info(f"🔓 [XLWINGS_CSV] Loading CSV file with security bypass: {os.path.basename(csv_path)}")
    
    for attempt in range(max_retries):
        app = None
        wb = None
        
        try:
            # Excel 애플리케이션을 백그라운드에서 시작
            app = xw.App(visible=False, add_book=False)
            app.display_alerts = False
            app.screen_updating = False
            
            logger.info(f"📱 [XLWINGS_CSV] Excel app started for CSV (attempt {attempt + 1}/{max_retries})")
            
            # CSV 파일을 Excel로 열기 (CSV는 자동으로 파싱됨)
            wb = app.books.open(csv_path, read_only=True, update_links=False)
            logger.info(f"📖 [XLWINGS_CSV] CSV workbook opened: {wb.name}")
            
            # 첫 번째 시트 사용 (CSV는 항상 하나의 시트만 가짐)
            sheet = wb.sheets[0]
            
            # 사용된 범위 확인
            used_range = sheet.used_range
            if used_range is None:
                raise ValueError("CSV file appears to be empty")
            
            logger.info(f"📏 [XLWINGS_CSV] Used range: {used_range.address}")
            
            # 데이터를 DataFrame으로 읽기 (헤더 포함)
            df = sheet['A1'].options(pd.DataFrame, index=False, expand='table').value
            
            logger.info(f"📊 [XLWINGS_CSV] CSV data loaded: {df.shape}")
            logger.info(f"📋 [XLWINGS_CSV] Columns: {list(df.columns)}")
            
            # 데이터 검증
            if df is None or df.empty:
                raise ValueError("No data found in the CSV file")
            
            logger.info(f"✅ [XLWINGS_CSV] CSV loaded successfully: {df.shape}")
            return df
            
        except Exception as e:
            # RPC 오류 코드 확인 (Windows COM 오류)
            is_rpc_error = (
                hasattr(e, 'args') and 
                len(e.args) > 0 and 
                isinstance(e.args[0], tuple) and 
                len(e.args[0]) > 0 and 
                e.args[0][0] in [-2147023174, -2147023170, -2147023173]  # RPC 서버 오류들
            )
            
            if is_rpc_error and attempt < max_retries - 1:
                logger.warning(f"⚠️ [XLWINGS_CSV] RPC error detected (attempt {attempt + 1}/{max_retries}): {e}")
                logger.info(f"🔄 [XLWINGS_CSV] Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                logger.error(f"❌ [XLWINGS_CSV] Error loading CSV file (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise e
                
        finally:
            # 리소스 정리 - 더 안전한 방식
            try:
                if wb is not None:
                    wb.close()
                    logger.info("📖 [XLWINGS_CSV] CSV workbook closed")
            except Exception as cleanup_error:
                logger.debug(f"🔧 [XLWINGS_CSV] Workbook cleanup error: {cleanup_error}")
            
            try:
                if app is not None:
                    app.quit()
                    logger.info("📱 [XLWINGS_CSV] Excel app closed")
            except Exception as cleanup_error:
                logger.debug(f"🔧 [XLWINGS_CSV] App cleanup error: {cleanup_error}")
    
    # 모든 재시도가 실패한 경우
    raise RuntimeError(f"Failed to load CSV after {max_retries} attempts")

def load_csv_safe_with_fallback(csv_path):
    """
    안전한 CSV 로드 함수 - .cs 파일은 pandas 우선, 일반 CSV는 xlwings 우선
    
    Args:
        csv_path (str): CSV 파일 경로
    
    Returns:
        pd.DataFrame: CSV 데이터프레임
    """
    # 경로를 문자열로 변환 (Path 객체 대응)
    csv_path_str = str(csv_path)
    
    logger.info(f"📊 [SAFE_CSV] Loading CSV safely: {os.path.basename(csv_path_str)}")
    
    # .cs 확장자 파일인지 확인
    is_cs_file = csv_path_str.lower().endswith('.cs')
    
    if is_cs_file:
        logger.info(f"🔍 [SAFE_CSV] Detected .cs file - using pandas first to avoid C# parsing issues...")
        
        # .cs 파일의 경우 pandas를 먼저 시도
        separators = [',', ';', '\t']
        df = None
        
        for sep in separators:
            try:
                df_test = pd.read_csv(csv_path_str, sep=sep, encoding='utf-8')
                
                # 단일 컬럼에 구분자가 포함된 경우 (잘못된 파싱) 체크
                if len(df_test.columns) == 1:
                    col_name = df_test.columns[0]
                    if ',' in col_name or ';' in col_name or '\t' in col_name:
                        logger.warning(f"⚠️ [SAFE_CSV] Detected incorrect parsing with '{sep}' - single column contains separators")
                        continue
                
                df = df_test
                logger.info(f"✅ [SAFE_CSV] Successfully loaded .cs file with pandas using separator '{sep}': {df.shape}")
                logger.info(f"📋 [SAFE_CSV] Columns: {list(df.columns)}")
                return df
                
            except Exception as sep_error:
                logger.warning(f"⚠️ [SAFE_CSV] Failed with separator '{sep}': {str(sep_error)}")
                continue
        
        # .cs 파일의 경우 xlwings 사용하지 않음 (C# 파일로 인식할 수 있음)
        if df is None:
            logger.warning(f"⚠️ [SAFE_CSV] All pandas attempts failed for .cs file")
        
    else:
        # 일반 CSV 파일의 경우 기존 로직 (xlwings 우선)
        try:
            # xlwings로 먼저 시도
            if XLWINGS_AVAILABLE:
                logger.info(f"🔓 [SAFE_CSV] Attempting xlwings first...")
                try:
                    df = load_csv_with_xlwings(csv_path_str)
                    
                    # xlwings 결과 검증 (단일 컬럼 문제 확인)
                    if len(df.columns) == 1 and (',' in df.columns[0] or ';' in df.columns[0]):
                        logger.warning(f"⚠️ [SAFE_CSV] xlwings returned single column issue, falling back to pandas...")
                        raise ValueError("xlwings parsing issue - single column detected")
                    
                    logger.info(f"✅ [SAFE_CSV] Successfully loaded CSV with xlwings: {df.shape}")
                    logger.info(f"📋 [SAFE_CSV] Columns: {list(df.columns)}")
                    return df
                    
                except Exception as xlwings_error:
                    logger.warning(f"⚠️ [SAFE_CSV] xlwings failed: {str(xlwings_error)}")
                    logger.info(f"🔄 [SAFE_CSV] Falling back to pandas with multiple separators...")
            
            # xlwings 실패 또는 없는 경우 pandas로 fallback (여러 구분자 테스트)
            logger.info(f"📊 [SAFE_CSV] Attempting pandas with multiple separators...")
            
            separators = [',', ';', '\t']
            df = None
            
            for sep in separators:
                try:
                    df_test = pd.read_csv(csv_path_str, sep=sep, encoding='utf-8')
                    
                    # 단일 컬럼에 구분자가 포함된 경우 (잘못된 파싱) 체크
                    if len(df_test.columns) == 1:
                        col_name = df_test.columns[0]
                        if ',' in col_name or ';' in col_name or '\t' in col_name:
                            logger.warning(f"⚠️ [SAFE_CSV] Detected incorrect parsing with '{sep}' - single column contains separators")
                            continue
                    
                    df = df_test
                    logger.info(f"✅ [SAFE_CSV] Successfully loaded with pandas using separator '{sep}': {df.shape}")
                    logger.info(f"📋 [SAFE_CSV] Columns: {list(df.columns)}")
                    break
                    
                except Exception as sep_error:
                    logger.warning(f"⚠️ [SAFE_CSV] Failed with separator '{sep}': {str(sep_error)}")
                    continue
            
        except Exception as e:
            logger.error(f"❌ [SAFE_CSV] Error in normal CSV processing: {str(e)}")
            df = None
    
    # 최종 확인 및 오류 처리
    if df is None:
        # 마지막으로 기본 pandas 시도
        try:
            logger.info(f"🔧 [SAFE_CSV] Final attempt with default pandas settings...")
            df = pd.read_csv(csv_path_str, encoding='utf-8')
            logger.info(f"✅ [SAFE_CSV] Final attempt successful: {df.shape}")
            logger.info(f"📋 [SAFE_CSV] Columns: {list(df.columns)}")
            return df
        except Exception as final_error:
            logger.error(f"❌ [SAFE_CSV] All methods failed: {str(final_error)}")
            raise RuntimeError("Failed to load CSV with all methods")
    
    return df

# 데이터 로딩 및 전처리 함수
def load_data(file_path, model_type=None, use_cache=True):
    from app.data.preprocessor import process_excel_data_complete
    """
    데이터 로드 및 기본 전처리
    
    Args:
        file_path (str): 데이터 파일 경로
        model_type (str): 모델 타입 ('lstm', 'varmax', None)
                         - 'lstm': 단일/누적 예측용, 2022년 이전 데이터 제거
                         - 'varmax': 장기예측용, 모든 데이터 유지
                         - None: 기본 동작 (모든 데이터 유지)
        use_cache (bool): 메모리 캐시 사용 여부 (default: True)
    
    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    # 🔧 메모리 캐시 확인 (중복 로딩 방지)
    cache_key = f"{file_path}|{model_type}|{os.path.getmtime(file_path)}"
    current_time = time.time()
    
    if use_cache and cache_key in _dataframe_cache:
        cached_data, cache_time = _dataframe_cache[cache_key]
        if (current_time - cache_time) < _cache_expiry_seconds:
            logger.info(f"🚀 [CACHE_HIT] Using cached DataFrame for {os.path.basename(file_path)} (saved {current_time - cache_time:.1f}s ago)")
            return cached_data.copy()  # 복사본 반환으로 원본 보호
        else:
            # 만료된 캐시 제거
            del _dataframe_cache[cache_key]
            logger.info(f"🗑️ [CACHE_EXPIRED] Removed expired cache for {os.path.basename(file_path)}")
    
    logger.info(f"📁 [LOAD_DATA] Loading data with model_type: {model_type} from {os.path.basename(file_path)}")

    # 🔒 1단계: 보안 파일 처리 (확장자 물리적 변경 포함)
    processed_file_path, actual_ext, is_security_file = process_security_file_in_loader(file_path)
    
    # 처리된 파일 경로로 업데이트
    file_path = processed_file_path
    
    # 파일 타입 재확인
    actual_file_type, _ = normalize_security_extension(file_path)
    
    if actual_file_type is None:
        raise ValueError(f"Unsupported or undetectable file format: {file_path}")
    
    logger.info(f"📊 [FILE_TYPE] Processed file type: {actual_file_type} (security file: {is_security_file})")
    if is_security_file:
        logger.info(f"📝 [SECURITY_PROCESSED] File path updated: {os.path.basename(processed_file_path)}")
    
    # 2단계: 파일 타입에 따라 다른 로드 방법 사용
    if actual_file_type == 'csv':
        logger.info("Loading CSV file with xlwings fallback support")
        # 원본 CSV 파일은 기존 방식으로 처리
        try:
            if XLWINGS_AVAILABLE:
                logger.info(f"🔓 [XLWINGS_CSV] Attempting to load CSV with xlwings: {file_path}")
                df = load_csv_with_xlwings(file_path)
            else:
                df = pd.read_csv(file_path)
        except Exception as e:
            logger.warning(f"⚠️ [XLWINGS_CSV] xlwings failed, falling back to pandas: {str(e)}")
            df = pd.read_csv(file_path)
        
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # 기본적인 결측치 처리
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()
        
    elif actual_file_type in ['xlsx', 'xls', 'excel']:
        if is_security_file:
            logger.info(f"Loading Excel file (.{actual_file_type}) from security extension using CSV cache system")
        else:
            logger.info("Loading Excel file using CSV cache system")
        
        # 🚀 CSV 캐시 시스템 사용 (무한 루프 방지를 위해 캐시 생성 중에는 건너뛰기)
        if _cache_creation_in_progress:
            logger.info("🔄 [CSV_CACHE] Cache creation in progress - skipping cache check to prevent infinite loop")
            df = load_excel_as_dataframe(file_path, model_type)
        else:
            cache_valid, cache_paths, extension_info = is_csv_cache_valid(file_path, model_type)
        
            if cache_valid == True:
                # 캐시가 완전히 유효하면 CSV 캐시 로딩
                logger.info("✅ [CSV_CACHE] Using valid cache, loading from CSV...")
                df = load_csv_cache(file_path, model_type)
            elif cache_valid == "extension":
                # 🎯 데이터 확장인 경우: 기존 캐시 + 새로운 데이터 처리
                logger.info("🔄 [CSV_CACHE] Data extension detected, updating cache...")
                logger.info(f"    📈 Extension: {extension_info.get('new_rows_count', 0)} new rows")
                logger.info(f"    📅 Date range: {extension_info.get('old_end_date')} → {extension_info.get('new_end_date')}")
                
                try:
                    # 기존 캐시 정보 로드
                    logger.info("📋 [CSV_CACHE] Loading existing cache info...")
                    csv_cache_file = cache_paths['csv_file']
                    df_cached = load_csv_safe_with_fallback(str(csv_cache_file))
                    logger.info(f"    📊 Existing cache: {df_cached.shape}")
                    
                    # 새로운 데이터로 완전한 CSV 캐시 재생성
                    # (확장된 데이터의 경우 전체 재처리가 더 안전함)
                    logger.info("🔧 [CSV_CACHE] Regenerating cache with extended data...")
                    df = create_csv_cache_from_excel(file_path, model_type)
                    
                    logger.info(f"✅ [CSV_CACHE] Successfully updated cache for extended data")
                    logger.info(f"    📊 Old cache: {df_cached.shape} → New cache: {df.shape}")
                    
                    # Date 컬럼을 인덱스로 복원 (create_csv_cache_from_excel 결과에 맞춰)
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                    
                except Exception as ext_error:
                    logger.warning(f"⚠️ [CSV_CACHE] Failed to update extended cache: {str(ext_error)}")
                    logger.info("🔄 [CSV_CACHE] Falling back to complete cache regeneration...")
                    df = create_csv_cache_from_excel(file_path, model_type)
            else:
                # 캐시가 없거나 무효하면 Excel에서 CSV 캐시 생성
                logger.info("📊 [CSV_CACHE] Cache invalid or missing, creating new cache from Excel...")
                df = create_csv_cache_from_excel(file_path, model_type)
    
    else:
        raise ValueError(f"Unsupported file format after normalization: {actual_file_type}")
    
    logger.info(f"Original data shape: {df.shape} (from {df.index.min()} to {df.index.max()})")
    
    # 🔑 모델 타입별 데이터 필터링
    if model_type == 'lstm':
        # LSTM 모델용: 2022년 이전 데이터 제거
        cutoff_date = pd.to_datetime('2022-01-01')
        original_shape = df.shape
        df = df[df.index >= cutoff_date]
        
        logger.info(f"📊 LSTM model: Filtered data from 2022-01-01")
        logger.info(f"  Original: {original_shape[0]} records")
        logger.info(f"  Filtered: {df.shape[0]} records (removed {original_shape[0] - df.shape[0]} records)")
        logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")
        
        if df.empty:
            raise ValueError("No data available after 2022-01-01 filter for LSTM model")
            
    elif model_type == 'varmax':
        # VARMAX 모델용: 모든 데이터 사용
        logger.info(f"📊 VARMAX model: Using all available data")
        logger.info(f"  Full date range: {df.index.min()} to {df.index.max()}")
        
    else:
        # 기본 동작: 모든 데이터 사용
        logger.info(f"📊 Default mode: Using all available data")
        logger.info(f"  Full date range: {df.index.min()} to {df.index.max()}")
    
    # 모든 inf 값을 NaN으로 변환
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 결측치 처리 - 모든 컬럼에 동일하게 적용
    df = df.ffill().bfill()
    
    # 처리 후 남아있는 inf나 nan 확인
    # 숫자 컬럼만 선택해서 isinf 검사
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    has_nan = df.isnull().any().any()
    has_inf = False
    if len(numeric_cols) > 0:
        has_inf = np.isinf(df[numeric_cols].values).any()
    
    if has_nan or has_inf:
        logger.warning("Dataset still contains NaN or inf values after preprocessing")
        
        # 📊 상세한 컬럼 분석 및 문제 진단
        logger.warning("=" * 60)
        logger.warning("📊 DATA QUALITY ANALYSIS")
        logger.warning("=" * 60)
        
        # 1. 데이터 타입 정보
        logger.warning(f"📋 Total columns: {len(df.columns)}")
        logger.warning(f"🔢 Numeric columns: {len(numeric_cols)} - {list(numeric_cols)}")
        non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
        logger.warning(f"🔤 Non-numeric columns: {len(non_numeric_cols)} - {list(non_numeric_cols)}")
        
        # 2. NaN 값 분석
        problematic_cols_nan = df.columns[df.isnull().any()]
        if len(problematic_cols_nan) > 0:
            logger.warning(f"⚠️ Columns with NaN values: {len(problematic_cols_nan)}")
            for col in problematic_cols_nan:
                nan_count = df[col].isnull().sum()
                total_count = len(df[col])
                percentage = (nan_count / total_count) * 100
                logger.warning(f"   • {col}: {nan_count}/{total_count} ({percentage:.1f}%) NaN")
        
        # 3. inf 값 분석 (숫자 컬럼만)
        problematic_cols_inf = []
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                if np.isinf(df[col]).any():
                    problematic_cols_inf.append(col)
                    inf_count = np.isinf(df[col]).sum()
                    total_count = len(df[col])
                    percentage = (inf_count / total_count) * 100
                    logger.warning(f"   • {col}: {inf_count}/{total_count} ({percentage:.1f}%) inf values")
        
        if len(problematic_cols_inf) > 0:
            logger.warning(f"⚠️ Columns with inf values: {len(problematic_cols_inf)} - {problematic_cols_inf}")
        
        # 4. 각 컬럼의 데이터 타입과 샘플 값
        logger.warning("📝 Column details:")
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null_count = df[col].count()
            sample_values = df[col].dropna().head(3).tolist()
            logger.warning(f"   • {col}: {dtype} ({non_null_count} non-null) - Sample: {sample_values}")
        
        problematic_cols = list(set(list(problematic_cols_nan) + problematic_cols_inf))
        logger.warning("=" * 60)
        logger.warning(f"🎯 SUMMARY: {len(problematic_cols)} problematic columns found: {problematic_cols}")
        logger.warning("=" * 60)
        
        # 추가적인 전처리: 남은 inf/nan 값을 해당 컬럼의 평균값으로 대체 (숫자 컬럼만)
        for col in problematic_cols:
            if col in numeric_cols:
                # 숫자 컬럼에 대해서만 inf 처리
                col_mean = df[col].replace([np.inf, -np.inf], np.nan).mean()
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(col_mean)
            else:
                # 비숫자 컬럼에 대해서는 NaN만 처리
                df[col] = df[col].ffill().bfill()
    
    logger.info(f"Final shape after preprocessing: {df.shape}")
    
    # 🔧 메모리 캐시에 저장 (성공적으로 로딩된 경우)
    if use_cache:
        _dataframe_cache[cache_key] = (df.copy(), current_time)
        logger.info(f"💾 [CACHE_SAVE] Saved DataFrame to cache for {os.path.basename(file_path)} (expires in {_cache_expiry_seconds}s)")
        
        # 메모리 관리: 오래된 캐시 정리
        expired_keys = []
        for key, (cached_df, cache_time) in _dataframe_cache.items():
            if (current_time - cache_time) >= _cache_expiry_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del _dataframe_cache[key]
        
        if expired_keys:
            logger.info(f"🗑️ [CACHE_CLEANUP] Removed {len(expired_keys)} expired cache entries")
    
    return df

# 변수 그룹 정의
variable_groups = {
    'crude_oil': ['WTI', 'Brent', 'Dubai'],
    'gasoline': ['Gasoline_92RON', 'Gasoline_95RON', 'Europe_M.G_10ppm', 'RBOB (NYMEX)_M1'],
    'naphtha': ['MOPAG', 'MOPS', 'Europe_CIF NWE'],
    'lpg': ['C3_LPG', 'C4_LPG'],
    'product': ['EL_CRF NEA', 'EL_CRF SEA', 'PL_FOB Korea', 'BZ_FOB Korea', 'BZ_FOB SEA', 'BZ_FOB US M1', 'BZ_FOB US M2', 'TL_FOB Korea', 'TL_FOB US M1', 'TL_FOB US M2',
    'MX_FOB Korea', 'PX_FOB Korea', 'SM_FOB Korea', 'RPG Value_FOB PG', 'FO_HSFO 180 CST', 'MTBE_FOB Singapore'],
    'spread': ['biweekly Spread','BZ_H2-TIME SPREAD', 'Brent_WTI', 'MOPJ_MOPAG', 'MOPJ_MOPS', 'Naphtha_Spread', 'MG92_E Nap', 'C3_MOPJ', 'C4_MOPJ', 'Nap_Dubai',
    'MG92_Nap_MOPS', '95R_92R_Asia', 'M1_M2_RBOB', 'RBOB_Brent_m1', 'RBOB_Brent_m2', 'EL_MOPJ', 'PL_MOPJ', 'BZ_MOPJ', 'TL_MOPJ', 'PX_MOPJ', 'HD_EL', 'LD_EL', 'LLD_EL', 'PP_PL',
    'SM_EL+BZ', 'US_FOBK_BZ', 'NAP_HSFO_180', 'MTBE_MOPJ'],
    'economics': ['Dow_Jones', 'Euro', 'Gold'],
    'freight': ['Freight_55_PG', 'Freight_55_Maili', 'Freight_55_Yosu', 'Freight_55_Daes', 'Freight_55_Chiba',
    'Freight_75_PG', 'Freight_75_Maili', 'Freight_75_Yosu', 'Freight_75_Daes', 'Freight_75_Chiba', 'Flat Rate_PG', 'Flat Rate_Maili', 'Flat Rate_Yosu', 'Flat Rate_Daes',
    'Flat Rate_Chiba'],
    'ETF': ['DIG', 'DUG', 'IYE', 'VDE', 'XLE']
}

def load_holidays_from_file(filepath=None):
    """
    CSV 또는 Excel 파일에서 휴일 목록을 로드하는 함수
    
    Args:
        filepath (str): 휴일 목록 파일 경로, None이면 기본 경로 사용
    
    Returns:
        set: 휴일 날짜 집합 (YYYY-MM-DD 형식)
    """
    # 기본 파일 경로 - holidays 폴더로 변경
    if filepath is None:
        holidays_dir = Path('holidays')
        holidays_dir.mkdir(exist_ok=True)
        filepath = str(holidays_dir / 'holidays.csv')
    
    # 파일 확장자 확인
    _, ext = os.path.splitext(filepath)
    
    # 파일이 존재하지 않으면 기본 휴일 목록 생성
    if not os.path.exists(filepath):
        logger.warning(f"Holiday file {filepath} not found. Creating default holiday file.")
        
        # 기본 2025년 싱가폴 공휴일
        default_holidays = [
            "2025-01-01", "2025-01-29", "2025-01-30", "2025-03-31", "2025-04-18", 
            "2025-05-01", "2025-05-12", "2025-06-07", "2025-08-09", "2025-10-20", 
            "2025-12-25", "2026-01-01"
        ]
        
        # 기본 파일 생성
        df = pd.DataFrame({'date': default_holidays, 'description': ['Singapore Holiday']*len(default_holidays)})
        
        if ext.lower() == '.xlsx':
            df.to_excel(filepath, index=False)
        else:
            df.to_csv(filepath, index=False)
        
        logger.info(f"Created default holiday file at {filepath}")
        return set(default_holidays)
    
    try:
        # 파일 로드 - 보안 문제를 고려한 안전한 로딩 사용
        if ext.lower() == '.xlsx':
            # Excel 파일의 경우 xlwings 보안 우회 기능 사용
            try:
                df = load_data_safe_holidays(filepath)
            except Exception as e:
                logger.warning(f"⚠️ [HOLIDAYS] xlwings loading failed, using safe_read_excel: {str(e)}")
                df = safe_read_excel(filepath)
        else:
            # CSV 파일도 보안 문제를 고려하여 안전한 로딩 사용
            try:
                # 일반 pandas로 먼저 시도
                df = pd.read_csv(filepath)
                logger.info(f"✅ [HOLIDAYS] CSV loaded with pandas: {len(df)} rows")
            except Exception as pandas_error:
                logger.warning(f"⚠️ [HOLIDAYS] pandas CSV loading failed: {str(pandas_error)}")
                try:
                    # xlwings로 CSV 읽기 시도 (보안 우회)
                    if XLWINGS_AVAILABLE:
                        logger.info(f"🔄 [HOLIDAYS] Trying xlwings for CSV: {os.path.basename(filepath)}")
                        df = load_csv_with_xlwings(filepath)
                        logger.info(f"✅ [HOLIDAYS] CSV loaded with xlwings: {len(df)} rows")
                    else:
                        # xlwings 없으면 다시 pandas로 시도 (오류 재발생)
                        raise pandas_error
                except Exception as xlwings_error:
                    logger.error(f"❌ [HOLIDAYS] Both pandas and xlwings failed for CSV")
                    logger.error(f"   Pandas error: {str(pandas_error)}")
                    logger.error(f"   xlwings error: {str(xlwings_error)}")
                    raise pandas_error  # 원래 pandas 오류를 재발생
        
        # 'date' 컬럼이 있는지 확인
        if 'date' not in df.columns:
            logger.error(f"Holiday file {filepath} does not have 'date' column")
            return set()
        
        # 날짜 형식 표준화
        holidays = set()
        for date_str in df['date']:
            try:
                date = pd.to_datetime(date_str)
                holidays.add(date.strftime('%Y-%m-%d'))
            except:
                logger.warning(f"Invalid date format: {date_str}")
        
        logger.info(f"Loaded {len(holidays)} holidays from {filepath}")
        return holidays
        
    except Exception as e:
        logger.error(f"Error loading holiday file: {str(e)}")
        logger.error(traceback.format_exc())
        return set()

# 전역 변수로 휴일 집합 관리
holidays = load_holidays_from_file()

def detect_file_type_by_content(file_path):
    """
    파일 내용을 분석하여 실제 파일 타입을 감지하는 함수
    회사 보안으로 인해 확장자가 변경된 파일들을 처리
    """
    try:
        # 파일의 첫 몇 바이트를 읽어서 파일 타입 감지
        with open(file_path, 'rb') as f:
            header = f.read(8)
        
        # Excel 파일 시그니처 확인
        if header[:4] == b'PK\x03\x04':  # ZIP 기반 파일 (xlsx)
            return 'xlsx'
        elif header[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':  # OLE2 기반 파일 (xls)
            return 'xls'
        
        # CSV 파일인지 확인 (텍스트 기반)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                # CSV 특성 확인: 쉼표나 탭이 포함되어 있고, Date 컬럼이 있는지
                if (',' in first_line or '\t' in first_line) and ('date' in first_line.lower() or 'Date' in first_line):
                    return 'csv'
        except:
            # UTF-8로 읽기 실패시 다른 인코딩 시도
            try:
                with open(file_path, 'r', encoding='cp949') as f:
                    first_line = f.readline()
                    if (',' in first_line or '\t' in first_line) and ('date' in first_line.lower() or 'Date' in first_line):
                        return 'csv'
            except:
                pass
        
        # 기본값 반환
        return None
        
    except Exception as e:
        logger.warning(f"File type detection failed: {str(e)}")
        return None

def normalize_security_extension(file_path):
    """
    회사 보안정책으로 변경된 확장자를 원래 확장자로 복원
    
    Args:
        file_path (str): 파일 경로
    
    Returns:
        tuple: (실제 파일 타입, 보안 확장자인지 여부)
    """
    # 보안 확장자 매핑
    security_extensions = {
        '.cs': 'csv',      # csv -> cs
        '.xl': 'xlsx',     # xlsx -> xl  
        '.log': 'xlsx',    # log -> xlsx (보안 정책으로 Excel 파일을 log로 위장)
        '.dat': None,      # 내용 분석 필요
        '.txt': None,      # 내용 분석 필요
    }
    
    filename_lower = file_path.lower()
    original_ext = os.path.splitext(filename_lower)[1]
    
    # 보안 확장자인지 확인
    if original_ext in security_extensions:
        logger.info(f"🔒 [SECURITY] Security extension detected: {original_ext}")
        
        if security_extensions[original_ext]:
            # 직접 매핑이 있는 경우
            normalized_type = security_extensions[original_ext]
            logger.info(f"🔄 [SECURITY] Extension normalization: {original_ext} -> {normalized_type}")
            return normalized_type, True
        else:
            # 내용 분석이 필요한 경우
            detected_type = detect_file_type_by_content(file_path)
            if detected_type:
                logger.info(f"📊 [CONTENT_DETECTION] Detected file type by content: {detected_type}")
                return detected_type, True
            else:
                logger.warning(f"⚠️ [CONTENT_DETECTION] Failed to detect file type for: {original_ext}")
                return None, True
    
    # 일반 확장자인 경우
    if original_ext == '.csv':
        return 'csv', False
    elif original_ext in ['.xlsx', '.xls']:
        return 'excel', False
    else:
        return None, False

def get_csv_cache_path(file_path, model_type=None, use_file_cache_dirs=True):
    """
    CSV 캐시 파일 경로를 생성하는 함수
    
    Args:
        file_path (str): 원본 파일 경로
        model_type (str): 모델 타입
        use_file_cache_dirs (bool): 기존 파일 캐시 디렉토리 구조 사용 여부
    
    Returns:
        dict: 캐시 경로 정보
    """
    try:
        if use_file_cache_dirs:
            # 기존 파일별 캐시 디렉토리 구조 활용
            cache_dirs = get_file_cache_dirs(file_path)
            cache_root = cache_dirs['root']
        else:
            # 단순 캐시 구조
            file_hash = get_data_content_hash(file_path)
            if not file_hash:
                raise ValueError("Cannot generate file hash for caching")
            
            file_name = Path(file_path).stem
            cache_dir_name = f"{file_hash[:12]}_{file_name}"
            cache_root = Path(CACHE_ROOT_DIR) / cache_dir_name
        
        # CSV 캐시 디렉토리 생성
        csv_cache_dir = Path(cache_root) / 'processed_csv'
        csv_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 타입별 캐시 파일명 (보안을 위해 .cs 확장자 사용)
        if model_type:
            cache_filename = f"data_{model_type}.cs"
            metadata_filename = f"metadata_{model_type}.json"
        else:
            cache_filename = "data.cs"
            metadata_filename = "metadata.json"
        
        return {
            'csv_cache_dir': csv_cache_dir,
            'csv_file': csv_cache_dir / cache_filename,
            'metadata_file': csv_cache_dir / metadata_filename,
            'cache_root': cache_root
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get CSV cache path: {str(e)}")
        raise e

def create_csv_cache_metadata(file_path, model_type, processing_info):
    """
    CSV 캐시 메타데이터 생성
    
    Args:
        file_path (str): 원본 파일 경로
        model_type (str): 모델 타입
        processing_info (dict): 처리 정보
    
    Returns:
        dict: 메타데이터
    """
    import hashlib
    import os
    from datetime import datetime
    
    file_stat = os.stat(file_path)
    file_hash = get_data_content_hash(file_path)
    
    metadata = {
        'original_file': {
            'name': os.path.basename(file_path),
            'path': file_path,
            'size': file_stat.st_size,
            'modified_time': file_stat.st_mtime,
            'content_hash': file_hash
        },
        'processing': {
            'model_type': model_type,
            'pipeline_version': '1.0.0',  # 전처리 파이프라인 버전
            'created_time': datetime.now().isoformat(),
            'processing_info': processing_info
        },
        'csv_cache': {
            'created_time': datetime.now().isoformat(),
            'format_version': '1.0'
        }
    }
    
    return metadata

def is_csv_cache_valid(file_path, model_type=None):
    """
    CSV 캐시가 유효한지 확인하는 함수 (데이터 확장 고려)
    
    Args:
        file_path (str): 원본 파일 경로
        model_type (str): 모델 타입
    
    Returns:
        tuple: (유효성, 캐시 경로 정보, 확장 정보)
    """
    try:
        from app.data.cache_manager import check_data_extension
        
        cache_paths = get_csv_cache_path(file_path, model_type)
        csv_file = cache_paths['csv_file']
        metadata_file = cache_paths['metadata_file']
        
        # 파일 존재 확인
        if not csv_file.exists() or not metadata_file.exists():
            logger.info(f"📋 [CSV_CACHE] Cache files not found for {os.path.basename(file_path)}")
            return False, cache_paths, None
        
        # 메타데이터 로드
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 모델 타입 확인
        processing_info = metadata.get('processing', {})
        if processing_info.get('model_type') != model_type:
            logger.info(f"📋 [CSV_CACHE] Model type changed ({processing_info.get('model_type')} -> {model_type}), cache invalid")
            return False, cache_paths, None
        
        # 🔍 원본 파일 변경 확인 - 데이터 확장 고려
        original_info = metadata.get('original_file', {})
        original_hash = original_info.get('content_hash')
        current_hash = get_data_content_hash(file_path)
        
        if original_hash == current_hash:
            # 해시가 동일하면 캐시 유효
            logger.info(f"✅ [CSV_CACHE] Identical file hash, cache valid for {os.path.basename(file_path)}")
            return True, cache_paths, None
        
        # 🚀 해시가 다른 경우: 데이터 확장인지 확인
        logger.info(f"🔍 [CSV_CACHE] File hash changed, checking for data extension...")
        
        # 이전 캐시된 파일 경로 추출 (메타데이터에서)
        original_file_path = original_info.get('path')
        if not original_file_path or not os.path.exists(original_file_path):
            logger.info(f"📋 [CSV_CACHE] Original file path not found, treating as new file")
            return False, cache_paths, None
        
        # 데이터 확장 검사
        try:
            extension_result = check_data_extension(original_file_path, file_path)
            
            if extension_result.get('is_extension', False):
                # 🎯 데이터 확장으로 확인됨 - 캐시 재활용 가능
                logger.info(f"✅ [CSV_CACHE] Data extension detected!")
                logger.info(f"    📈 Extension type: {extension_result.get('validation_details', {}).get('extension_type', 'Unknown')}")
                logger.info(f"    ➕ New rows: {extension_result.get('new_rows_count', 0)}")
                logger.info(f"    📅 Old range: {extension_result.get('old_start_date')} ~ {extension_result.get('old_end_date')}")
                logger.info(f"    📅 New range: {extension_result.get('new_start_date')} ~ {extension_result.get('new_end_date')}")
                
                # 확장된 데이터의 경우 캐시를 부분적으로 유효한 것으로 처리
                # 기존 캐시 + 새로운 데이터 부분만 처리하도록 정보 제공
                return "extension", cache_paths, extension_result
            else:
                # 확장이 아닌 완전히 다른 데이터
                logger.info(f"📋 [CSV_CACHE] File changed but not a data extension, cache invalid")
                logger.info(f"    Reason: {extension_result.get('validation_details', {}).get('reason', 'Unknown')}")
                return False, cache_paths, None
                
        except Exception as ext_error:
            logger.warning(f"⚠️ [CSV_CACHE] Extension check failed: {str(ext_error)}")
            logger.info(f"📋 [CSV_CACHE] Treating as file change, cache invalid")
            return False, cache_paths, None
        
    except Exception as e:
        logger.warning(f"⚠️ [CSV_CACHE] Error checking cache validity: {str(e)}")
        return False, None, None

def create_csv_cache_from_excel(file_path, model_type=None):
    """
    Excel 파일을 완전한 전처리 파이프라인을 거쳐 CSV 캐시로 생성하는 함수
    
    Args:
        file_path (str): Excel 파일 경로
        model_type (str): 모델 타입
    
    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    global _cache_creation_in_progress
    
    # 무한 루프 방지: 이미 캐시 생성 중이면 에러 발생
    if _cache_creation_in_progress:
        logger.error("❌ [CSV_CACHE] Cache creation already in progress - preventing infinite loop")
        raise RuntimeError("Cache creation already in progress to prevent infinite loop")
    
    logger.info(f"📊 [CSV_CACHE] Creating CSV cache from Excel with preprocessor: {os.path.basename(file_path)}")
    
    # 캐시 생성 시작 플래그 설정
    _cache_creation_in_progress = True
    
    # 🔒 보안 파일 처리 (확장자 물리적 변경 포함)
    processed_file_path, actual_ext, is_security_file = process_security_file_in_loader(file_path)
    
    # 처리된 파일 경로로 업데이트
    file_path = processed_file_path
    
    # 파일 타입 재확인
    actual_file_type, _ = normalize_security_extension(file_path)
    
    if is_security_file:
        logger.info(f"🔒 [CSV_CACHE] Security file processed: {actual_file_type}")
    
    processing_start_time = time.time()
    processing_info = {
        'start_time': processing_start_time,
        'file_type': actual_file_type,
        'is_security_file': is_security_file,
        'errors_encountered': [],
        'processing_method': 'unknown'
    }
    
    try:
        # 🚀 완전한 전처리 파이프라인을 통해 DataFrame 로딩
        logger.info("🔧 [CSV_CACHE] Using complete preprocessing pipeline...")
        df = load_excel_as_dataframe(file_path, model_type)
        
        if df is None or df.empty:
            raise ValueError("Failed to load Excel file - empty or None DataFrame returned")
        
        processing_info['processing_method'] = 'preprocessor_pipeline'
        processing_info['preprocessing_success'] = True
        
        # 로딩 성공 후 정보 수집
        processing_info['end_time'] = time.time()
        processing_info['duration_seconds'] = processing_info['end_time'] - processing_start_time
        processing_info['final_shape'] = list(df.shape)
        processing_info['date_range'] = [df.index.min().strftime('%Y-%m-%d'), df.index.max().strftime('%Y-%m-%d')]
        processing_info['columns_count'] = len(df.columns)
        processing_info['columns_sample'] = list(df.columns)[:10]  # 처음 10개 컬럼만
        
        logger.info(f"📊 [CSV_CACHE] Preprocessor pipeline completed: {df.shape} ({processing_info['duration_seconds']:.2f}s)")
        logger.info(f"📅 [CSV_CACHE] Date range: {processing_info['date_range'][0]} ~ {processing_info['date_range'][1]}")
        logger.info(f"📋 [CSV_CACHE] Columns: {processing_info['columns_count']} total")
        
        # 데이터 품질 검증
        missing_count = df.isnull().sum().sum()
        inf_count = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            inf_count = np.isinf(df[numeric_cols].values).sum()
        
        processing_info['data_quality'] = {
            'missing_values': int(missing_count),
            'infinite_values': int(inf_count),
            'numeric_columns': len(numeric_cols),
            'total_columns': len(df.columns)
        }
        
        if missing_count > 0:
            logger.info(f"📊 [CSV_CACHE] Data quality: {missing_count} missing values, {inf_count} infinite values")
        else:
            logger.info(f"✅ [CSV_CACHE] Data quality: No missing or infinite values")
        
        # CSV 캐시 저장
        cache_paths = get_csv_cache_path(file_path, model_type)
        csv_file = cache_paths['csv_file']
        metadata_file = cache_paths['metadata_file']
        
        # Date 인덱스를 컬럼으로 변환하여 CSV 저장
        df_for_csv = df.reset_index()
        
        logger.info(f"💾 [CSV_CACHE] Saving processed data to CSV: {csv_file}")
        
        # CSV 저장 (UTF-8 인코딩, 인덱스 제외)
        df_for_csv.to_csv(csv_file, index=False, encoding='utf-8')
        
        # 저장된 파일 크기 확인
        csv_file_size = os.path.getsize(csv_file) / (1024 * 1024)  # MB
        processing_info['csv_file_size_mb'] = round(csv_file_size, 2)
        
        # 메타데이터 저장
        metadata = create_csv_cache_metadata(file_path, model_type, processing_info)
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ [CSV_CACHE] Cache created successfully:")
        logger.info(f"  📁 CSV file: {csv_file} ({processing_info['csv_file_size_mb']} MB)")
        logger.info(f"  📋 Metadata: {metadata_file}")
        logger.info(f"  ⏱️ Processing time: {processing_info['duration_seconds']:.2f}s")
        
        # 캐시 생성 완료 플래그 해제
        _cache_creation_in_progress = False
        
        return df
        
    except Exception as e:
        processing_info['end_time'] = time.time()
        processing_info['duration_seconds'] = processing_info['end_time'] - processing_start_time
        processing_info['final_error'] = str(e)
        processing_info['preprocessing_success'] = False
        
        logger.error(f"❌ [CSV_CACHE] Failed to create cache from Excel:")
        logger.error(f"  📁 File: {os.path.basename(file_path)}")
        logger.error(f"  🔴 Error: {str(e)}")
        logger.error(f"  ⏱️ Duration: {processing_info['duration_seconds']:.2f}s")
        
        # 실패한 경우에도 메타데이터는 저장 (디버깅용)
        try:
            cache_paths = get_csv_cache_path(file_path, model_type)
            metadata_file = cache_paths['metadata_file']
            
            # 실패 정보가 포함된 메타데이터 저장
            error_metadata = create_csv_cache_metadata(file_path, model_type, processing_info)
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(error_metadata, f, indent=2, ensure_ascii=False)
                
            logger.info(f"📋 [CSV_CACHE] Error metadata saved for debugging: {metadata_file}")
        except Exception as meta_error:
            logger.warning(f"⚠️ [CSV_CACHE] Failed to save error metadata: {str(meta_error)}")
        
        # 캐시 생성 실패 플래그 해제
        _cache_creation_in_progress = False
        
        raise e

def load_csv_cache(file_path, model_type=None):
    """
    CSV 캐시를 안전하게 로딩하는 함수 (pandas 우선, xlwings는 fallback)
    
    Args:
        file_path (str): 원본 파일 경로
        model_type (str): 모델 타입
    
    Returns:
        pd.DataFrame: 캐시된 데이터프레임
    """
    cache_paths = get_csv_cache_path(file_path, model_type)
    csv_file = cache_paths['csv_file']
    
    logger.info(f"📖 [CSV_CACHE] Loading cached CSV: {csv_file}")
    
    try:
        # pandas로 먼저 시도 (CSV는 pandas가 더 안정적)
        logger.info(f"📊 [CSV_CACHE] Attempting pandas CSV loading...")
        
        # 여러 구분자로 시도
        separators = [',', ';', '\t']
        df = None
        
        for sep in separators:
            try:
                df_test = pd.read_csv(csv_file, sep=sep, encoding='utf-8')
                # 단일 컬럼에 쉼표가 포함된 경우 체크
                if len(df_test.columns) == 1 and ',' in df_test.columns[0]:
                    logger.warning(f"⚠️ [CSV_CACHE] Single column with commas detected using separator '{sep}', trying next...")
                    continue
                else:
                    df = df_test
                    logger.info(f"✅ [CSV_CACHE] Successfully loaded with separator '{sep}': {df.shape}")
                    break
            except Exception as sep_error:
                logger.warning(f"⚠️ [CSV_CACHE] Failed with separator '{sep}': {str(sep_error)}")
                continue
        
        # pandas로 성공하지 못한 경우 xlwings 시도
        if df is None and XLWINGS_AVAILABLE:
            logger.info(f"🔄 [CSV_CACHE] Pandas failed, trying xlwings...")
            try:
                df = load_csv_with_xlwings(str(csv_file))
                
                # xlwings로 읽어도 단일 컬럼 문제가 있는지 확인
                if len(df.columns) == 1 and ',' in df.columns[0]:
                    logger.warning(f"⚠️ [CSV_CACHE] xlwings also returned single column issue, attempting manual parsing...")
                    # 수동으로 CSV 파싱 시도
                    with open(csv_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # 첫 번째 줄을 파싱하여 컬럼 확인
                    if lines:
                        header = lines[0].strip()
                        if ',' in header:
                            # 쉼표로 분리된 것으로 판단하고 재시도
                            df = pd.read_csv(csv_file, sep=',', encoding='utf-8')
                            logger.info(f"✅ [CSV_CACHE] Manual parsing successful: {df.shape}")
                        
            except Exception as xlwings_error:
                logger.warning(f"⚠️ [CSV_CACHE] xlwings also failed: {str(xlwings_error)}")
        
        # 모든 방법이 실패한 경우
        if df is None:
            raise ValueError("Failed to load CSV cache with both pandas and xlwings")
        
        # 컬럼 확인 및 로깅
        logger.info(f"📋 [CSV_CACHE] Loaded columns: {list(df.columns)}")
        
        # Date 컬럼 처리
        if 'Date' in df.columns:
            logger.info(f"✅ [CSV_CACHE] Date column found, converting to datetime...")
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            logger.info(f"📅 [CSV_CACHE] Date range: {df.index.min()} ~ {df.index.max()}")
        else:
            logger.error(f"❌ [DATE_COLUMN] Date column not found. Available columns: {list(df.columns)}")
            # 첫 번째 컬럼이 날짜 형식인지 확인
            first_col = df.columns[0]
            try:
                # 첫 번째 컬럼의 몇 개 값을 날짜로 파싱해보기
                test_dates = pd.to_datetime(df[first_col].head(5), errors='coerce')
                if test_dates.notna().sum() >= 3:  # 5개 중 3개 이상이 유효한 날짜면
                    logger.info(f"🔄 [DATE_COLUMN] Using first column '{first_col}' as Date")
                    df = df.rename(columns={first_col: 'Date'})
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                else:
                    logger.error(f"❌ [DATE_COLUMN] First column is not date format: {df[first_col].iloc[0]}")
                    raise ValueError(f"No valid Date column found in cached CSV")
            except Exception as date_error:
                logger.error(f"❌ [DATE_COLUMN] First column date parsing failed: {str(date_error)}")
                raise ValueError(f"No valid Date column found in cached CSV")
        
        logger.info(f"✅ [CSV_CACHE] Cache loaded successfully: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"❌ [CSV_CACHE] Failed to load cache: {str(e)}")
        raise e

def load_excel_as_dataframe(file_path, model_type=None):
    """
    Excel 파일을 DataFrame으로 로딩하는 함수 (preprocessor 전처리 파이프라인 사용)
    
    Args:
        file_path (str): Excel 파일 경로
        model_type (str): 모델 타입
    
    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    logger.info(f"📊 [EXCEL_TO_DF] Loading Excel as DataFrame with preprocessor: {os.path.basename(file_path)}")
    
    # 🔒 보안 파일 처리 (확장자 물리적 변경 포함)
    processed_file_path, actual_ext, is_security_file = process_security_file_in_loader(file_path)
    
    # 처리된 파일 경로로 업데이트
    file_path = processed_file_path
    
    # 파일 타입 재확인
    actual_file_type, _ = normalize_security_extension(file_path)
    
    if is_security_file:
        logger.info(f"🔒 [EXCEL_TO_DF] Security file processed: {actual_file_type}")
    
    df = None
    
    try:
        # 🚀 1단계: preprocessor의 완전한 전처리 파이프라인 우선 시도
        logger.info("🔧 [EXCEL_TO_DF] Attempting complete preprocessing pipeline...")
        try:
            # 적절한 시트 이름 결정
            sheet_name = '29 Nov 2010 till todate'
            try:
                # 시트 목록 확인 (보안 파일도 지원하도록 안전한 방식 사용)
                if XLWINGS_AVAILABLE:
                    try:
                        # xlwings로 시트 목록 확인 시도
                        import xlwings as xw
                        app = xw.App(visible=False, add_book=False)
                        wb = app.books.open(file_path, read_only=True)
                        available_sheets = [sheet.name for sheet in wb.sheets]
                        wb.close()
                        app.quit()
                        logger.info(f"📋 [EXCEL_TO_DF] Available sheets (xlwings): {available_sheets}")
                    except:
                        # xlwings 실패 시 pandas로 시트 확인
                        excel_file = pd.ExcelFile(file_path, engine='openpyxl')
                        available_sheets = excel_file.sheet_names
                        logger.info(f"📋 [EXCEL_TO_DF] Available sheets (pandas): {available_sheets}")
                else:
                    excel_file = pd.ExcelFile(file_path, engine='openpyxl')
                    available_sheets = excel_file.sheet_names
                    logger.info(f"📋 [EXCEL_TO_DF] Available sheets: {available_sheets}")
                
                if sheet_name not in available_sheets:
                    sheet_name = available_sheets[0]
                    logger.info(f"🎯 [EXCEL_TO_DF] Using first sheet: {sheet_name}")
                else:
                    logger.info(f"🎯 [EXCEL_TO_DF] Using target sheet: {sheet_name}")
            except Exception as sheet_error:
                logger.warning(f"⚠️ [EXCEL_TO_DF] Sheet detection failed: {str(sheet_error)}")
                sheet_name = 0  # 인덱스로 첫 번째 시트 사용
                logger.info(f"🎯 [EXCEL_TO_DF] Using first sheet by index: {sheet_name}")
            
            # 완전한 전처리 파이프라인 실행
            from app.data.preprocessor import process_excel_data_complete
            df = process_excel_data_complete(file_path, sheet_name, start_date='2013-01-04')
            
            if df is not None and not df.empty:
                logger.info("✅ [EXCEL_TO_DF] Complete preprocessing pipeline succeeded")
                logger.info(f"📊 [EXCEL_TO_DF] Preprocessed data shape: {df.shape}")
                logger.info(f"📋 [EXCEL_TO_DF] Columns: {len(df.columns)} - {list(df.columns)[:5]}...")
            else:
                raise ValueError("Preprocessor returned None or empty DataFrame")
                
        except Exception as preprocessor_error:
            logger.warning(f"⚠️ [EXCEL_TO_DF] Complete preprocessing failed: {str(preprocessor_error)}")
            logger.info("🔄 [EXCEL_TO_DF] Falling back to individual processing steps...")
            
            # 2단계: 개별 처리 단계로 fallback
            df = None
            
            # xlwings 시도
            if XLWINGS_AVAILABLE and actual_file_type in ['xlsx', 'xls', 'excel']:
                try:
                    logger.info("🔓 [EXCEL_TO_DF] Attempting Excel load with xlwings...")
                    df = load_data_with_xlwings(file_path, model_type)
                    logger.info("✅ [EXCEL_TO_DF] Excel loaded successfully with xlwings")
                except Exception as xlwings_error:
                    logger.warning(f"⚠️ [EXCEL_TO_DF] xlwings failed: {str(xlwings_error)}")
            
            # pandas fallback
            if df is None:
                logger.info("🔄 [EXCEL_TO_DF] Attempting pandas Excel loading...")
                try:
                    df = safe_read_excel(file_path, sheet_name=sheet_name)
                    # 더 견고한 날짜 파싱 - 잘못된 형식(예: "1 Mac 2011") 처리
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True, format='mixed')
                    # 파싱 실패한 날짜 제거
                    invalid_dates = df['Date'].isna().sum()
                    if invalid_dates > 0:
                        logger.warning(f"⚠️ {invalid_dates}개의 잘못된 날짜 형식을 발견하여 제거했습니다.")
                        df = df.dropna(subset=['Date'])
                    logger.info("✅ [EXCEL_TO_DF] Excel loaded successfully with pandas")
                except Exception as pandas_error:
                    error_msg = f"pandas also failed: {str(pandas_error)}"
                    logger.error(f"❌ [EXCEL_TO_DF] {error_msg}")
                    raise Exception(f"All loading methods failed. Preprocessor: {preprocessor_error}, Pandas: {pandas_error}")
        
        # Date 컬럼이 인덱스가 아닌 경우 인덱스로 설정
        if 'Date' in df.columns and df.index.name != 'Date':
            df.set_index('Date', inplace=True)
        elif df.index.name != 'Date' and 'Date' not in df.columns:
            # Date 컬럼도 인덱스도 아닌 경우 첫 번째 datetime 컬럼 찾기
            for col in df.columns:
                if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
                    df = df.rename(columns={col: 'Date'})
                    df.set_index('Date', inplace=True)
                    logger.info(f"🔄 [EXCEL_TO_DF] Set '{col}' as Date index")
                    break
        
        # 모델 타입별 필터링 적용
        if model_type == 'lstm':
            cutoff_date = pd.to_datetime('2022-01-01')
            original_shape = df.shape
            df = df[df.index >= cutoff_date]
            
            logger.info(f"📊 [EXCEL_TO_DF] LSTM filter applied: {original_shape[0]} -> {df.shape[0]} rows")
            
            if df.empty:
                raise ValueError("No data available after 2022-01-01 filter for LSTM model")
        
        # 최종 데이터 정제 (preprocessor가 실패한 경우에만)
        if df is not None:
            # 무한값 처리
            df = df.replace([np.inf, -np.inf], np.nan)
            # 결측치 처리 (전처리 파이프라인에서 이미 처리되었을 수 있음)
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                logger.info(f"🔧 [EXCEL_TO_DF] Handling {missing_count} remaining missing values...")
                df = df.ffill().bfill()
        
        logger.info(f"✅ [EXCEL_TO_DF] DataFrame created successfully: {df.shape}")
        logger.info(f"📅 [EXCEL_TO_DF] Date range: {df.index.min()} ~ {df.index.max()}")
        return df
        
    except Exception as e:
        logger.error(f"❌ [EXCEL_TO_DF] Failed to load Excel as DataFrame: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise e

def convert_excel_to_temp_csv(file_path, model_type=None, temp_dir=None):
    """
    Excel 파일을 완전한 전처리를 거쳐 임시 CSV로 변환하는 함수 (비교용)
    
    Args:
        file_path (str): Excel 파일 경로
        model_type (str): 모델 타입
        temp_dir (str): 임시 디렉토리 (None이면 시스템 기본값)
    
    Returns:
        tuple: (임시 CSV 파일 경로, DataFrame)
    """
    import tempfile
    
    logger.info(f"🔄 [TEMP_CSV] Converting Excel to temporary CSV with preprocessor: {os.path.basename(file_path)}")
    
    try:
        # 완전한 전처리 파이프라인을 통한 DataFrame 로딩 (캐시 생성 없이)
        logger.info(f"🔧 [TEMP_CSV] Using preprocessing pipeline for comparison...")
        df = load_excel_as_dataframe(file_path, model_type)
        
        if df is None or df.empty:
            raise ValueError("Failed to load Excel file for temporary CSV conversion")
        
        logger.info(f"📊 [TEMP_CSV] Preprocessed data loaded: {df.shape}")
        
        # 임시 CSV 파일 생성
        if temp_dir is None:
            temp_dir = tempfile.gettempdir()
        
        # 파일명에 타임스탬프와 모델타입 추가 (구분을 위해)
        timestamp = int(time.time())
        model_suffix = f"_{model_type}" if model_type else ""
        temp_csv_path = os.path.join(temp_dir, f"temp_{timestamp}{model_suffix}_{os.path.basename(file_path)}.csv")
        
        # Date 인덱스를 컬럼으로 변환하여 CSV 저장
        df_for_csv = df.reset_index()
        df_for_csv.to_csv(temp_csv_path, index=False, encoding='utf-8')
        
        # 임시 파일 크기 확인
        temp_file_size = os.path.getsize(temp_csv_path) / (1024 * 1024)  # MB
        
        logger.info(f"✅ [TEMP_CSV] Temporary CSV created:")
        logger.info(f"  📁 File: {temp_csv_path}")
        logger.info(f"  📊 Data: {df.shape} ({temp_file_size:.2f} MB)")
        logger.info(f"  📅 Date range: {df.index.min()} ~ {df.index.max()}")
        
        return temp_csv_path, df
        
    except Exception as e:
        logger.error(f"❌ [TEMP_CSV] Failed to convert Excel to temporary CSV:")
        logger.error(f"  📁 File: {os.path.basename(file_path)}")
        logger.error(f"  🔴 Error: {str(e)}")
        raise e

def compare_csv_files(csv_file1, csv_file2, tolerance=1e-6):
    """
    두 CSV 파일의 내용을 비교하는 함수
    
    Args:
        csv_file1 (str): 첫 번째 CSV 파일 경로
        csv_file2 (str): 두 번째 CSV 파일 경로
        tolerance (float): 수치 비교 허용 오차
    
    Returns:
        dict: 비교 결과 정보
    """
    try:
        logger.info(f"🔍 [CSV_COMPARE] Comparing CSV files:")
        logger.info(f"  📄 File 1: {os.path.basename(csv_file1)}")
        logger.info(f"  📄 File 2: {os.path.basename(csv_file2)}")
        
        # CSV 파일 로딩 (xlwings 사용)
        if XLWINGS_AVAILABLE:
            df1 = load_csv_safe_with_fallback(csv_file1)
            df2 = load_csv_safe_with_fallback(csv_file2)
        else:
            df1 = pd.read_csv(csv_file1)
            df2 = pd.read_csv(csv_file2)
        
        # 기본 형태 비교
        if df1.shape != df2.shape:
            logger.info(f"📊 [CSV_COMPARE] Shape mismatch: {df1.shape} vs {df2.shape}")
            return {
                'is_identical': False,
                'is_extension': df2.shape[0] > df1.shape[0] and df2.shape[1] == df1.shape[1],
                'reason': f'Shape difference: {df1.shape} vs {df2.shape}',
                'shape1': df1.shape,
                'shape2': df2.shape
            }
        
        # 컬럼 비교
        if list(df1.columns) != list(df2.columns):
            logger.info(f"📋 [CSV_COMPARE] Column mismatch")
            return {
                'is_identical': False,
                'is_extension': False,
                'reason': 'Column structure difference',
                'columns1': list(df1.columns),
                'columns2': list(df2.columns)
            }
        
        # Date 컬럼 처리
        if 'Date' in df1.columns:
            df1['Date'] = pd.to_datetime(df1['Date'])
            df2['Date'] = pd.to_datetime(df2['Date'])
            
            # 날짜로 정렬
            df1 = df1.sort_values('Date').reset_index(drop=True)
            df2 = df2.sort_values('Date').reset_index(drop=True)
        
        # 내용 비교
        differences = []
        
        # 각 컬럼별로 비교
        for col in df1.columns:
            if col == 'Date':
                # 날짜 비교
                if not df1[col].equals(df2[col]):
                    date_diffs = df1[col] != df2[col]
                    diff_count = date_diffs.sum()
                    if diff_count > 0:
                        differences.append(f"Date column: {diff_count} differences")
            else:
                # 수치 데이터 비교
                try:
                    # 수치 변환 시도
                    col1_numeric = pd.to_numeric(df1[col], errors='coerce')
                    col2_numeric = pd.to_numeric(df2[col], errors='coerce')
                    
                    # NaN 개수 비교
                    nan1 = col1_numeric.isna().sum()
                    nan2 = col2_numeric.isna().sum()
                    
                    if nan1 != nan2:
                        differences.append(f"Column {col}: NaN count difference ({nan1} vs {nan2})")
                        continue
                    
                    # 수치 비교 (NaN 제외)
                    valid_mask = col1_numeric.notna() & col2_numeric.notna()
                    if valid_mask.sum() > 0:
                        valid1 = col1_numeric[valid_mask]
                        valid2 = col2_numeric[valid_mask]
                        
                        if not np.allclose(valid1, valid2, rtol=tolerance, atol=tolerance, equal_nan=True):
                            max_diff = np.abs(valid1 - valid2).max()
                            differences.append(f"Column {col}: numeric differences (max: {max_diff})")
                        
                except:
                    # 수치 변환 실패 시 문자열 비교
                    if not df1[col].equals(df2[col]):
                        diff_count = (df1[col] != df2[col]).sum()
                        differences.append(f"Column {col}: {diff_count} string differences")
        
        # 결과 판정
        is_identical = len(differences) == 0
        
        comparison_result = {
            'is_identical': is_identical,
            'is_extension': False,  # 같은 shape에서는 확장이 아님
            'reason': 'Identical files' if is_identical else f'{len(differences)} differences found',
            'differences': differences,
            'shape1': df1.shape,
            'shape2': df2.shape,
            'tolerance_used': tolerance
        }
        
        if is_identical:
            logger.info(f"✅ [CSV_COMPARE] Files are identical")
        else:
            logger.info(f"❌ [CSV_COMPARE] Files differ: {len(differences)} differences")
            for diff in differences[:3]:  # 처음 3개만 로깅
                logger.info(f"  - {diff}")
        
        return comparison_result
        
    except Exception as e:
        logger.error(f"❌ [CSV_COMPARE] Error comparing CSV files: {str(e)}")
        return {
            'is_identical': False,
            'is_extension': False,
            'reason': f'Comparison error: {str(e)}',
            'error': str(e)
        }

def check_data_extension_csv_based(existing_excel_path, new_excel_path, model_type=None):
    """
    CSV 기반으로 데이터 확장을 확인하는 함수
    
    Args:
        existing_excel_path (str): 기존 Excel 파일 경로
        new_excel_path (str): 새 Excel 파일 경로
        model_type (str): 모델 타입
    
    Returns:
        dict: 확장 정보
    """
    temp_csv1 = None
    temp_csv2 = None
    
    try:
        logger.info(f"🔍 [CSV_EXTENSION] Checking data extension using CSV conversion")
        logger.info(f"  📄 Existing: {os.path.basename(existing_excel_path)}")
        logger.info(f"  📄 New: {os.path.basename(new_excel_path)}")
        
        # 두 Excel 파일을 임시 CSV로 변환
        temp_csv1, df1 = convert_excel_to_temp_csv(existing_excel_path, model_type)
        temp_csv2, df2 = convert_excel_to_temp_csv(new_excel_path, model_type)
        
        # 기존 check_data_extension 로직을 DataFrame으로 적용
        result = check_dataframes_extension(df1, df2, existing_excel_path, new_excel_path)
        
        # CSV 기반 처리 정보 추가
        result['csv_based_comparison'] = True
        result['temp_csv_paths'] = [temp_csv1, temp_csv2]
        
        return result
        
    except Exception as e:
        logger.error(f"❌ [CSV_EXTENSION] CSV-based extension check failed: {str(e)}")
        return {
            'is_extension': False,
            'new_rows_count': 0,
            'csv_based_comparison': True,
            'validation_details': {'error': str(e)}
        }
        
    finally:
        # 임시 파일 정리
        for temp_csv in [temp_csv1, temp_csv2]:
            if temp_csv and os.path.exists(temp_csv):
                try:
                    os.remove(temp_csv)
                    logger.info(f"🗑️ [CSV_EXTENSION] Cleaned up temporary file: {os.path.basename(temp_csv)}")
                except:
                    logger.warning(f"⚠️ [CSV_EXTENSION] Failed to clean up: {temp_csv}")

def check_dataframes_extension(old_df, new_df, old_file_path, new_file_path):
    """
    DataFrame 기반으로 데이터 확장을 확인하는 함수 (기존 check_data_extension 로직 재활용)
    
    Args:
        old_df (pd.DataFrame): 기존 데이터프레임
        new_df (pd.DataFrame): 새 데이터프레임
        old_file_path (str): 기존 파일 경로 (로깅용)
        new_file_path (str): 새 파일 경로 (로깅용)
    
    Returns:
        dict: 확장 정보
    """
    try:
        # Date 인덱스를 컬럼으로 변환 (비교용)
        if old_df.index.name == 'Date':
            old_df = old_df.reset_index()
        if new_df.index.name == 'Date':
            new_df = new_df.reset_index()
        
        # 날짜 컬럼 확인
        if 'Date' not in old_df.columns or 'Date' not in new_df.columns:
            return {
                'is_extension': False, 
                'new_rows_count': 0,
                'validation_details': {'error': 'No Date column found'}
            }
        
        # 날짜로 정렬
        old_df = old_df.sort_values('Date').reset_index(drop=True)
        new_df = new_df.sort_values('Date').reset_index(drop=True)
        
        # 날짜를 datetime으로 변환
        old_df['Date'] = pd.to_datetime(old_df['Date'])
        new_df['Date'] = pd.to_datetime(new_df['Date'])
        
        # 기본 정보 추출
        old_start_date = old_df['Date'].iloc[0]
        old_end_date = old_df['Date'].iloc[-1]
        new_start_date = new_df['Date'].iloc[0]
        new_end_date = new_df['Date'].iloc[-1]
        
        logger.info(f"🔍 [DF_EXTENSION] Old data: {old_start_date.strftime('%Y-%m-%d')} ~ {old_end_date.strftime('%Y-%m-%d')} ({len(old_df)} rows)")
        logger.info(f"🔍 [DF_EXTENSION] New data: {new_start_date.strftime('%Y-%m-%d')} ~ {new_end_date.strftime('%Y-%m-%d')} ({len(new_df)} rows)")
        
        # 확장 검증 (기존 로직 재사용)
        if len(new_df) <= len(old_df):
            logger.info(f"❌ [DF_EXTENSION] New file is not longer ({len(new_df)} <= {len(old_df)})")
            return {
                'is_extension': False, 
                'new_rows_count': 0,
                'old_start_date': old_start_date.strftime('%Y-%m-%d'),
                'old_end_date': old_end_date.strftime('%Y-%m-%d'),
                'new_start_date': new_start_date.strftime('%Y-%m-%d'),
                'new_end_date': new_end_date.strftime('%Y-%m-%d'),
                'validation_details': {'reason': 'New file is not longer than old file'}
            }
        
        # 새 데이터가 기존 데이터보다 더 많은 정보를 포함하는지 확인
        has_more_data = (new_start_date < old_start_date) or (new_end_date > old_end_date) or (len(new_df) > len(old_df))
        if not has_more_data:
            logger.info(f"❌ [DF_EXTENSION] New data doesn't provide additional information")
            return {
                'is_extension': False,
                'new_rows_count': 0,
                'old_start_date': old_start_date.strftime('%Y-%m-%d'),
                'old_end_date': old_end_date.strftime('%Y-%m-%d'),
                'new_start_date': new_start_date.strftime('%Y-%m-%d'),
                'new_end_date': new_end_date.strftime('%Y-%m-%d'),
                'validation_details': {'reason': 'New data does not provide additional information'}
            }
        
        # 기존 데이터의 모든 날짜가 새 데이터에 포함되어야 함
        old_dates = set(old_df['Date'].dt.strftime('%Y-%m-%d'))
        new_dates = set(new_df['Date'].dt.strftime('%Y-%m-%d'))
        
        missing_dates = old_dates - new_dates
        if missing_dates:
            logger.info(f"❌ [DF_EXTENSION] Some old dates are missing in new data: {len(missing_dates)} dates")
            return {
                'is_extension': False,
                'new_rows_count': 0,
                'old_start_date': old_start_date.strftime('%Y-%m-%d'),
                'old_end_date': old_end_date.strftime('%Y-%m-%d'),
                'new_start_date': new_start_date.strftime('%Y-%m-%d'),
                'new_end_date': new_end_date.strftime('%Y-%m-%d'),
                'validation_details': {'reason': f'Missing {len(missing_dates)} dates from old data'}
            }
        
        # 컬럼 구조 확인
        if list(old_df.columns) != list(new_df.columns):
            logger.info(f"❌ [DF_EXTENSION] Column structure differs")
            return {
                'is_extension': False,
                'new_rows_count': 0,
                'validation_details': {'reason': 'Column structure differs'}
            }
        
        # 확장 유형 분석
        new_only_dates = new_dates - old_dates
        extension_type = []
        if new_start_date < old_start_date:
            past_dates = len([d for d in new_only_dates if pd.to_datetime(d) < old_start_date])
            extension_type.append(f"과거 데이터 {past_dates}개 추가")
        if new_end_date > old_end_date:
            future_dates = len([d for d in new_only_dates if pd.to_datetime(d) > old_end_date])
            extension_type.append(f"미래 데이터 {future_dates}개 추가")
        
        extension_desc = " + ".join(extension_type) if extension_type else "데이터 보완"
        
        # 모든 검증 통과: 데이터 확장으로 인정
        new_rows_count = len(new_only_dates)
        
        logger.info(f"✅ [DF_EXTENSION] Valid data extension: {extension_desc} (+{new_rows_count} new dates)")
        
        return {
            'is_extension': True,
            'new_rows_count': new_rows_count,
            'old_start_date': old_start_date.strftime('%Y-%m-%d'),
            'old_end_date': old_end_date.strftime('%Y-%m-%d'),
            'new_start_date': new_start_date.strftime('%Y-%m-%d'),
            'new_end_date': new_end_date.strftime('%Y-%m-%d'),
            'validation_details': {
                'reason': f'Valid data extension: {extension_desc}',
                'new_dates_added': sorted(list(new_only_dates)),
                'extension_type': extension_type
            }
        }
        
    except Exception as e:
        logger.error(f"DataFrame extension check failed: {str(e)}")
        return {
            'is_extension': False, 
            'new_rows_count': 0,
            'validation_details': {'error': str(e)}
        }

def process_security_file_in_loader(file_path):
    """
    loader.py용 보안 파일 처리 함수 - 실제 파일 확장자를 물리적으로 변경
    
    Args:
        file_path (str): 원본 파일 경로
    
    Returns:
        tuple: (처리된 파일 경로, 실제 확장자, 보안 파일 여부)
    """
    # 보안 확장자 확인
    actual_file_type, is_security_file = normalize_security_extension(file_path)
    
    if not is_security_file:
        # 보안 파일이 아니면 원본 경로 반환
        original_ext = os.path.splitext(file_path.lower())[1]
        return file_path, original_ext, False
    
    logger.info(f"🔒 [LOADER_SECURITY] Processing security file: {os.path.basename(file_path)}")
    
    # 파일 내용으로 실제 타입 감지 (필요한 경우)
    if actual_file_type is None:
        detected_type = detect_file_type_by_content(file_path)
        if detected_type:
            actual_file_type = detected_type
            logger.info(f"📊 [LOADER_CONTENT_DETECTION] Detected file type: {detected_type}")
        else:
            logger.error(f"❌ [LOADER_SECURITY] Cannot determine file type for: {os.path.basename(file_path)}")
            return file_path, None, True
    
    # 새로운 확장자 결정
    if actual_file_type == 'csv':
        new_ext = '.csv'
    elif actual_file_type in ['xlsx', 'excel']:
        new_ext = '.xlsx'
    elif actual_file_type == 'xls':
        new_ext = '.xls'
    else:
        logger.warning(f"⚠️ [LOADER_SECURITY] Unsupported file type: {actual_file_type}")
        return file_path, None, True
    
    # 새로운 파일 경로 생성
    base_path = os.path.splitext(file_path)[0]
    new_file_path = f"{base_path}{new_ext}"
    
    # 파일명이 이미 올바른 확장자인 경우
    if new_file_path == file_path:
        logger.info(f"📋 [LOADER_SECURITY] File already has correct extension: {os.path.basename(file_path)}")
        return file_path, new_ext, True
    
    # 파일 이름 변경 (확장자 수정)
    try:
        shutil.move(file_path, new_file_path)
        logger.info(f"📝 [LOADER_SECURITY] File extension corrected: {os.path.basename(file_path)} -> {os.path.basename(new_file_path)}")
        return new_file_path, new_ext, True
    except Exception as e:
        logger.warning(f"⚠️ [LOADER_SECURITY] Failed to rename file: {str(e)}")
        # 파일명 변경에 실패해도 원본 경로로 계속 진행
        return file_path, new_ext, True