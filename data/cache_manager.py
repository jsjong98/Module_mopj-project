import os
import json
import csv
import hashlib
import pandas as pd
import numpy as np
import logging
import shutil
import traceback
from pathlib import Path
from datetime import datetime
import time
import os
import csv

from app.config import CACHE_ROOT_DIR, UPLOAD_FOLDER
from app.utils.date_utils import get_semimonthly_period, format_date, is_holiday
from app.core.state_manager import prediction_state
from app.utils.serialization import safe_serialize_value, clean_interval_scores_safe, convert_to_legacy_format, clean_predictions_data

# 순환 import 방지를 위해 함수 내부에서 지연 import 사용

logger = logging.getLogger(__name__)

# 파일 해시 캐시 (메모리 캐싱으로 성능 최적화)
_file_hash_cache = {}
_cache_lookup_index = {}
_dataframe_cache = {}
_cache_expiry_seconds = 120

def get_file_cache_dirs(file_path=None):
    """
    🚀 통합 저장소 시스템: 파일과 무관하게 모든 것을 통합 관리
    
    이제 파일별 캐시 대신 통합 저장소를 사용합니다:
    - 모든 예측 결과 → app/predictions/
    - 모든 모델 → app/models/
    - 모든 플롯 → app/plots/
    
    기존 파일별 캐시 시스템과의 호환성을 위해 동일한 구조를 반환하지만
    실제로는 통합 디렉토리를 가리킵니다.
    """
    try:
        # 🌟 통합 저장소 사용 - 파일 경로와 무관
        logger.info(f"🌟 [UNIFIED_SYSTEM] Using unified storage system (file-agnostic)")
        
        return get_unified_storage_dirs()
        
    except Exception as e:
        logger.error(f"❌ Error in get_file_cache_dirs: {str(e)}")
        logger.error(traceback.format_exc())
        raise e  # 오류 발생 시 예외 전파
    
def calculate_file_hash(file_path, chunk_size=8192):
    """파일 내용의 SHA256 해시를 계산"""
    import hashlib
    
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"File hash calculation failed: {str(e)}")
        return None
    
def get_data_content_hash(file_path):
    """데이터 파일(CSV/Excel)의 전처리된 내용으로 해시 생성 (캐싱 최적화)"""
    import hashlib
    import os
    
    try:
        # 파일 수정 시간 기반 캐시 확인
        if file_path in _file_hash_cache:
            cached_mtime, cached_hash = _file_hash_cache[file_path]
            current_mtime = os.path.getmtime(file_path)
            
            # 파일이 수정되지 않았다면 캐시된 해시 반환
            if abs(current_mtime - cached_mtime) < 1.0:  # 1초 이내 차이는 무시
                logger.debug(f"📋 Using cached hash for {os.path.basename(file_path)}")
                return cached_hash
        
        # 파일이 수정되었거나 캐시가 없는 경우 새로 계산
        logger.info(f"🔄 Calculating new hash for {os.path.basename(file_path)}")
        
        # 🔧 순환 참조 방지: load_data 대신 파일 내용 해시 직접 계산
        # Excel 파일의 경우에도 파일 내용 해시를 사용하여 순환 참조 방지
        logger.info(f"🔄 Using file content hash for {os.path.basename(file_path)} (avoid circular reference)")
        file_content_hash = calculate_file_hash(file_path)
        if file_content_hash:
            file_hash = file_content_hash[:16]
            logger.info(f"✅ File-based hash calculated: {file_hash} for {os.path.basename(file_path)}")
        else:
            logger.error(f"❌ File hash calculation failed for {os.path.basename(file_path)}")
            return None
        
        # 캐시 저장
        _file_hash_cache[file_path] = (os.path.getmtime(file_path), file_hash)
        
        return file_hash
    except Exception as e:
        logger.error(f"Data content hash calculation failed: {str(e)}")
        # 해시 계산에 실패하면 파일 기본 해시를 사용
        try:
            fallback_hash = calculate_file_hash(file_path)
            if fallback_hash:
                return fallback_hash[:16]
        except Exception:
            pass
        return None
    
def build_cache_lookup_index():
    """캐시 디렉토리의 인덱스를 빌드하여 빠른 검색 가능"""
    global _cache_lookup_index
    
    try:
        _cache_lookup_index = {}
        cache_root = Path(CACHE_ROOT_DIR)
        
        if not cache_root.exists():
            return
        
        for file_dir in cache_root.iterdir():
            if not file_dir.is_dir() or file_dir.name == "default":
                continue
            
            predictions_dir = file_dir / 'predictions'
            if not predictions_dir.exists():
                continue
            
            prediction_files = list(predictions_dir.glob("prediction_start_*_meta.json"))
            
            for meta_file in prediction_files:
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                    
                    file_hash = meta_data.get('file_content_hash')
                    data_end_date = meta_data.get('data_end_date')
                    
                    if file_hash and data_end_date:
                        semimonthly = get_semimonthly_period(pd.to_datetime(data_end_date))
                        cache_key = f"{file_hash}_{semimonthly}"
                        
                        _cache_lookup_index[cache_key] = {
                            'meta_file': str(meta_file),
                            'predictions_dir': str(predictions_dir),
                            'data_end_date': data_end_date,
                            'semimonthly': semimonthly
                        }
                        
                except Exception:
                    continue
                    
        logger.info(f"📊 Built cache lookup index with {len(_cache_lookup_index)} entries")
        
    except Exception as e:
        logger.error(f"Failed to build cache lookup index: {str(e)}")
        _cache_lookup_index = {}

def refresh_cache_index():
    """캐시 인덱스를 새로고침 (새로운 캐시 파일이 생성된 후 호출)"""
    global _cache_lookup_index
    logger.info("🔄 Refreshing cache lookup index...")
    build_cache_lookup_index()

def clear_cache_memory():
    """메모리 캐시를 클리어 (메모리 절약용)"""
    global _file_hash_cache, _cache_lookup_index
    _file_hash_cache.clear()
    _cache_lookup_index.clear()
    logger.info("🧹 Cleared memory cache")

def check_existing_prediction(current_date, file_path=None):
    """
    파일별 디렉토리 구조에서 저장된 예측을 확인하고 불러오는 함수
    🎯 현재 파일의 디렉토리에서 우선 검색 (정확한 날짜 매칭만 사용)
    
    수정사항:
    - 반월 기간(semimonthly) 매칭 제거
    - 정확한 날짜 매칭만 허용하여 동일한 날짜의 캐시만 사용
    """
    try:
        # 현재 날짜(데이터 기준일)에서 첫 번째 예측 날짜 계산
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
        
        # 다음 영업일 찾기 (현재 날짜의 다음 영업일이 첫 번째 예측 날짜)
        next_date = current_date + pd.Timedelta(days=1)
        while next_date.weekday() >= 5 or is_holiday(next_date):
            next_date += pd.Timedelta(days=1)
        
        first_prediction_date = next_date
        date_str = first_prediction_date.strftime('%Y%m%d')
        
        logger.info(f"🔍 Checking cache for EXACT prediction date: {first_prediction_date.strftime('%Y-%m-%d')}")
        logger.info(f"  📅 Data end date: {current_date.strftime('%Y-%m-%d')}")
        logger.info(f"  📅 Expected prediction start: {first_prediction_date.strftime('%Y-%m-%d')}")
        logger.info(f"  📄 Expected filename pattern: prediction_start_{date_str}.*")
        
        # 🎯 1단계: 현재 파일의 캐시 디렉토리에서 정확한 날짜 매치로 캐시 찾기
        try:
            # 🔧 수정: 파일 경로를 명시적으로 전달
            cache_dirs = get_file_cache_dirs(file_path)
            file_predictions_dir = cache_dirs['predictions']
            
            logger.info(f"  📁 Cache directory: {cache_dirs['root']}")
            logger.info(f"  📁 Predictions directory: {file_predictions_dir}")
            logger.info(f"  📁 Directory exists: {file_predictions_dir.exists()}")
            
        except Exception as e:
            logger.error(f"❌ Failed to get cache directories: {str(e)}")
            return None
        
        if file_predictions_dir.exists():
            exact_csv = file_predictions_dir / f"prediction_start_{date_str}.cs"
            exact_meta = file_predictions_dir / f"prediction_start_{date_str}_meta.json"
            
            logger.info(f"  🔍 Looking for: {exact_csv}")
            logger.info(f"  🔍 CSV exists: {exact_csv.exists()}")
            logger.info(f"  🔍 Meta exists: {exact_meta.exists()}")
            
            if exact_csv.exists() and exact_meta.exists():
                from app.data.cache_manager import load_prediction_with_attention_from_csv_in_dir
                logger.info(f"✅ Found EXACT prediction cache in file directory: {exact_csv.name}")
                return load_prediction_with_attention_from_csv_in_dir(first_prediction_date, file_predictions_dir)
            
            # 해당 파일 디렉토리에서 다른 날짜의 예측 찾기
            logger.info("🔍 Searching for other predictions in file directory...")
            prediction_files = list(file_predictions_dir.glob("prediction_start_*_meta.json"))
            
            logger.info(f"  📋 Found {len(prediction_files)} prediction files:")
            for i, pf in enumerate(prediction_files):
                logger.info(f"    {i+1}. {pf.name}")
            
            # 🔧 수정: 반월 기간 매칭 제거 - 정확한 날짜만 허용
            logger.info("❌ No exact date match found in file directory")
            logger.info("  📝 Note: Only exact date matches are allowed (no approximate/semimonthly matching)")
        else:
            logger.warning(f"❌ Predictions directory does not exist: {file_predictions_dir}")
        
        # 🎯 2단계: 다른 파일들의 캐시에서 호환 가능한 예측 찾기
        current_file_path = file_path or prediction_state.get('current_file', None)
        if current_file_path:
            # 🔧 수정: 모든 기존 파일의 캐시 디렉토리 탐색
            upload_dir = Path(UPLOAD_FOLDER)
            existing_files = [f for f in upload_dir.glob('*.xlsx') if str(f) != current_file_path]
            
            logger.info(f"🔍 [EXACT_DATE_CACHE] Searching other files for EXACT date match: {len(existing_files)} files")
            
            for existing_file in existing_files:
                try:
                    # 기존 파일의 캐시 디렉토리 확인
                    existing_cache_dirs = get_file_cache_dirs(str(existing_file))
                    existing_predictions_dir = existing_cache_dirs['predictions']
                    
                    if existing_predictions_dir.exists():
                        # 동일한 반월 기간의 예측 파일 찾기
                        pattern = f"prediction_start_*_meta.json"
                        meta_files = list(existing_predictions_dir.glob(pattern))
                        
                        logger.info(f"    📁 {existing_file.name}: {len(meta_files)}개 예측 파일")
                        
                        # 🔧 수정: 정확한 날짜의 예측 파일만 찾기
                        exact_csv_other = existing_predictions_dir / f"prediction_start_{date_str}.cs"
                        exact_meta_other = existing_predictions_dir / f"prediction_start_{date_str}_meta.json"
                        
                        if exact_csv_other.exists() and exact_meta_other.exists():
                            logger.info(f"    🎯 Found EXACT date match in {existing_file.name}!")
                            logger.info(f"    📅 Exact prediction date: {first_prediction_date.strftime('%Y-%m-%d')}")
                            logger.info(f"    📄 Using file: {exact_csv_other.name}")
                            
                            return load_prediction_with_attention_from_csv_in_dir(first_prediction_date, existing_predictions_dir)
                        else:
                            logger.debug(f"    ❌ No exact date match in {existing_file.name}")
                except Exception as e:
                    logger.debug(f"    ⚠️ 캐시 디렉토리 접근 실패 {existing_file.name}: {str(e)}")
                    continue
                    
            logger.info("❌ No exact date match found in other files' caches")
            
        logger.info("❌ No EXACT prediction cache found - only exact date matches are allowed")
        return None
        
    except Exception as e:
        logger.error(f"❌ Error checking existing prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def check_data_extension(old_file_path, new_file_path):
    """
    새 파일이 기존 파일의 순차적 확장(기존 데이터 이후에만 새 행 추가)인지 엄격하게 확인
    
    ⚠️ 중요: 다음 경우만 확장으로 인정:
    1. 기존 데이터와 정확히 동일한 부분이 있음
    2. 새 데이터가 기존 데이터의 마지막 날짜 이후에만 추가됨
    3. 기존 데이터의 시작/중간 날짜가 변경되지 않음
    
    Returns:
    --------
    dict: {
        'is_extension': bool,
        'new_rows_count': int,
        'base_hash': str,  # 기존 데이터 부분의 해시
        'old_start_date': str,
        'old_end_date': str,
        'new_start_date': str,
        'new_end_date': str,
        'validation_details': dict
    }
    """
    try:
                # 파일 형식에 맞게 로드 (순환 참조 방지를 위해 직접 로드)
        def load_file_safely(filepath, is_new_file=False):
            file_ext = os.path.splitext(filepath.lower())[1]
            if file_ext == '.csv':
                return pd.read_csv(filepath)
            else:
                # Excel 파일인 경우 pandas를 직접 사용하여 순환 참조 방지
                try:
                    # 모든 시트명 확인하여 데이터가 있는 시트 찾기
                    excel_file = pd.ExcelFile(filepath, engine='openpyxl')
                    sheet_names = excel_file.sheet_names
                    logger.info(f"🔍 [EXTENSION_CHECK] Available sheets in {os.path.basename(filepath)}: {sheet_names}")
                    
                    # 실제 데이터가 있는 시트 찾기 - 특정 패턴 우선 확인
                    target_sheet_patterns = [
                        '29 Nov 2010 till todate',  # 실제 데이터 시트
                        'till todate',
                        'data'
                    ]
                    
                    target_sheet = None
                    # 패턴 매칭 시도
                    for pattern in target_sheet_patterns:
                        for sheet_name in sheet_names:
                            if pattern.lower() in sheet_name.lower():
                                target_sheet = sheet_name
                                logger.info(f"📋 [EXTENSION_CHECK] Found target sheet by pattern '{pattern}': {target_sheet}")
                                break
                        if target_sheet:
                            break
                    
                    # 패턴으로 찾지 못하면 첫 번째 시트 사용
                    if not target_sheet:
                        target_sheet = sheet_names[0]
                        logger.info(f"📋 [EXTENSION_CHECK] Using first sheet: {target_sheet}")
                    
                    # 선택된 시트에서 데이터 로드
                    df = pd.read_excel(filepath, sheet_name=target_sheet, engine='openpyxl')
                    logger.info(f"📊 [EXTENSION_CHECK] Loaded sheet '{target_sheet}': {df.shape}")
                    
                    # Date 컬럼 찾기 및 파싱
                    date_col = None
                    for col in df.columns:
                        if 'date' in str(col).lower() or col == 'Date':
                            date_col = col
                            break
                    
                    if date_col and len(df) > 0:
                        logger.info(f"📅 [EXTENSION_CHECK] Found date column: {date_col}")
                        # 견고한 날짜 파싱 - 잘못된 형식 처리
                        df['Date'] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True, format='mixed')
                        # 파싱 실패한 날짜 제거
                        invalid_dates = df['Date'].isna().sum()
                        if invalid_dates > 0:
                            logger.warning(f"⚠️ [EXTENSION_CHECK] {invalid_dates}개의 잘못된 날짜 형식을 발견하여 제거했습니다.")
                            df = df.dropna(subset=['Date'])
                        
                        logger.info(f"📅 [EXTENSION_CHECK] Date range after parsing: {df['Date'].min()} ~ {df['Date'].max()}")
                        return df
                    else:
                        logger.warning(f"⚠️ [EXTENSION_CHECK] No date column found in {os.path.basename(filepath)}")
                        return df
                    
                except Exception as e:
                    logger.warning(f"⚠️ [EXTENSION_CHECK] Failed to load Excel file {filepath}: {e}")
                    # 빈 DataFrame 반환 (확장 체크 실패)
                    return pd.DataFrame()
        
        logger.info(f"🔍 [EXTENSION_CHECK] Loading data files for comparison...")
        old_df = load_file_safely(old_file_path, is_new_file=False)
        new_df = load_file_safely(new_file_path, is_new_file=True)
        
        # 날짜 컬럼이 있는지 확인
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
        
        logger.info(f"🔍 [EXTENSION_CHECK] Old data: {old_start_date.strftime('%Y-%m-%d')} ~ {old_end_date.strftime('%Y-%m-%d')} ({len(old_df)} rows)")
        logger.info(f"🔍 [EXTENSION_CHECK] New data: {new_start_date.strftime('%Y-%m-%d')} ~ {new_end_date.strftime('%Y-%m-%d')} ({len(new_df)} rows)")
        
        # ✅ 검증 1: 새 파일이 더 길어야 함
        if len(new_df) <= len(old_df):
            logger.info(f"❌ [EXTENSION_CHECK] New file is not longer ({len(new_df)} <= {len(old_df)})")
            return {
                'is_extension': False, 
                'new_rows_count': 0,
                'old_start_date': old_start_date.strftime('%Y-%m-%d'),
                'old_end_date': old_end_date.strftime('%Y-%m-%d'),
                'new_start_date': new_start_date.strftime('%Y-%m-%d'),
                'new_end_date': new_end_date.strftime('%Y-%m-%d'),
                'validation_details': {'reason': 'New file is not longer than old file'}
            }
        
        # ✅ 검증 2: 새 파일이 더 길거나 최소한 같은 길이여야 함 (과거 데이터 허용)
        # 과거 데이터가 포함된 경우도 허용하도록 변경
        logger.info(f"📅 [EXTENSION_CHECK] Date ranges - Old: {old_start_date} ~ {old_end_date}, New: {new_start_date} ~ {new_end_date}")
        
        # ✅ 검증 3: 새 데이터가 기존 데이터보다 더 많은 정보를 포함해야 함 (완화된 조건)
        # 과거 데이터 확장 또는 미래 데이터 확장 둘 다 허용
        has_more_data = (new_start_date < old_start_date) or (new_end_date > old_end_date) or (len(new_df) > len(old_df))
        if not has_more_data:
            logger.info(f"❌ [EXTENSION_CHECK] New data doesn't provide additional information")
            return {
                'is_extension': False,
                'new_rows_count': 0,
                'old_start_date': old_start_date.strftime('%Y-%m-%d'),
                'old_end_date': old_end_date.strftime('%Y-%m-%d'),
                'new_start_date': new_start_date.strftime('%Y-%m-%d'),
                'new_end_date': new_end_date.strftime('%Y-%m-%d'),
                'validation_details': {'reason': 'New data does not provide additional information beyond existing data'}
            }
        
        # ✅ 검증 4: 기존 데이터의 모든 날짜가 새 데이터에 포함되어야 함
        old_dates = set(old_df['Date'].dt.strftime('%Y-%m-%d'))
        new_dates = set(new_df['Date'].dt.strftime('%Y-%m-%d'))
        
        missing_dates = old_dates - new_dates
        if missing_dates:
            logger.info(f"❌ [EXTENSION_CHECK] Some old dates are missing in new data: {missing_dates}")
            return {
                'is_extension': False,
                'new_rows_count': 0,
                'old_start_date': old_start_date.strftime('%Y-%m-%d'),
                'old_end_date': old_end_date.strftime('%Y-%m-%d'),
                'new_start_date': new_start_date.strftime('%Y-%m-%d'),
                'new_end_date': new_end_date.strftime('%Y-%m-%d'),
                'validation_details': {'reason': f'Missing dates from old data: {list(missing_dates)}'}
            }
        
        # ✅ 검증 5: 컬럼이 동일해야 함
        if list(old_df.columns) != list(new_df.columns):
            logger.info(f"❌ [EXTENSION_CHECK] Column structure differs")
            return {
                'is_extension': False,
                'new_rows_count': 0,
                'old_start_date': old_start_date.strftime('%Y-%m-%d'),
                'old_end_date': old_end_date.strftime('%Y-%m-%d'),
                'new_start_date': new_start_date.strftime('%Y-%m-%d'),
                'new_end_date': new_end_date.strftime('%Y-%m-%d'),
                'validation_details': {'reason': 'Column structure differs'}
            }
        
        # ✅ 검증 6: 기존 데이터 부분이 정확히 동일한지 확인 (관대한 조건으로 완화)
        logger.info(f"🔍 [EXTENSION_CHECK] Comparing overlapping data...")
        logger.info(f"  📊 Checking {len(old_df)} existing dates...")
        
        # 🔧 관대한 확장 검증: 샘플링 방식으로 변경 (전체 데이터가 아닌 일부만 검사)
        sample_size = min(50, len(old_df))  # 최대 50개 날짜만 검사
        sample_indices = list(range(0, len(old_df), max(1, len(old_df) // sample_size)))
        
        logger.info(f"  🔬 Sampling {len(sample_indices)} dates out of {len(old_df)} for validation...")
        
        # 기존 데이터의 각 날짜에 해당하는 새 데이터 행 찾기
        data_matches = True
        mismatch_details = []
        checked_dates = 0
        mismatched_dates = 0
        allowed_mismatches = max(1, len(sample_indices) // 10)  # 10% 정도의 미스매치는 허용
        
        for idx in sample_indices:
            if idx >= len(old_df):
                continue
                
            old_row = old_df.iloc[idx]
            old_date = old_row['Date']
            old_date_str = old_date.strftime('%Y-%m-%d')
            checked_dates += 1
            
            # 새 데이터에서 해당 날짜 찾기
            new_matching_rows = new_df[new_df['Date'] == old_date]
            
            if len(new_matching_rows) == 0:
                data_matches = False
                mismatch_details.append(f"Date {old_date_str} missing in new data")
                break
            elif len(new_matching_rows) > 1:
                data_matches = False
                mismatch_details.append(f"Duplicate date {old_date_str} in new data")
                break
            
            new_row = new_matching_rows.iloc[0]
            
            # 수치 컬럼 비교 (Date 제외) - 완화된 조건
            numeric_cols = old_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                old_val = old_row[col]
                new_val = new_row[col]
                
                # NaN 값 처리
                if pd.isna(old_val) and pd.isna(new_val):
                    continue
                elif pd.isna(old_val) or pd.isna(new_val):
                    data_matches = False
                    mismatch_details.append(f"NaN mismatch on {old_date_str}, column {col}: {old_val} != {new_val}")
                    break
                
                # 수치 비교 - 상대적으로 관대한 조건 (0.01% 오차 허용)
                if not np.allclose([old_val], [new_val], rtol=1e-4, atol=1e-6, equal_nan=True):
                    # 추가 검증: 정수값이 소수점으로 변환된 경우 허용 (예: 100 vs 100.0)
                    try:
                        if abs(float(old_val) - float(new_val)) < 1e-6:
                            continue
                    except:
                        pass
                    
                    mismatch_details.append(f"Value mismatch on {old_date_str}, column {col}: {old_val} != {new_val}")
                    mismatched_dates += 1
                    # 🔧 관대한 조건: 즉시 중단하지 않고 허용 한도까지 계속 검사
                    if mismatched_dates > allowed_mismatches:
                        data_matches = False
                        break
            
            if not data_matches:
                break
            
            # 문자열 컬럼 비교 (Date 제외) - 완화된 조건
            str_cols = old_df.select_dtypes(include=['object']).columns
            str_cols = [col for col in str_cols if col != 'Date']
            for col in str_cols:
                old_str = str(old_row[col]).strip() if not pd.isna(old_row[col]) else ''
                new_str = str(new_row[col]).strip() if not pd.isna(new_row[col]) else ''
                
                if old_str != new_str:
                    mismatch_details.append(f"String mismatch on {old_date_str}, column {col}: '{old_str}' != '{new_str}'")
                    mismatched_dates += 1
                    # 🔧 관대한 조건: 허용 한도까지 계속 검사
                    if mismatched_dates > allowed_mismatches:
                        data_matches = False
                        break
            
            if not data_matches:
                break
        
        # 🔧 관대한 검증 결과 평가
        logger.info(f"  ✅ Checked {checked_dates} sample dates, {mismatched_dates} mismatches found (allowed: {allowed_mismatches})")
        if mismatch_details:
            logger.info(f"  ⚠️ Sample mismatches: {mismatch_details[:3]}...")
        
        if not data_matches:
            logger.info(f"❌ [EXTENSION_CHECK] Too many data mismatches ({mismatched_dates} > {allowed_mismatches})")
            return {
                'is_extension': False,
                'new_rows_count': 0,
                'old_start_date': old_start_date.strftime('%Y-%m-%d'),
                'old_end_date': old_end_date.strftime('%Y-%m-%d'),
                'new_start_date': new_start_date.strftime('%Y-%m-%d'),
                'new_end_date': new_end_date.strftime('%Y-%m-%d'),
                'validation_details': {
                    'reason': f'Too many data mismatches: {mismatched_dates} > {allowed_mismatches}',
                    'mismatches_found': mismatched_dates,
                    'allowed_mismatches': allowed_mismatches,
                    'sample_details': mismatch_details[:5]
                }
            }
        elif mismatched_dates > 0:
            logger.info(f"⚠️ [EXTENSION_CHECK] Minor mismatches found but within tolerance ({mismatched_dates} <= {allowed_mismatches})")
        
        # ✅ 검증 7: 새로 추가된 데이터 분석 (과거/미래 데이터 모두 허용)
        new_only_dates = new_dates - old_dates
        
        # 확장 유형 분석
        extension_type = []
        if new_start_date < old_start_date:
            past_dates = len([d for d in new_only_dates if pd.to_datetime(d) < old_start_date])
            extension_type.append(f"과거 데이터 {past_dates}개 추가")
        if new_end_date > old_end_date:
            future_dates = len([d for d in new_only_dates if pd.to_datetime(d) > old_end_date])
            extension_type.append(f"미래 데이터 {future_dates}개 추가")
        
        extension_desc = " + ".join(extension_type) if extension_type else "데이터 보완"
        
        # ✅ 모든 검증 통과: 데이터 확장으로 인정 (과거/미래 모두 허용)
        new_rows_count = len(new_only_dates)
        base_hash = get_data_content_hash(old_file_path)
        
        logger.info(f"✅ [EXTENSION_CHECK] Valid data extension: {extension_desc} (+{new_rows_count} new dates)")
        
        return {
            'is_extension': True,
            'new_rows_count': new_rows_count,
            'base_hash': base_hash,
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
        logger.error(f"Data extension check failed: {str(e)}")
        return {
            'is_extension': False, 
            'new_rows_count': 0,
            'old_start_date': None,
            'old_end_date': None,
            'new_start_date': None,
            'new_end_date': None,
            'validation_details': {'error': str(e)}
        }

def find_existing_cache_range(file_path):
    """
    기존 파일의 캐시에서 사용된 데이터 범위 정보를 찾는 함수
    
    Returns:
    --------
    dict or None: {'start_date': 'YYYY-MM-DD', 'cutoff_date': 'YYYY-MM-DD'} 또는 None
    """
    try:
        # 파일에 대응하는 캐시 디렉토리 찾기
        cache_dirs = get_file_cache_dirs(file_path)
        predictions_dir = Path(cache_dirs['predictions'])
        
        if not predictions_dir.exists():
            return None
            
        # 최근 메타 파일에서 데이터 범위 정보 확인
        meta_files = list(predictions_dir.glob("*_meta.json"))
        if not meta_files:
            return None
            
        # 가장 최근 메타 파일 선택
        latest_meta = max(meta_files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_meta, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
            
        # 데이터 범위 정보 추출
        model_config = meta_data.get('model_config', {})
        lstm_config = model_config.get('lstm', {})
        
        start_date = lstm_config.get('data_start_date')
        cutoff_date = lstm_config.get('data_cutoff_date') or meta_data.get('data_end_date')
        
        if start_date and cutoff_date:
            return {
                'start_date': start_date,
                'cutoff_date': cutoff_date,
                'meta_file': str(latest_meta)
            }
            
        return None
        
    except Exception as e:
        logger.warning(f"Failed to find cache range for {file_path}: {str(e)}")
        return None

def find_existing_cache_range(file_path):
    """
    기존 파일의 캐시에서 사용된 데이터 범위 정보를 찾는 함수
    
    Returns:
    --------
    dict or None: {'start_date': 'YYYY-MM-DD', 'cutoff_date': 'YYYY-MM-DD'} 또는 None
    """
    try:
        # 파일에 대응하는 캐시 디렉토리 찾기
        cache_dirs = get_file_cache_dirs(file_path)
        predictions_dir = Path(cache_dirs['predictions'])
        
        if not predictions_dir.exists():
            return None
            
        # 최근 메타 파일에서 데이터 범위 정보 확인
        meta_files = list(predictions_dir.glob("*_meta.json"))
        if not meta_files:
            return None
            
        # 가장 최근 메타 파일 선택
        latest_meta = max(meta_files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_meta, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
            
        # 데이터 범위 정보 추출
        model_config = meta_data.get('model_config', {})
        lstm_config = model_config.get('lstm', {})
        
        start_date = lstm_config.get('data_start_date')
        cutoff_date = lstm_config.get('data_cutoff_date') or meta_data.get('data_end_date')
        
        if start_date and cutoff_date:
            return {
                'start_date': start_date,
                'cutoff_date': cutoff_date,
                'meta_file': str(latest_meta)
            }
            
        return None
        
    except Exception as e:
        logger.warning(f"Failed to find cache range for {file_path}: {str(e)}")
        return None

def migrate_legacy_cache_if_needed(file_path):
    """
    이전 파일 해시로 생성된 캐시를 새로운 데이터 해시와 연결
    Excel 파일 읽기 실패/성공에 따른 해시 차이 문제 해결
    """
    try:
        current_hash = get_data_content_hash(file_path)
        if not current_hash:
            return False
            
        # 파일 기반 해시도 계산
        file_hash = calculate_file_hash(file_path)
        if not file_hash:
            return False
            
        file_hash_short = file_hash[:16]
        
        # 현재 해시와 파일 해시가 다른 경우에만 마이그레이션 시도
        if current_hash == file_hash_short:
            return False
            
        logger.info(f"🔄 [CACHE_MIGRATION] Checking for legacy cache migration:")
        logger.info(f"  📄 File: {os.path.basename(file_path)}")
        logger.info(f"  🔑 Data hash: {current_hash}")
        logger.info(f"  🔑 File hash: {file_hash_short}")
        
        # 기존 캐시 디렉토리들 확인
        cache_root = Path(CACHE_ROOT_DIR)
        if not cache_root.exists():
            return False
            
        file_name = Path(file_path).stem
        
        # 파일 해시로 된 캐시 디렉토리 찾기
        legacy_cache_dir = None
        for cache_dir in cache_root.iterdir():
            if cache_dir.is_dir() and cache_dir.name.startswith(file_hash_short):
                if file_name in cache_dir.name:
                    legacy_cache_dir = cache_dir
                    break
                    
        if not legacy_cache_dir or not legacy_cache_dir.exists():
            logger.info(f"📋 [CACHE_MIGRATION] No legacy cache found")
            return False
            
        # 새로운 데이터 해시로 된 캐시 디렉토리명
        new_cache_dir_name = f"{current_hash}_{file_name}"
        new_cache_dir = cache_root / new_cache_dir_name
        
        if new_cache_dir.exists():
            logger.info(f"📋 [CACHE_MIGRATION] New cache already exists, no migration needed")
            return False
            
        # 캐시 디렉토리 이름 변경 (마이그레이션)
        try:
            legacy_cache_dir.rename(new_cache_dir)
            logger.info(f"✅ [CACHE_MIGRATION] Successfully migrated cache:")
            logger.info(f"  📁 From: {legacy_cache_dir.name}")
            logger.info(f"  📁 To: {new_cache_dir.name}")
            
            # 메타데이터 파일들의 해시 정보 업데이트
            predictions_dir = new_cache_dir / 'predictions'
            if predictions_dir.exists():
                for meta_file in predictions_dir.glob("*_meta.json"):
                    try:
                        with open(meta_file, 'r', encoding='utf-8') as f:
                            meta_data = json.load(f)
                        
                        # 해시 정보 업데이트
                        meta_data['file_content_hash'] = current_hash
                        meta_data['migration_info'] = {
                            'original_hash': file_hash_short,
                            'migrated_to': current_hash,
                            'migration_date': datetime.now().isoformat()
                        }
                        
                        with open(meta_file, 'w', encoding='utf-8') as f:
                            json.dump(meta_data, f, indent=2, ensure_ascii=False)
                            
                        logger.info(f"📝 [CACHE_MIGRATION] Updated meta file: {meta_file.name}")
                    except Exception as e:
                        logger.warning(f"⚠️ [CACHE_MIGRATION] Failed to update meta file {meta_file.name}: {e}")
            
            return True
            
        except Exception as rename_error:
            logger.error(f"❌ [CACHE_MIGRATION] Failed to rename cache directory: {rename_error}")
            return False
            
    except Exception as e:
        logger.error(f"❌ [CACHE_MIGRATION] Migration check failed: {e}")
        return False

def load_existing_predictions_for_extension(file_path, target_date, compatibility_info):
    """
    확장된 데이터의 기존 캐시에서 사용 가능한 예측들을 로드
    
    Args:
        file_path (str): 현재 파일 경로
        target_date (pd.Timestamp): 목표 날짜
        compatibility_info (dict): 호환성 정보
    
    Returns:
        list: 기존 예측 결과들
    """
    try:
        if not compatibility_info.get('found'):
            return []
            
        # 개선된 호환성 정보 활용
        cache_files = compatibility_info.get('cache_files', [])
        predictions_dir_path = compatibility_info.get('compatibility_info', {}).get('predictions_dir', '')
        
        logger.info(f"🔄 [EXTENSION_CACHE] Using compatibility info:")
        logger.info(f"    Cache files: {len(cache_files)}")
        logger.info(f"    Predictions dir: {predictions_dir_path}")
        
        if not cache_files and not predictions_dir_path:
            logger.warning(f"❌ [EXTENSION_CACHE] No cache files or predictions directory found")
            return []
            
        # 직접 predictions 디렉토리가 주어진 경우 바로 사용
        if predictions_dir_path and os.path.exists(predictions_dir_path):
            predictions_dir = Path(predictions_dir_path)
            logger.info(f"✅ [EXTENSION_CACHE] Using direct predictions directory: {predictions_dir}")
        else:
            # 기존 방식으로 폴백
            original_cache_file = cache_files[0] if cache_files else file_path
            logger.info(f"🔄 [EXTENSION_CACHE] Loading existing predictions from: {os.path.basename(original_cache_file)}")
            
            # 원본 파일의 캐시 디렉토리 찾기
            original_hash = get_data_content_hash(original_cache_file)
            if not original_hash:
                logger.warning(f"⚠️ [EXTENSION_CACHE] Cannot get hash for original file")
                return []
                
            cache_root = Path(CACHE_ROOT_DIR)
            original_file_name = Path(original_cache_file).stem
        
        # 🔧 강화된 캐시 디렉토리 찾기 로직
        logger.info(f"🔍 [EXTENSION_CACHE] Searching for cache directory with hash: {original_hash[:12]}")
        logger.info(f"🔍 [EXTENSION_CACHE] Original file name: {original_file_name}")
        
        original_cache_dir = None
        possible_dirs = []
        
        # 모든 캐시 디렉토리를 확인
        for cache_dir in cache_root.iterdir():
            if cache_dir.is_dir():
                logger.info(f"🔍 [EXTENSION_CACHE] Checking directory: {cache_dir.name}")
                
                # 해시 매칭 체크
                hash_match = cache_dir.name.startswith(original_hash[:12])
                # 파일명 매칭 체크 (더 유연하게)
                name_match = (original_file_name in cache_dir.name or 
                             any(part in cache_dir.name for part in original_file_name.split('_')))
                
                logger.info(f"    Hash match: {hash_match}, Name match: {name_match}")
                
                if hash_match:
                    possible_dirs.append(cache_dir)
                    if name_match:
                        original_cache_dir = cache_dir
                        logger.info(f"✅ [EXTENSION_CACHE] Found perfect match: {cache_dir.name}")
                        break
        
        # 정확한 매치가 없으면 해시만 매칭되는 것 중 첫 번째 사용
        if not original_cache_dir and possible_dirs:
            original_cache_dir = possible_dirs[0]
            logger.info(f"⚠️ [EXTENSION_CACHE] Using hash-only match: {original_cache_dir.name}")
        
        if not original_cache_dir:
            logger.warning(f"❌ [EXTENSION_CACHE] No cache directory found for hash {original_hash[:12]}")
            logger.warning(f"❌ [EXTENSION_CACHE] Available directories: {[d.name for d in cache_root.iterdir() if d.is_dir()]}")
            return []
        
        logger.info(f"✅ [EXTENSION_CACHE] Using cache directory: {original_cache_dir.name}")
            
        predictions_dir = original_cache_dir / 'predictions'
        if not predictions_dir.exists():
            logger.warning(f"❌ [EXTENSION_CACHE] Predictions directory not found: {predictions_dir}")
            return []
            
        # 🔧 강화된 예측 파일 로드 로직
        logger.info(f"📂 [EXTENSION_CACHE] Predictions directory: {predictions_dir}")
        
        # 모든 파일 목록 확인
        all_files = list(predictions_dir.iterdir())
        logger.info(f"📊 [EXTENSION_CACHE] Total files in predictions directory: {len(all_files)}")
        
        # CSV 파일들을 찾아서 로드
        csv_files = list(predictions_dir.glob("prediction_start_*.cs"))
        json_files = list(predictions_dir.glob("prediction_start_*.json"))
        
        logger.info(f"📊 [EXTENSION_CACHE] Found {len(csv_files)} CSV files and {len(json_files)} JSON files")
        logger.info(f"📊 [EXTENSION_CACHE] CSV files: {[f.name for f in csv_files]}")
        
        existing_predictions = []
        
        # CSV 파일들 우선 처리
        for csv_file in csv_files:
            try:
                logger.info(f"📄 [EXTENSION_CACHE] Loading CSV file: {csv_file.name}")
                
                # CSV 로드
                df = pd.read_csv(csv_file)
                logger.info(f"📊 [EXTENSION_CACHE] CSV shape: {df.shape}, columns: {list(df.columns)}")
                
                # 컬럼명 정규화
                if 'date' in df.columns:
                    df['Date'] = pd.to_datetime(df['date'])
                elif 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                else:
                    logger.warning(f"⚠️ [EXTENSION_CACHE] No Date column found in {csv_file.name}")
                    continue
                    
                if 'prediction' in df.columns:
                    df['Prediction'] = df['prediction']
                elif 'Prediction' not in df.columns:
                    logger.warning(f"⚠️ [EXTENSION_CACHE] No Prediction column found in {csv_file.name}")
                    continue
                    
                # 🔴 날짜 필터링 로직을 제거하고 모든 예측을 로드합니다.
                if 'Date' in df.columns:
                    # valid_predictions = df[df['Date'] <= target_date] # 이 줄을 삭제하거나 주석 처리합니다.
                    logger.info(f"📊 [EXTENSION_CACHE] Loading all {len(df)} predictions from cached file (filter removed).")
                    
                    for _, row in df.iterrows(): # df를 직접 순회하도록 변경합니다.
                        try:
                            date_str = pd.to_datetime(row['Date']).strftime('%Y-%m-%d')
                        except:
                            date_str = str(row['Date'])
                        pred_value = float(row.get('Prediction') or row.get('prediction') or 0)
                        actual_value = row.get('Actual', row.get('actual', None))
                        
                        existing_predictions.append({
                            'Date': date_str,
                            'Prediction': pred_value,
                            'Actual': actual_value
                        })
                        
            except Exception as e:
                logger.warning(f"❌ [EXTENSION_CACHE] Error loading {csv_file.name}: {e}")
                continue
        
        # CSV 파일이 없으면 JSON 파일도 시도
        if not existing_predictions and json_files:
            logger.info(f"📄 [EXTENSION_CACHE] No data from CSV files, trying JSON files...")
            
            for json_file in json_files:
                try:
                    logger.info(f"📄 [EXTENSION_CACHE] Loading JSON file: {json_file.name}")
                    
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # JSON 구조에서 예측 데이터 추출
                    predictions_data = data.get('predictions', [])
                    if not predictions_data:
                        predictions_data = data.get('predictions_flat', [])
                    
                    logger.info(f"📊 [EXTENSION_CACHE] JSON predictions count: {len(predictions_data)}")
                    
                    for pred in predictions_data:
                        pred_date = pd.to_datetime(pred.get('date') or pred.get('Date'))
                        if pred_date <= target_date:
                            existing_predictions.append({
                                'Date': pred_date.strftime('%Y-%m-%d'),
                                'Prediction': float(pred.get('prediction', 0)),
                                'Actual': pred.get('actual', None)
                            })
                            
                except Exception as e:
                    logger.warning(f"❌ [EXTENSION_CACHE] Error loading JSON {json_file.name}: {e}")
                    continue
                
        # 날짜순 정렬
        existing_predictions.sort(key=lambda x: x['Date'])
        
        logger.info(f"✅ [EXTENSION_CACHE] Loaded {len(existing_predictions)} existing predictions")
        return existing_predictions
        
    except Exception as e:
        logger.error(f"❌ [EXTENSION_CACHE] Failed to load existing predictions: {e}")
        return []

def find_compatible_predictions(current_file_path, prediction_start_date):
    """
    현재 파일이 기존 파일의 확장인 경우, 기존 파일의 호환 가능한 예측 결과를 찾는 함수
    
    Parameters:
    -----------
    current_file_path : str
        현재 파일 경로
    prediction_start_date : str or pd.Timestamp
        예측 시작 날짜
        
    Returns:
    --------
    dict or None: {
        'predictions': list,
        'metadata': dict,
        'attention_data': dict,
        'ma_results': dict,
        'source_file': str,
        'extension_info': dict
    } 또는 None (호환 가능한 예측 결과가 없을 경우)
    """
    try:
        # uploads 폴더의 다른 파일들을 확인
        upload_dir = Path(UPLOAD_FOLDER)
        existing_files = [f for f in upload_dir.glob('*.xlsx') if str(f) != current_file_path]
        logger.info(f"🔍 [PREDICTIONS_SEARCH] 탐색할 기존 파일 수: {len(existing_files)}")
        
        for existing_file in existing_files:
            try:
                # 확장 관계 확인 + 단순 파일명 유사성 확인
                extension_result = check_data_extension(str(existing_file), current_file_path)
                is_extension = extension_result.get('is_extension', False)
                
                # 확장 관계가 인식되지 않는 경우 파일명 유사성으로 대체 확인
                if not is_extension:
                    existing_name = existing_file.stem.lower()
                    current_name = Path(current_file_path).stem.lower()
                    # 기본 이름이 같거나 하나가 다른 하나를 포함하는 경우
                    if (existing_name in current_name or current_name in existing_name or 
                        existing_name.replace('_', '') == current_name.replace('_', '')):
                        is_extension = True
                        logger.info(f"🔍 [PREDICTIONS_SEARCH] 파일명 유사성으로 확장 관계 인정: {existing_file.name} -> {Path(current_file_path).name}")
                
                if is_extension:
                    if extension_result.get('is_extension', False):
                        logger.info(f"🔍 [PREDICTIONS_SEARCH] 확장 관계 발견: {existing_file.name} -> {Path(current_file_path).name}")
                        logger.info(f"    📈 Extension type: {extension_result.get('validation_details', {}).get('extension_type', 'Unknown')}")
                        logger.info(f"    ➕ New rows: {extension_result.get('new_rows_count', 0)}")
                    else:
                        logger.info(f"🔍 [PREDICTIONS_SEARCH] 파일명 유사성 기반 호환성 인정: {existing_file.name} -> {Path(current_file_path).name}")
                    
                    # 기존 파일의 예측 결과 캐시 확인
                    existing_cache_dirs = get_file_cache_dirs(str(existing_file))
                    existing_predictions_dir = existing_cache_dirs['predictions']
                    
                    if os.path.exists(existing_predictions_dir):
                        # 해당 날짜의 예측 결과 파일 찾기
                        if isinstance(prediction_start_date, str):
                            start_date = pd.to_datetime(prediction_start_date)
                        else:
                            start_date = prediction_start_date
                        
                        date_str = start_date.strftime('%Y%m%d')
                        
                        # 파일명 패턴 시도 (보안을 위해 .cs 확장자 우선 사용)
                        csv_patterns = [
                            f"prediction_start_{date_str}.cs",
                            f"prediction_{date_str}.cs"
                        ]
                        meta_patterns = [
                            f"prediction_start_{date_str}_meta.json",
                            f"prediction_{date_str}_meta.json"
                        ]
                        
                        csv_filepath = None
                        meta_filepath = None
                        
                        for csv_pattern, meta_pattern in zip(csv_patterns, meta_patterns):
                            csv_test = existing_predictions_dir / csv_pattern
                            meta_test = existing_predictions_dir / meta_pattern
                            
                            if csv_test.exists() and meta_test.exists():
                                csv_filepath = csv_test
                                meta_filepath = meta_test
                                break
                        
                        if csv_filepath and meta_filepath:
                            try:
                                # CSV 로드 - 안전한 fallback 사용
                                from app.data.loader import load_csv_safe_with_fallback
                                predictions_df = load_csv_safe_with_fallback(csv_filepath)
                                
                                # 컬럼명 호환성 처리
                                if 'date' in predictions_df.columns:
                                    predictions_df['Date'] = pd.to_datetime(predictions_df['date'])
                                elif 'Date' in predictions_df.columns:
                                    predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
                                
                                if 'prediction' in predictions_df.columns:
                                    predictions_df['Prediction'] = predictions_df['prediction']
                                
                                if 'prediction_from' in predictions_df.columns:
                                    predictions_df['Prediction_From'] = pd.to_datetime(predictions_df['prediction_from'])
                                elif 'Prediction_From' in predictions_df.columns:
                                    predictions_df['Prediction_From'] = pd.to_datetime(predictions_df['Prediction_From'])
                                
                                predictions = predictions_df.to_dict('records')
                                
                                # 메타데이터 로드
                                with open(meta_filepath, 'r', encoding='utf-8') as f:
                                    metadata = json.load(f)
                                
                                # Attention 데이터 로드
                                attention_data = {}
                                attention_patterns = [
                                    f"prediction_start_{date_str}_attention.json",
                                    f"prediction_{date_str}_attention.json"
                                ]
                                
                                for attention_pattern in attention_patterns:
                                    attention_filepath = existing_predictions_dir / attention_pattern
                                    if attention_filepath.exists():
                                        try:
                                            with open(attention_filepath, 'r', encoding='utf-8') as f:
                                                attention_raw = json.load(f)
                                            attention_data = {
                                                'image': attention_raw.get('image_base64', ''),
                                                'feature_importance': attention_raw.get('feature_importance', {}),
                                                'temporal_importance': attention_raw.get('temporal_importance', {})
                                            }
                                            break
                                        except Exception as e:
                                            logger.warning(f"⚠️ Failed to load attention data: {str(e)}")
                                
                                # 이동평균 데이터 로드
                                ma_results = {}
                                ma_patterns = [
                                    f"prediction_start_{date_str}_ma.json",
                                    f"prediction_{date_str}_ma.json"
                                ]
                                
                                for ma_pattern in ma_patterns:
                                    ma_filepath = existing_predictions_dir / ma_pattern
                                    if ma_filepath.exists():
                                        try:
                                            with open(ma_filepath, 'r', encoding='utf-8') as f:
                                                ma_results = json.load(f)
                                            break
                                        except Exception as e:
                                            logger.warning(f"⚠️ Failed to load MA results: {str(e)}")
                                
                                logger.info(f"✅ [PREDICTIONS_SEARCH] 기존 파일에서 호환 예측 결과 발견!")
                                logger.info(f"    📁 Source file: {existing_file.name}")
                                logger.info(f"    📊 Predictions: {len(predictions)}")
                                logger.info(f"    🧠 Attention data: {'Yes' if attention_data else 'No'}")
                                logger.info(f"    📈 MA results: {'Yes' if ma_results else 'No'}")
                                
                                return {
                                    'predictions': predictions,
                                    'metadata': metadata,
                                    'attention_data': attention_data,
                                    'ma_results': ma_results,
                                    'source_file': str(existing_file),
                                    'extension_info': extension_result
                                }
                                
                            except Exception as e:
                                logger.warning(f"기존 예측 결과 파일 로드 실패 ({existing_file.name}): {str(e)}")
                        else:
                            logger.info(f"🔍 [PREDICTIONS_SEARCH] {start_date.strftime('%Y-%m-%d')} 날짜의 예측 결과가 없습니다.")
                    
            except Exception as e:
                logger.warning(f"파일 확장 관계 확인 실패 ({existing_file.name}): {str(e)}")
                continue
        
        logger.info(f"❌ [PREDICTIONS_SEARCH] 호환 가능한 예측 결과를 찾지 못했습니다.")
        return None
        
    except Exception as e:
        logger.error(f"호환 가능한 예측 결과 찾기 실패: {str(e)}")
        return None

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

def find_compatible_cache_file(new_file_path, intended_data_range=None, cached_df=None):
    """
    새 파일과 호환되는 기존 캐시를 찾는 함수 (데이터 범위 고려)
    
    🔧 핵심 개선:
    - 파일 내용 + 사용 데이터 범위를 모두 고려
    - 같은 파일이라도 다른 데이터 범위면 새 예측으로 인식
    - 사용자 의도를 반영한 스마트 캐시 매칭
    - 중복 로딩 방지를 위한 캐시된 DataFrame 재사용
    
    Parameters:
    -----------
    new_file_path : str
        새 파일 경로
    intended_data_range : dict, optional
        사용자가 의도한 데이터 범위 {'start_date': 'YYYY-MM-DD', 'cutoff_date': 'YYYY-MM-DD'}
    cached_df : DataFrame, optional
        이미 로딩된 DataFrame (중복 로딩 방지)
    
    Returns:
    --------
    dict: {
        'found': bool,
        'cache_type': str,  # 'exact_with_range', 'extension', 'partial', 'range_mismatch'
        'cache_files': list,
        'compatibility_info': dict
    }
    """
    try:
        from app.data.loader import load_data
        # 🔧 캐시된 DataFrame이 있으면 재사용, 없으면 새로 로딩
        if cached_df is not None:
            logger.info(f"🔄 [CACHE_OPTIMIZATION] Using cached DataFrame (avoiding duplicate load)")
            new_df = cached_df.copy()
        else:
            logger.info(f"📁 [CACHE_COMPATIBILITY] Loading data for cache check...")
            # 새 파일의 데이터 분석 (파일 형식에 맞게)
            file_ext = os.path.splitext(new_file_path.lower())[1]
            if file_ext == '.csv':
                new_df = pd.read_csv(new_file_path)
            else:
                # Excel 파일인 경우 load_data 함수 사용
                from app.data.loader import load_data
                new_df = load_data(new_file_path)
                # 인덱스가 Date인 경우 컬럼으로 복원
                if new_df.index.name == 'Date':
                    new_df = new_df.reset_index()
        
        if 'Date' not in new_df.columns:
            return {'found': False, 'cache_type': None, 'reason': 'No Date column'}
            
        new_df['Date'] = pd.to_datetime(new_df['Date'])
        new_start_date = new_df['Date'].min()
        new_end_date = new_df['Date'].max()
        new_hash = get_data_content_hash(new_file_path)
        
        logger.info(f"🔍 [ENHANCED_CACHE] Analyzing new file:")
        logger.info(f"  📅 Full date range: {new_start_date.strftime('%Y-%m-%d')} ~ {new_end_date.strftime('%Y-%m-%d')}")
        logger.info(f"  📊 Records: {len(new_df)}")
        logger.info(f"  🔑 Hash: {new_hash[:12] if new_hash else 'None'}...")
        
        # 사용자 의도 데이터 범위 확인
        if intended_data_range:
            intended_start = pd.to_datetime(intended_data_range.get('start_date', new_start_date))
            intended_cutoff = pd.to_datetime(intended_data_range.get('cutoff_date', new_end_date))
            logger.info(f"  🎯 Intended range: {intended_start.strftime('%Y-%m-%d')} ~ {intended_cutoff.strftime('%Y-%m-%d')}")
        else:
            intended_start = new_start_date
            intended_cutoff = new_end_date
            logger.info(f"  🎯 Using full range (no specific intention provided)")
        
        compatible_caches = []
        
        # 1. uploads 폴더의 파일들 검사 (데이터 범위 고려)
        upload_dir = Path(UPLOAD_FOLDER)
        existing_files = list(upload_dir.glob('*.csv')) + list(upload_dir.glob('*.xlsx')) + list(upload_dir.glob('*.xls'))
        
        logger.info(f"🔍 [ENHANCED_CACHE] Checking {len(existing_files)} upload files with range consideration...")
        
        for existing_file in existing_files:
            if existing_file.name == os.path.basename(new_file_path):
                continue
                
            try:
                # 파일 해시 확인
                existing_hash = get_data_content_hash(str(existing_file))
                if existing_hash == new_hash:
                    logger.info(f"📄 [ENHANCED_CACHE] Same file content found: {existing_file.name}")
                    
                    # 🔑 같은 파일이지만 데이터 범위 의도 확인
                    # 기존 캐시의 데이터 범위 정보를 찾아야 함
                    existing_cache_range = find_existing_cache_range(str(existing_file))
                    
                    if existing_cache_range and intended_data_range:
                        cache_start = existing_cache_range.get('start_date')
                        cache_cutoff = existing_cache_range.get('cutoff_date') 
                        
                        if cache_start and cache_cutoff:
                            cache_start = pd.to_datetime(cache_start)
                            cache_cutoff = pd.to_datetime(cache_cutoff)
                            
                            # 데이터 범위 비교
                            range_match = (
                                abs((intended_start - cache_start).days) <= 30 and 
                                abs((intended_cutoff - cache_cutoff).days) <= 30
                            )
                            
                            if range_match:
                                logger.info(f"✅ [ENHANCED_CACHE] Exact match with same intended range!")
                                return {
                                    'found': True,
                                    'cache_type': 'exact_with_range',
                                    'cache_files': [str(existing_file)],
                                    'compatibility_info': {
                                        'match_type': 'file_hash_and_range',
                                        'cache_range': existing_cache_range,
                                        'intended_range': intended_data_range
                                    }
                                }
                            else:
                                logger.info(f"⚠️ [ENHANCED_CACHE] Same file but different intended range:")
                                logger.info(f"    💾 Cached range: {cache_start.strftime('%Y-%m-%d')} ~ {cache_cutoff.strftime('%Y-%m-%d')}")
                                logger.info(f"    🎯 Intended range: {intended_start.strftime('%Y-%m-%d')} ~ {intended_cutoff.strftime('%Y-%m-%d')}")
                                logger.info(f"    🔄 Will create new cache for different range")
                                # 같은 파일이지만 다른 범위 의도 → 새 예측 필요
                                continue
                    
                    # 범위 정보가 없으면 기존 로직 적용
                    logger.info(f"✅ [ENHANCED_CACHE] Exact file match (no range info): {existing_file.name}")
                    return {
                        'found': True,
                        'cache_type': 'exact',
                        'cache_files': [str(existing_file)],
                        'compatibility_info': {'match_type': 'file_hash_only'}
                    }
                
                # 확장 파일 확인 (기존 로직 유지) - 디버깅 강화
                logger.info(f"🔍 [EXTENSION_CHECK] Testing extension: {existing_file.name} → {os.path.basename(new_file_path)}")
                extension_info = check_data_extension(str(existing_file), new_file_path)
                
                logger.info(f"📊 [EXTENSION_RESULT] is_extension: {extension_info['is_extension']}")
                if extension_info.get('validation_details'):
                    logger.info(f"📊 [EXTENSION_RESULT] reason: {extension_info['validation_details'].get('reason', 'N/A')}")
                
                if extension_info['is_extension']:
                    logger.info(f"📈 [ENHANCED_CACHE] Found extension base: {existing_file.name} (+{extension_info.get('new_rows_count', 0)} rows)")
                    return {
                        'found': True,
                        'cache_type': 'extension', 
                        'cache_files': [str(existing_file)],
                        'compatibility_info': extension_info
                    }
                else:
                    logger.info(f"❌ [EXTENSION_CHECK] Not an extension: {extension_info['validation_details'].get('reason', 'Unknown reason')}")
                    
            except Exception as e:
                logger.warning(f"Error checking upload file {existing_file}: {str(e)}")
                continue
        
        # 2. 🔧 강화된 캐시 디렉토리 검사
        cache_root = Path(CACHE_ROOT_DIR)
        if not cache_root.exists():
            logger.info("❌ [ENHANCED_CACHE] No cache directory found")
            return {'found': False, 'cache_type': None}
            
        logger.info(f"🔍 [ENHANCED_CACHE] Scanning cache directories at: {cache_root}")
        
        all_cache_dirs = list(cache_root.iterdir())
        valid_cache_dirs = [d for d in all_cache_dirs if d.is_dir()]
        
        logger.info(f"📊 [ENHANCED_CACHE] Found {len(valid_cache_dirs)} cache directories")
        
        for file_cache_dir in valid_cache_dirs:
            logger.info(f"🔍 [ENHANCED_CACHE] Checking directory: {file_cache_dir.name}")
            
            predictions_dir = file_cache_dir / 'predictions'
            if not predictions_dir.exists():
                logger.info(f"    ❌ No predictions directory found")
                continue
                
            # predictions 디렉토리 내 파일 확인
            pred_files = list(predictions_dir.iterdir())
            csv_files = [f for f in pred_files if f.suffix == '.csv']
            json_files = [f for f in pred_files if f.suffix == '.json']
            
            logger.info(f"    📊 Found {len(csv_files)} CSV and {len(json_files)} JSON files")
            
            # 간단한 확장 감지: 파일 해시 기반
            if any(new_hash and new_hash[:12] in file_cache_dir.name for _ in [1] if new_hash):
                logger.info(f"    ✅ Hash match detected: {file_cache_dir.name}")
                
                # 기존 파일 경로 추정
                cache_files = [str(f) for f in pred_files if f.suffix in ['.csv', '.xlsx', '.xls']]
                if cache_files:
                    compatible_caches.append({
                        'cache_dir': str(file_cache_dir),
                        'predictions_dir': str(predictions_dir),
                        'cache_files': cache_files,
                        'match_type': 'hash_based'
                    })
                    logger.info(f"    📁 Added to compatible caches")
                
                # predictions_index.cs 파일에서 캐시된 예측들의 날짜 범위 확인 (보안을 위해 .cs 확장자 사용)
                index_file = predictions_dir / 'predictions_index.cs'
                if not index_file.exists():
                    continue
                
            try:
                cache_index = pd.read_csv(index_file)
                if 'data_end_date' not in cache_index.columns:
                    continue
                    
                cache_index['data_end_date'] = pd.to_datetime(cache_index['data_end_date'])
                cache_start = cache_index['data_end_date'].min()
                cache_end = cache_index['data_end_date'].max()
                
                logger.info(f"  📁 {file_cache_dir.name}: {cache_start.strftime('%Y-%m-%d')} ~ {cache_end.strftime('%Y-%m-%d')} ({len(cache_index)} predictions)")
                
                # 날짜 범위 중복 확인
                overlap_start = max(new_start_date, cache_start)
                overlap_end = min(new_end_date, cache_end)
                
                if overlap_start <= overlap_end:
                    overlap_days = (overlap_end - overlap_start).days + 1
                    new_total_days = (new_end_date - new_start_date).days + 1
                    coverage_ratio = overlap_days / new_total_days
                    
                    logger.info(f"    📊 Overlap: {overlap_days} days ({coverage_ratio:.1%} coverage)")
                    
                    if coverage_ratio >= 0.7:  # 70% 이상 겹치면 호환 가능
                        compatible_caches.append({
                            'cache_dir': str(file_cache_dir),
                            'predictions_dir': str(predictions_dir),
                            'coverage_ratio': coverage_ratio,
                            'overlap_days': overlap_days,
                            'cache_range': (cache_start, cache_end),
                            'prediction_count': len(cache_index)
                        })
                        
            except Exception as e:
                logger.warning(f"Error analyzing cache {file_cache_dir.name}: {str(e)}")
                continue
        
        # 3. 호환 가능한 캐시 결과 처리
        if compatible_caches:
            logger.info(f"🎯 [ENHANCED_CACHE] Found {len(compatible_caches)} compatible cache(s)")
            
            # 해시 기반 매칭이 있으면 우선 사용
            hash_based_caches = [c for c in compatible_caches if c.get('match_type') == 'hash_based']
            
            if hash_based_caches:
                best_cache = hash_based_caches[0]
                cache_type = 'hash_based'
                logger.info(f"  🥇 Using hash-based match: {Path(best_cache['cache_dir']).name}")
            else:
                # 커버리지 비율로 정렬 (높은 순)
                compatible_caches.sort(key=lambda x: x.get('coverage_ratio', 0), reverse=True)
                best_cache = compatible_caches[0]
                
                if best_cache.get('coverage_ratio', 0) >= 0.95:  # 95% 이상이면 거의 완전
                    cache_type = 'near_complete'
                elif len(compatible_caches) > 1:  # 여러 캐시 조합 가능
                    cache_type = 'multi_cache' 
                else:
                    cache_type = 'partial'
                    
                logger.info(f"  🥇 Best: {Path(best_cache['cache_dir']).name} ({best_cache.get('coverage_ratio', 0):.1%} coverage)")
                
            return {
                'found': True,
                'cache_type': cache_type,
                'cache_files': [best_cache['predictions_dir']] if 'predictions_dir' in best_cache else best_cache.get('cache_files', []),
                'compatibility_info': {
                    'best_cache_dir': best_cache['cache_dir'],
                    'predictions_dir': best_cache.get('predictions_dir', ''),
                    'total_compatible_caches': len(compatible_caches),
                    'match_type': best_cache.get('match_type', 'coverage_based')
                }
            }
        
        logger.info("❌ [ENHANCED_CACHE] No compatible cache found")
        return {'found': False, 'cache_type': None}
        
    except Exception as e:
        logger.error(f"Enhanced cache compatibility check failed: {str(e)}")
        return {'found': False, 'cache_type': None, 'error': str(e)}
    
def save_prediction_simple(prediction_results: dict, prediction_date):
    """완전 통합 저장소에 저장하는 최종 버전 🌟"""
    try:
        # 통합 예측 디렉토리 확인
        ensure_unified_predictions_dir()
        # 통합 저장소 디렉토리들 확인
        storage_dirs = get_unified_storage_dirs()
        
        preds_root = prediction_results.get("predictions")

        # ── 첫 예측 레코드 추출 ─────────────────────────
        if isinstance(preds_root, dict) and preds_root:
            preds_seq = preds_root.get("future") or []
        else:                                   # list 혹은 None
            preds_seq = preds_root or prediction_results.get("predictions_flat", [])

        if not preds_seq:
            raise ValueError("prediction_results 안에 예측 데이터가 비어 있습니다.")

        first_rec = preds_seq[0]
        first_date = pd.to_datetime(first_rec.get("date") or first_rec.get("Date"))
        if pd.isna(first_date):
            raise ValueError("첫 예측 레코드에 날짜 정보가 없습니다.")

        # 🌟 완전 통합 저장소 사용
        predictions_dir = storage_dirs['predictions']
        hyperparameters_dir = storage_dirs['hyperparameters']
        plots_dir = storage_dirs['plots']
        
        # 🌟 통합 저장소 파일 경로 설정 (보안을 위해 .cs 확장자 사용)
        base_name = f"prediction_start_{first_date:%Y%m%d}"
        csv_path = predictions_dir / f"{base_name}.cs"
        meta_path = predictions_dir / f"{base_name}_meta.json"
        attention_path = predictions_dir / f"{base_name}_attention.json"
        ma_path = predictions_dir / f"{base_name}_ma.json"
        
        # 🔧 중복 확인 (기존 파일이 있으면 타임스탬프 추가)
        if csv_path.exists() and meta_path.exists():
            logger.info(f"🔄 [UNIFIED_SAVE] Existing prediction found, preserving with timestamp")
            timestamp = datetime.now().strftime('%H%M%S')
            csv_path = predictions_dir / f"{base_name}_{timestamp}.cs"
            meta_path = predictions_dir / f"{base_name}_{timestamp}_meta.json"
            attention_path = predictions_dir / f"{base_name}_{timestamp}_attention.json"
            ma_path = predictions_dir / f"{base_name}_{timestamp}_ma.json"
            logger.info(f"  📁 Files will be saved with timestamp: {timestamp}")
        
        logger.info(f"🌟 [UNIFIED_SAVE] Saving to unified storage:")
        logger.info(f"  📄 Predictions: {csv_path.name}")
        logger.info(f"  📄 Meta: {meta_path.name}")
        logger.info(f"  📁 Directory: {predictions_dir}")
        
        # 하이퍼파라미터/모델 저장 경로도 설정 (필요한 경우)
        hyperparameter_path = hyperparameters_dir / f"hyperparameter_{first_date:%Y%m%d}.json"

        # ── validation 개수 계산 ──────────────────────
        if isinstance(preds_root, dict):
            validation_cnt = len(preds_root.get("validation", []))
        else:
            validation_cnt = 0

        # ── 메타 + 본문 구성 (파일 캐시 정보 포함) ──────────────────────────
        current_file_path = prediction_state.get('current_file', None)
        file_content_hash = get_data_content_hash(current_file_path) if current_file_path else None
        
        meta = {
            "prediction_start_date": first_date.strftime("%Y-%m-%d"),
            "data_end_date": str(prediction_date)[:10],
            "created_at": datetime.now().isoformat(),
            "semimonthly_period": prediction_results.get("semimonthly_period"),
            "next_semimonthly_period": prediction_results.get("next_semimonthly_period"),
            "selected_features": prediction_results.get("selected_features", []),
            "total_predictions": len(prediction_results.get("predictions_flat", preds_seq)),
            "validation_points": validation_cnt,
            "is_pure_future_prediction": prediction_results.get("summary", {}).get(
                "is_pure_future_prediction", validation_cnt == 0
            ),
            "metrics": prediction_results.get("metrics"),
            "interval_scores": prediction_results.get("interval_scores", {}),
            # 🔑 원본 파일 정보 (참조용)
            "source_file_path": current_file_path,
            "source_file_hash": file_content_hash,
            "model_type": prediction_results.get("model_type", "ImprovedLSTMPredictor"),
            "loss_function": prediction_results.get("loss_function", "DirectionalLoss"),
            "prediction_mode": "일반 모드",
            # 🌟 통합 저장소 정보
            "storage_system": "unified_complete",
            "storage_paths": {
                "predictions": str(csv_path),
                "metadata": str(meta_path),
                "attention": str(attention_path),
                "ma_results": str(ma_path),
                "hyperparameters": str(hyperparameter_path)
            }
        }

        # ✅ CSV 파일 저장 - NaN 값 안전 처리 (통합 저장)
        predictions_data = clean_predictions_data(
            prediction_results.get("predictions_flat", preds_seq)
        )
        
        if predictions_data:
            # 🔧 NaN 값 추가 정리
            for pred in predictions_data:
                for key, value in list(pred.items()):
                    pred[key] = safe_serialize_value(value)
            
            pred_df = pd.DataFrame(predictions_data)
            
            # 🌟 통합 저장소에 저장
            pred_df.to_csv(csv_path, index=False)
            logger.info(f"✅ [UNIFIED] CSV saved: {csv_path}")

        # ✅ 메타데이터 저장 - NaN 값 안전 처리 (통합 저장)
        safe_meta = {}
        for key, value in meta.items():
            if isinstance(value, dict):
                safe_meta[key] = {}
                for k, v in value.items():
                    if isinstance(v, dict):
                        safe_meta[key][k] = {}
                        for kk, vv in v.items():
                            safe_meta[key][k][kk] = safe_serialize_value(vv)
                    else:
                        safe_meta[key][k] = safe_serialize_value(v)
            else:
                safe_meta[key] = safe_serialize_value(value)
        
        # 🌟 통합 저장소에 저장
        with open(meta_path, "w", encoding="utf-8") as fp:
            json.dump(safe_meta, fp, ensure_ascii=False, indent=2)
        logger.info(f"✅ [UNIFIED] Metadata saved: {meta_path}")

        # ✅ Attention 데이터 저장 (있는 경우) - NaN 값 안전 처리 (통합 저장)
        attention_data = prediction_results.get("attention_data")
        if attention_data:
            attention_save_data = {
                "image_base64": safe_serialize_value(attention_data.get("image", "")),
                "feature_importance": {},
                "temporal_importance": {}
            }
            
            # feature_importance 안전 처리
            if attention_data.get("feature_importance"):
                for k, v in attention_data["feature_importance"].items():
                    attention_save_data["feature_importance"][k] = safe_serialize_value(v)
            
            # temporal_importance 안전 처리  
            if attention_data.get("temporal_importance"):
                for k, v in attention_data["temporal_importance"].items():
                    attention_save_data["temporal_importance"][k] = safe_serialize_value(v)
            
            # 🌟 통합 저장소에 저장
            with open(attention_path, "w", encoding="utf-8") as fp:
                json.dump(attention_save_data, fp, ensure_ascii=False, indent=2)
            logger.info(f"✅ [UNIFIED] Attention saved: {attention_path}")

        # ✅ 이동평균 데이터 저장 (있는 경우) - NaN 값 안전 처리 (통합 저장)
        ma_results = prediction_results.get("ma_results")
        ma_file = None
        
        if ma_results:
            try:
                # MA 결과 안전 처리
                safe_ma_results = {}
                for window, results in ma_results.items():
                    safe_ma_results[str(window)] = []
                    for result in results:
                        if isinstance(result, dict):
                            safe_result = {}
                            for k, v in result.items():
                                safe_result[k] = safe_serialize_value(v)
                            safe_ma_results[str(window)].append(safe_result)
                        else:
                            safe_ma_results[str(window)].append(safe_serialize_value(result))
                
                # 🌟 통합 저장소에 저장
                with open(ma_path, "w", encoding="utf-8") as fp:
                    json.dump(safe_ma_results, fp, ensure_ascii=False, indent=2)
                logger.info(f"✅ [UNIFIED] MA results saved: {ma_path}")
                ma_file = str(ma_path)
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to save MA results: {str(e)}")
                logger.error(f"MA results error details: {traceback.format_exc()}")

        # ✅ 통합 인덱스 업데이트
        update_unified_predictions_index(safe_meta)
        
        logger.info(f"✅ Complete unified prediction save → start date: {meta['prediction_start_date']}")
        return {
            "success": True, 
            # 🌟 통합 저장소 파일들
            "csv_file": str(csv_path),
            "meta_file": str(meta_path),
            "attention_file": str(attention_path) if attention_data else None,
            "ma_file": ma_file,
            "hyperparameter_file": str(hyperparameter_path),
            # 공통 정보
            "prediction_start_date": meta["prediction_start_date"],
            "preserved_existing": csv_path.name != f"{base_name}.csv",
            "prediction_count": len(preds_seq),
            "storage_system": "unified_complete",
            "storage_directories": {
                "predictions": str(predictions_dir),
                "hyperparameters": str(hyperparameters_dir),
                "plots": str(plots_dir)
            }
        }

    except Exception as e:
        logger.error(f"❌ save_prediction_simple 오류: {e}")
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

# 2. Attention 데이터를 포함한 로드 함수
def load_prediction_simple(prediction_start_date):
    """
    단순화된 예측 결과 로드 함수
    """
    try:
        predictions_dir = Path(PREDICTIONS_DIR)
        
        if isinstance(prediction_start_date, str):
            start_date = pd.to_datetime(prediction_start_date)
        else:
            start_date = prediction_start_date
        
        date_str = start_date.strftime('%Y%m%d')
        
        csv_filepath = predictions_dir / f"prediction_{date_str}.csv"
        meta_filepath = predictions_dir / f"prediction_{date_str}_meta.json"
        
        if not csv_filepath.exists() or not meta_filepath.exists():
            # 🔧 호환 가능한 예측 결과 찾기 (확장된 파일의 경우)
            current_file = prediction_state.get('current_file')
            if current_file:
                logger.info(f"🔍 [PREDICTION_SIMPLE] 현재 파일 캐시에서 예측 결과를 찾을 수 없습니다. 호환 가능한 예측 결과를 찾는 중...")
                compatible_predictions = find_compatible_predictions(current_file, prediction_start_date)
                
                if compatible_predictions:
                    logger.info(f"✅ [PREDICTION_SIMPLE] 호환 가능한 예측 결과를 발견했습니다!")
                    logger.info(f"    📁 Source file: {os.path.basename(compatible_predictions['source_file'])}")
                    return {
                        'success': True,
                        'predictions': compatible_predictions['predictions'],
                        'metadata': compatible_predictions['metadata'],
                        'attention_data': compatible_predictions['attention_data'],
                        'source_info': {
                            'source_file': compatible_predictions['source_file'],
                            'extension_info': compatible_predictions['extension_info']
                        }
                    }
            
            return {'success': False, 'error': f'Prediction files not found for {start_date.strftime("%Y-%m-%d")}'}
        
        # CSV 로드 - 안전한 fallback 사용
        from app.data.loader import load_csv_safe_with_fallback
        predictions_df = load_csv_safe_with_fallback(csv_filepath)
        predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
        if 'Prediction_From' in predictions_df.columns:
            predictions_df['Prediction_From'] = pd.to_datetime(predictions_df['Prediction_From'])
        
        predictions = predictions_df.to_dict('records')
        
        # 메타데이터 로드
        with open(meta_filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        logger.info(f"Simple prediction load completed: {len(predictions)} predictions")
        
        return {
            'success': True,
            'predictions': predictions,
            'metadata': metadata,
            'attention_data': {
                'image': metadata.get('attention_map', {}).get('image_base64', ''),
                'feature_importance': metadata.get('attention_map', {}).get('feature_importance', {}),
                'temporal_importance': metadata.get('attention_map', {}).get('temporal_importance', {})
            }
        }
        
    except Exception as e:
        logger.error(f"Error loading prediction: {str(e)}")
        return {'success': False, 'error': str(e)}

def update_predictions_index_simple(metadata):
    """단순화된 예측 인덱스 업데이트 - 파일별 캐시 디렉토리 사용"""
    try:
        # 🔧 metadata가 None인 경우 처리
        if metadata is None:
            logger.warning("⚠️ [INDEX] metadata가 None입니다. 인덱스 업데이트를 건너뜁니다.")
            return False
            
        # 🎯 파일별 캐시 디렉토리 사용
        cache_dirs = get_file_cache_dirs()
        predictions_index_file = cache_dirs['predictions'] / 'predictions_index.cs'
        
        # 기존 인덱스 읽기
        index_data = []
        if predictions_index_file.exists():
            with open(predictions_index_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                index_data = list(reader)
        
        # 중복 제거
        prediction_start_date = metadata.get('prediction_start_date')
        if not prediction_start_date:
            logger.warning("⚠️ [INDEX] metadata에 prediction_start_date가 없습니다.")
            return False
            
        index_data = [row for row in index_data 
                     if row.get('prediction_start_date') != prediction_start_date]
        
        # metrics가 None일 수도 있으므로 안전하게 처리
        metrics = metadata.get('metrics') or {}
        
        # 새 데이터 추가 (🔧 필드명 수정)
        new_row = {
            'prediction_start_date': metadata.get('prediction_start_date', ''),
            'data_end_date': metadata.get('data_end_date', ''),
            'created_at': metadata.get('created_at', ''),
            'semimonthly_period': metadata.get('semimonthly_period', ''),
            'next_semimonthly_period': metadata.get('next_semimonthly_period', ''),
            'prediction_count': metadata.get('total_predictions', metadata.get('prediction_count', 0)),  # 🔧 수정
            'f1_score': metrics.get('f1', 0) if isinstance(metrics, dict) else 0,
            'accuracy': metrics.get('accuracy', 0) if isinstance(metrics, dict) else 0,
            'mape': metrics.get('mape', 0) if isinstance(metrics, dict) else 0,
            'weighted_score': metrics.get('weighted_score', 0) if isinstance(metrics, dict) else 0
        }
        index_data.append(new_row)
        
        # 날짜순 정렬 후 저장
        index_data.sort(key=lambda x: x.get('data_end_date', ''), reverse=True)
        
        if index_data:
            fieldnames = new_row.keys()  # 🔧 일관된 필드명 사용
            with open(predictions_index_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(index_data)
            
            logger.info(f"✅ Predictions index updated successfully: {len(index_data)} entries")
            logger.info(f"📄 Index file: {predictions_index_file}")
            return True
        else:
            logger.warning("⚠️ No data to write to index file")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error updating simple predictions index: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def rebuild_predictions_index_from_existing_files():
    """
    기존 예측 파일들로부터 predictions_index.cs를 재생성하는 함수 (보안을 위해 .cs 확장자 사용)
    🔧 누적 예측이 기존 단일 예측 캐시를 인식할 수 있도록 함
    """
    global predictions_index
    
    try:
        current_file = prediction_state.get('current_file')
        if not current_file:
            logger.warning("⚠️ No current file set, cannot rebuild index")
            return False
        
        cache_dirs = get_file_cache_dirs(current_file)
        predictions_dir = cache_dirs['predictions']
        predictions_index_file = predictions_dir / 'predictions_index.cs'
        
        logger.info(f"🔄 Rebuilding predictions index from existing files in: {predictions_dir}")
        
        # 기존 메타 파일들 찾기
        meta_files = list(predictions_dir.glob("*_meta.json"))
        logger.info(f"📋 Found {len(meta_files)} meta files")
        
        if not meta_files:
            logger.warning("⚠️ No meta files found to rebuild index")
            return False
        
        index_data = []
        
        for meta_file in meta_files:
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # 인덱스 레코드 생성 (동일한 필드명 사용)
                new_row = {
                    'prediction_start_date': metadata.get('prediction_start_date', ''),
                    'data_end_date': metadata.get('data_end_date', ''),
                    'created_at': metadata.get('created_at', ''),
                    'semimonthly_period': metadata.get('semimonthly_period', ''),
                    'next_semimonthly_period': metadata.get('next_semimonthly_period', ''),
                    'prediction_count': metadata.get('total_predictions', metadata.get('prediction_count', 0)),
                    'f1_score': metadata.get('metrics', {}).get('f1', 0),
                    'accuracy': metadata.get('metrics', {}).get('accuracy', 0),
                    'mape': metadata.get('metrics', {}).get('mape', 0),
                    'weighted_score': metadata.get('metrics', {}).get('weighted_score', 0)
                }
                
                index_data.append(new_row)
                logger.info(f"  ✅ {meta_file.name}: {new_row['prediction_start_date']}")
                
            except Exception as e:
                logger.warning(f"  ⚠️  Error reading {meta_file.name}: {str(e)}")
                continue
        
        if not index_data:
            logger.error("❌ No valid metadata found")
            return False
        
        # 날짜순 정렬
        index_data.sort(key=lambda x: x.get('data_end_date', ''), reverse=True)
        
        # CSV 파일 생성
        fieldnames = index_data[0].keys()
        
        with open(predictions_index_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(index_data)
        
        logger.info(f"✅ Successfully rebuilt predictions_index.cs with {len(index_data)} entries")
        logger.info(f"📄 Index file: {predictions_index_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error rebuilding predictions index: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def rebuild_predictions_index_from_existing_files():
    """
    기존 예측 파일들로부터 predictions_index.cs를 재생성하는 함수 (보안을 위해 .cs 확장자 사용)
    🔧 누적 예측이 기존 단일 예측 캐시를 인식할 수 있도록 함
    """
    try:
        current_file = prediction_state.get('current_file')
        if not current_file:
            logger.warning("⚠️ No current file set, cannot rebuild index")
            return False
        
        cache_dirs = get_file_cache_dirs(current_file)
        predictions_dir = cache_dirs['predictions']
        predictions_index_file = predictions_dir / 'predictions_index.cs'
        
        logger.info(f"🔄 Rebuilding predictions index from existing files in: {predictions_dir}")
        
        # 기존 메타 파일들 찾기
        meta_files = list(predictions_dir.glob("*_meta.json"))
        logger.info(f"📋 Found {len(meta_files)} meta files")
        
        if not meta_files:
            logger.warning("⚠️ No meta files found to rebuild index")
            return False
        
        index_data = []
        
        for meta_file in meta_files:
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # 인덱스 레코드 생성 (동일한 필드명 사용)
                new_row = {
                    'prediction_start_date': metadata.get('prediction_start_date', ''),
                    'data_end_date': metadata.get('data_end_date', ''),
                    'created_at': metadata.get('created_at', ''),
                    'semimonthly_period': metadata.get('semimonthly_period', ''),
                    'next_semimonthly_period': metadata.get('next_semimonthly_period', ''),
                    'prediction_count': metadata.get('total_predictions', metadata.get('prediction_count', 0)),
                    'f1_score': metadata.get('metrics', {}).get('f1', 0),
                    'accuracy': metadata.get('metrics', {}).get('accuracy', 0),
                    'mape': metadata.get('metrics', {}).get('mape', 0),
                    'weighted_score': metadata.get('metrics', {}).get('weighted_score', 0)
                }
                
                index_data.append(new_row)
                logger.info(f"  ✅ {meta_file.name}: {new_row['prediction_start_date']}")
                
            except Exception as e:
                logger.warning(f"  ⚠️  Error reading {meta_file.name}: {str(e)}")
                continue
        
        if not index_data:
            logger.error("❌ No valid metadata found")
            return False
        
        # 날짜순 정렬
        index_data.sort(key=lambda x: x.get('data_end_date', ''), reverse=True)
        
        # CSV 파일 생성
        fieldnames = index_data[0].keys()
        
        with open(predictions_index_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(index_data)
        
        logger.info(f"✅ Successfully rebuilt predictions_index.cs with {len(index_data)} entries")
        logger.info(f"📄 Index file: {predictions_index_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error rebuilding predictions index: {str(e)}")
        logger.error(traceback.format_exc())
        return False
    
def update_cached_prediction_actual_values(prediction_start_date, update_latest_only=True):
    """
    캐시된 예측의 실제값만 선택적으로 업데이트하는 최적화된 함수
    
    Args:
        prediction_start_date: 예측 시작 날짜
        update_latest_only: True면 최신 데이터만 체크하여 성능 최적화
    
    Returns:
        dict: 업데이트 결과
    """
    try:
        from app.data.loader import load_data
        current_file = prediction_state.get('current_file')
        if not current_file:
            return {'success': False, 'error': 'No current file context available'}
        
        # 캐시된 예측 로드 (실제값 업데이트 없이)
        cached_result = load_prediction_with_attention_from_csv(prediction_start_date)
        if not cached_result['success']:
            return cached_result
        
        predictions = cached_result['predictions']
        
        # 데이터 로드 (캐시 활용)
        logger.info(f"🔄 [ACTUAL_UPDATE] Loading data for actual value update...")
        from app.data.loader import load_data
        df = load_data(current_file, use_cache=True)
        
        if df is None or df.empty:
            logger.warning(f"⚠️ [ACTUAL_UPDATE] Could not load data file")
            return {'success': False, 'error': 'Could not load data file'}
        
        last_data_date = df.index.max()
        updated_count = 0
        
        # 각 예측에 대해 실제값 확인 및 설정
        for pred in predictions:
            pred_date = pd.to_datetime(pred['Date'])
            
            # 최신 데이터만 체크하는 경우 성능 최적화
            if update_latest_only and pred_date < last_data_date - pd.Timedelta(days=30):
                continue
            
            # 실제 데이터가 존재하는 날짜면 실제값 설정
            if (pred_date in df.index and 
                pd.notna(df.loc[pred_date, 'MOPJ']) and 
                pred_date <= last_data_date):
                actual_val = float(df.loc[pred_date, 'MOPJ'])
                pred['Actual'] = actual_val
                updated_count += 1
                logger.debug(f"  📊 Updated actual value for {pred_date.strftime('%Y-%m-%d')}: {actual_val:.2f}")
            elif 'Actual' not in pred or pred['Actual'] is None:
                pred['Actual'] = None
        
        logger.info(f"✅ [ACTUAL_UPDATE] Updated {updated_count} actual values")
        
        # 업데이트된 결과 반환
        cached_result['predictions'] = predictions
        cached_result['actual_values_updated'] = True
        cached_result['updated_count'] = updated_count
        
        return cached_result
        
    except Exception as e:
        logger.error(f"❌ [ACTUAL_UPDATE] Error updating actual values: {str(e)}")
        return {'success': False, 'error': str(e)}

def load_prediction_from_csv(prediction_start_date_or_data_end_date):
    """
    하위 호환성을 위한 함수 - 자동으로 새로운 함수로 리다이렉트
    """
    logger.info("Using compatibility wrapper - redirecting to new smart cache function")
    return load_prediction_with_attention_from_csv(prediction_start_date_or_data_end_date)

# xlwings 대안 로더 (보안프로그램이 파일을 잠그는 경우 사용)
try:
    import xlwings as xw
    XLWINGS_AVAILABLE = True
    logger.info("✅ xlwings library available - Excel security bypass enabled")
except ImportError:
    XLWINGS_AVAILABLE = False
    logger.warning("⚠️ xlwings not available - falling back to pandas only")

def load_prediction_with_attention_from_csv_in_dir(prediction_start_date, file_predictions_dir):
    """
    파일별 디렉토리에서 저장된 예측 결과와 attention 데이터를 함께 불러오는 함수
    """
    try:
        if isinstance(prediction_start_date, str):
            start_date = pd.to_datetime(prediction_start_date)
        else:
            start_date = prediction_start_date
        
        date_str = start_date.strftime('%Y%m%d')
        
        # 파일별 디렉토리에서 파일 경로 설정 (보안을 위해 .cs 확장자 사용)
        csv_filepath = file_predictions_dir / f"prediction_start_{date_str}.cs"
        meta_filepath = file_predictions_dir / f"prediction_start_{date_str}_meta.json"
        attention_filepath = file_predictions_dir / f"prediction_start_{date_str}_attention.json"
        ma_filepath = file_predictions_dir / f"prediction_start_{date_str}_ma.json"
        
        logger.info(f"📂 Loading from file directory: {file_predictions_dir.name}")
        logger.info(f"  📄 CSV: {csv_filepath.name}")
        
        if not csv_filepath.exists() or not meta_filepath.exists():
            logger.warning(f"  ❌ Required files missing in {file_predictions_dir.name}")
            
            # 🔧 호환 가능한 예측 결과 찾기 (확장된 파일의 경우)
            current_file = prediction_state.get('current_file')
            if current_file:
                logger.info(f"🔍 [PREDICTION_DIR] 현재 파일 캐시에서 예측 결과를 찾을 수 없습니다. 호환 가능한 예측 결과를 찾는 중...")
                compatible_predictions = find_compatible_predictions(current_file, prediction_start_date)
                
                if compatible_predictions:
                    logger.info(f"✅ [PREDICTION_DIR] 호환 가능한 예측 결과를 발견했습니다!")
                    logger.info(f"    📁 Source file: {os.path.basename(compatible_predictions['source_file'])}")
                    return {
                        'success': True,
                        'predictions': compatible_predictions['predictions'],
                        'metadata': compatible_predictions['metadata'],
                        'attention_data': compatible_predictions['attention_data'],
                        'ma_results': compatible_predictions['ma_results'],
                        'source_info': {
                            'source_file': compatible_predictions['source_file'],
                            'extension_info': compatible_predictions['extension_info']
                        }
                    }
            
            return {'success': False, 'error': f'Prediction files not found for {start_date.strftime("%Y-%m-%d")}'}
        
        # CSV 로드 - .cs 파일 호환 fallback 사용
        from app.data.loader import load_csv_safe_with_fallback
        predictions_df = load_csv_safe_with_fallback(csv_filepath)
        
        # 🔧 컬럼명 호환성 처리: 소문자로 저장된 컬럼을 대문자로 변환 및 중복 제거
        if 'date' in predictions_df.columns:
            predictions_df['Date'] = pd.to_datetime(predictions_df['date'])
            predictions_df.drop('date', axis=1, inplace=True)  # 원본 소문자 컬럼 제거
        elif 'Date' in predictions_df.columns:
            predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
        
        if 'prediction' in predictions_df.columns:
            predictions_df['Prediction'] = predictions_df['prediction']
            predictions_df.drop('prediction', axis=1, inplace=True)  # 원본 소문자 컬럼 제거
        
        if 'prediction_from' in predictions_df.columns:
            predictions_df['Prediction_From'] = pd.to_datetime(predictions_df['prediction_from'])
            predictions_df.drop('prediction_from', axis=1, inplace=True)  # 원본 소문자 컬럼 제거
        elif 'Prediction_From' in predictions_df.columns:
            predictions_df['Prediction_From'] = pd.to_datetime(predictions_df['Prediction_From'])
        
        # actual 컬럼도 호환성 처리
        if 'actual' in predictions_df.columns:
            predictions_df['Actual'] = pd.to_numeric(predictions_df['actual'], errors='coerce')
            predictions_df.drop('actual', axis=1, inplace=True)  # 원본 소문자 컬럼 제거
        
        logger.info(f"📊 [CSV_DIR_LOAD] DataFrame processed: {predictions_df.shape}")
        logger.info(f"📋 [CSV_DIR_LOAD] Final columns: {list(predictions_df.columns)}")
        
        predictions = predictions_df.to_dict('records')
        
        # ✅ JSON 직렬화를 위해 Timestamp 객체들을 문자열로 안전하게 변환
        for pred in predictions:
            for key, value in list(pred.items()):
                if pd.isna(value):
                    pred[key] = None
                elif isinstance(value, pd.Timestamp):
                    pred[key] = value.strftime('%Y-%m-%d')
                elif isinstance(value, (np.int64, np.float64)):
                    # 예측값과 실제값은 모두 float로 유지
                    pred[key] = float(value)
                elif hasattr(value, 'item'):  # numpy scalars
                    pred[key] = value.item()
        
        # ✅ 캐시에서 로드할 때 실제값 다시 설정 (선택적 - 성능 최적화)
        # 💡 캐시된 예측을 빠르게 불러오기 위해 실제값 업데이트를 스킵
        # 필요시에만 별도 API로 실제값 업데이트 수행
        logger.info(f"📦 [CACHE_FAST] Skipping actual value update for faster cache loading")
        logger.info(f"💡 [CACHE_FAST] Use separate API endpoint if actual value update is needed")
        
        # 메타데이터 로드
        with open(meta_filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Attention 데이터 로드
        attention_data = {}
        if attention_filepath.exists():
            try:
                with open(attention_filepath, 'r', encoding='utf-8') as f:
                    attention_raw = json.load(f)
                attention_data = {
                    'image': attention_raw.get('image_base64', ''),
                    'feature_importance': attention_raw.get('feature_importance', {}),
                    'temporal_importance': attention_raw.get('temporal_importance', {})
                }
                logger.info(f"  🧠 Attention data loaded successfully")
                logger.info(f"  🧠 Image data length: {len(attention_data['image']) if attention_data['image'] else 0}")
                logger.info(f"  🧠 Feature importance keys: {len(attention_data['feature_importance'])}")
                logger.info(f"  🧠 Temporal importance keys: {len(attention_data['temporal_importance'])}")
            except Exception as e:
                logger.warning(f"  ⚠️  Failed to load attention data: {str(e)}")
                attention_data = {}
        
        # 이동평균 데이터 로드
        ma_results = {}
        if ma_filepath.exists():
            try:
                with open(ma_filepath, 'r', encoding='utf-8') as f:
                    ma_results = json.load(f)
                logger.info(f"  📊 MA results loaded successfully")
            except Exception as e:
                logger.warning(f"  ⚠️  Failed to load MA results: {str(e)}")
        
        logger.info(f"✅ File directory cache load completed: {len(predictions)} predictions")
        
        return {
            'success': True,
            'predictions': predictions,
            'metadata': metadata,
            'attention_data': attention_data,
            'ma_results': ma_results
        }
        
    except Exception as e:
        logger.error(f"❌ Error loading prediction from file directory: {str(e)}")
        return {'success': False, 'error': str(e)}

def load_prediction_with_attention_from_csv(prediction_start_date):
    """
    통합 predictions 폴더에서 우선 로드하는 개선된 함수 🌟
    """
    try:
        # 🌟 통합 예측 디렉토리에서 우선 시도 (Primary)
        result = load_prediction_from_unified_storage(prediction_start_date)
        if result.get('success'):
            logger.info(f"✅ [UNIFIED_LOAD] Successfully loaded from unified storage")
            return result
        
        # 🎯 폴백: 파일별 캐시에서 로드 시도 (Secondary)
        logger.info(f"🔄 [FALLBACK_LOAD] Trying file-specific cache...")
        current_file = prediction_state.get('current_file')
        if not current_file:
            logger.error("❌ No current file set in prediction_state")
            return {'success': False, 'error': 'No current file context available'}
            
        cache_dirs = get_file_cache_dirs(current_file)
        predictions_dir = cache_dirs['predictions']
        
        if isinstance(prediction_start_date, str):
            start_date = pd.to_datetime(prediction_start_date)
        else:
            start_date = prediction_start_date
        
        date_str = start_date.strftime('%Y%m%d')
        
        # 파일 경로들 (보안을 위해 .cs 확장자 사용)
        csv_filepath = predictions_dir / f"prediction_start_{date_str}.cs"
        meta_filepath = predictions_dir / f"prediction_start_{date_str}_meta.json"
        attention_filepath = predictions_dir / f"prediction_start_{date_str}_attention.json"
        
        # 필수 파일 존재 확인
        if not csv_filepath.exists() or not meta_filepath.exists():
            # 🔧 호환 가능한 예측 결과 찾기 (확장된 파일의 경우)
            logger.info(f"🔍 [PREDICTION_LOAD] 현재 파일 캐시에서 예측 결과를 찾을 수 없습니다. 호환 가능한 예측 결과를 찾는 중...")
            compatible_predictions = find_compatible_predictions(current_file, prediction_start_date)
            
            if compatible_predictions:
                logger.info(f"✅ [PREDICTION_LOAD] 호환 가능한 예측 결과를 발견했습니다!")
                logger.info(f"    📁 Source file: {os.path.basename(compatible_predictions['source_file'])}")
                return {
                    'success': True,
                    'predictions': compatible_predictions['predictions'],
                    'metadata': compatible_predictions['metadata'],
                    'attention_data': compatible_predictions['attention_data'],
                    'ma_results': compatible_predictions['ma_results'],
                    'source_info': {
                        'source_file': compatible_predictions['source_file'],
                        'extension_info': compatible_predictions['extension_info']
                    }
                }
            else:
                return {
                    'success': False,
                    'error': f'Prediction files not found for start date {start_date.strftime("%Y-%m-%d")}'
                }
        
        # CSV 파일 읽기 - 안전한 fallback 사용
        from app.data.loader import load_csv_safe_with_fallback
        predictions_df = load_csv_safe_with_fallback(csv_filepath)
        
        # 🔧 컬럼명 호환성 처리: 소문자로 저장된 컬럼을 대문자로 변환
        if 'date' in predictions_df.columns:
            predictions_df['Date'] = pd.to_datetime(predictions_df['date'])
        elif 'Date' in predictions_df.columns:
            predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
        
        if 'prediction' in predictions_df.columns:
            predictions_df['Prediction'] = predictions_df['prediction']
        
        if 'prediction_from' in predictions_df.columns:
            predictions_df['Prediction_From'] = pd.to_datetime(predictions_df['prediction_from'])
        elif 'Prediction_From' in predictions_df.columns:
            predictions_df['Prediction_From'] = pd.to_datetime(predictions_df['Prediction_From'])
        
        predictions = predictions_df.to_dict('records')
        
        # ✅ JSON 직렬화를 위해 Timestamp 객체들을 문자열로 안전하게 변환
        for pred in predictions:
            for key, value in list(pred.items()):
                if pd.isna(value):
                    pred[key] = None
                elif isinstance(value, pd.Timestamp):
                    pred[key] = value.strftime('%Y-%m-%d')
                elif isinstance(value, (np.int64, np.float64)):
                    # 예측값과 실제값은 모두 float로 유지
                    pred[key] = float(value)
                elif hasattr(value, 'item'):  # numpy scalars
                    pred[key] = value.item()
        
        # ✅ 캐시에서 로드할 때 실제값 다시 설정 (선택적 - 성능 최적화)
        # 💡 캐시된 예측을 빠르게 불러오기 위해 실제값 업데이트를 스킵
        # 필요시에만 별도 API로 실제값 업데이트 수행
        logger.info(f"📦 [CACHE_FAST] Skipping actual value update for faster cache loading")
        logger.info(f"💡 [CACHE_FAST] Use separate API endpoint if actual value update is needed")
        
        # 메타데이터 읽기
        with open(meta_filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Attention 데이터 읽기 (있는 경우)
        attention_data = None
        if attention_filepath.exists():
            try:
                with open(attention_filepath, 'r', encoding='utf-8') as f:
                    stored_attention = json.load(f)
                
                attention_data = {
                    'image': stored_attention.get('image_base64', ''),
                    'file_path': None,  # 이미지는 base64로 저장됨
                    'feature_importance': stored_attention.get('feature_importance', {}),
                    'temporal_importance': stored_attention.get('temporal_importance', {})
                }
                logger.info(f"Attention data loaded from: {attention_filepath}")
            except Exception as e:
                logger.warning(f"Failed to load attention data: {str(e)}")
                attention_data = None

        # 🔄 이동평균 데이터 읽기 (있는 경우)
        ma_filepath = predictions_dir / f"prediction_start_{date_str}_ma.json"
        ma_results = None
        if ma_filepath.exists():
            try:
                with open(ma_filepath, 'r', encoding='utf-8') as f:
                    ma_results = json.load(f)
                logger.info(f"MA results loaded from: {ma_filepath} ({len(ma_results)} windows)")
            except Exception as e:
                logger.warning(f"Failed to load MA results: {str(e)}")
                ma_results = None
        
        logger.info(f"Complete prediction data loaded: {csv_filepath} ({len(predictions)} predictions)")
        
        return {
            'success': True,
            'predictions': predictions,
            'metadata': metadata,
            'attention_data': attention_data,
            'ma_results': ma_results,  # 🔑 이동평균 데이터 추가
            'prediction_start_date': start_date.strftime('%Y-%m-%d'),
            'data_end_date': metadata.get('data_end_date'),
            'semimonthly_period': metadata['semimonthly_period'],
            'next_semimonthly_period': metadata['next_semimonthly_period'],
            'metrics': metadata['metrics'],
            'interval_scores': metadata['interval_scores'],
            'selected_features': metadata['selected_features'],
            'has_cached_attention': attention_data is not None,
            'has_cached_ma': ma_results is not None  # 🔑 MA 캐시 여부 추가
        }
        
    except Exception as e:
        logger.error(f"Error loading prediction with attention: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'error': str(e)
        }

def get_saved_predictions_list_for_file(file_path, limit=100):
    """
    특정 파일의 캐시 디렉토리에서 저장된 예측 결과 목록을 조회하는 함수
    
    Parameters:
    -----------
    file_path : str
        현재 파일 경로
    limit : int
        반환할 최대 개수
    
    Returns:
    --------
    list : 저장된 예측 목록
    """
    try:
        predictions_list = []
        
        # 파일별 캐시 디렉토리 경로 구성
        cache_dirs = get_file_cache_dirs(file_path)
        predictions_dir = Path(cache_dirs['predictions'])
        predictions_index_file = predictions_dir / 'predictions_index.cs'
        
        logger.info(f"🔍 [CACHE] Searching predictions in: {predictions_dir}")
        
        if predictions_index_file.exists():
            with open(predictions_index_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if len(predictions_list) >= limit:
                        break
                    
                    prediction_start_date = row.get('prediction_start_date', row.get('first_prediction_date'))
                    data_end_date = row.get('data_end_date', row.get('prediction_base_date', row.get('prediction_date')))
                    
                    if prediction_start_date and data_end_date:
                        pred_info = {
                            'prediction_start_date': prediction_start_date,
                            'data_end_date': data_end_date,
                            'prediction_date': data_end_date,
                            'first_prediction_date': prediction_start_date,
                            'created_at': row.get('created_at'),
                            'semimonthly_period': row.get('semimonthly_period'),
                            'next_semimonthly_period': row.get('next_semimonthly_period'),
                            'prediction_count': row.get('prediction_count'),
                            'actual_business_days': row.get('actual_business_days'),
                            'csv_file': row.get('csv_file'),
                            'meta_file': row.get('meta_file'),
                            'f1_score': float(row.get('f1_score', 0)),
                            'accuracy': float(row.get('accuracy', 0)),
                            'mape': float(row.get('mape', 0)),
                            'weighted_score': float(row.get('weighted_score', 0)),
                            'naming_scheme': row.get('naming_scheme', 'file_based'),
                            'source_file': os.path.basename(file_path),
                            'cache_system': 'file_based'
                        }
                        predictions_list.append(pred_info)
            
            logger.info(f"🎯 [CACHE] Found {len(predictions_list)} predictions in file-specific cache")
        else:
            logger.info(f"📂 [CACHE] No predictions index found in {predictions_index_file}")
        
        # 날짜순으로 정렬 (최신 순)
        predictions_list.sort(key=lambda x: x['data_end_date'], reverse=True)
        
        return predictions_list
        
    except Exception as e:
        logger.error(f"Error reading file-specific predictions list: {str(e)}")
        return []

def get_saved_predictions_list(limit=100):
    """
    통합 predictions 폴더에서 우선 조회하는 개선된 함수 🌟
    
    Parameters:
    -----------
    limit : int
        반환할 최대 개수
    
    Returns:
    --------
    list : 저장된 예측 목록
    """
    try:
        # 🌟 통합 저장소에서 우선 조회 (Primary)
        predictions_list = get_unified_predictions_list(limit)
        
        if len(predictions_list) > 0:
            logger.info(f"🌟 [UNIFIED_LIST] Retrieved {len(predictions_list)} predictions from unified storage")
            return predictions_list
        
        # 🎯 폴백: 파일별 캐시 시스템에서 검색 (Secondary - 호환성)
        logger.info(f"🔄 [FALLBACK_LIST] No unified predictions, trying file-based cache...")
        
        # 1. 파일별 캐시 시스템에서 예측 검색
        cache_root = Path(CACHE_ROOT_DIR)
        if cache_root.exists():
            for file_dir in cache_root.iterdir():
                if not file_dir.is_dir():
                    continue
                
                predictions_dir = file_dir / 'predictions'
                predictions_index_file = predictions_dir / 'predictions_index.cs'
                
                if predictions_index_file.exists():
                    with open(predictions_index_file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if len(predictions_list) >= limit:
                                break
                            
                            prediction_start_date = row.get('prediction_start_date', row.get('first_prediction_date'))
                            data_end_date = row.get('data_end_date', row.get('prediction_base_date', row.get('prediction_date')))
                            
                            if prediction_start_date and data_end_date:
                                pred_info = {
                                    'prediction_start_date': prediction_start_date,
                                    'data_end_date': data_end_date,
                                    'prediction_date': data_end_date,
                                    'first_prediction_date': prediction_start_date,
                                    'created_at': row.get('created_at'),
                                    'semimonthly_period': row.get('semimonthly_period'),
                                    'next_semimonthly_period': row.get('next_semimonthly_period'),
                                    'prediction_count': row.get('prediction_count'),
                                    'actual_business_days': row.get('actual_business_days'),
                                    'csv_file': row.get('csv_file'),
                                    'meta_file': row.get('meta_file'),
                                    'f1_score': float(row.get('f1_score', 0)),
                                    'accuracy': float(row.get('accuracy', 0)),
                                    'mape': float(row.get('mape', 0)),
                                    'weighted_score': float(row.get('weighted_score', 0)),
                                    'naming_scheme': row.get('naming_scheme', 'file_based'),
                                    'source_file': file_dir.name,
                                    'cache_system': 'file_based',
                                    'storage_system': 'file_based'  # 구분을 위해 추가
                                }
                                predictions_list.append(pred_info)
        
        if len(predictions_list) == 0:
            logger.info("❌ [FALLBACK_LIST] No predictions found in file-based cache system")
        else:
            logger.info(f"🎯 [FALLBACK_LIST] Retrieved {len(predictions_list)} predictions from file-based cache")
        
        # 날짜순으로 정렬 (최신 순)
        predictions_list.sort(key=lambda x: x['data_end_date'], reverse=True)
        
        return predictions_list
        
    except Exception as e:
        logger.error(f"❌ Error reading predictions list: {str(e)}")
        return []

def load_accumulated_predictions_from_csv(start_date, end_date=None, limit=None, file_path=None):
    """
    CSV에서 누적 예측 결과를 빠르게 불러오는 함수 (수정됨)
    - 주어진 file_path의 캐시 디렉토리 내에서만 검색하여 명확성 확보
    """
    try:
        if not file_path:
            logger.warning("⚠️ load_accumulated_predictions_from_csv: file_path is required for accurate cache loading.")
            return []

        logger.info(f"🔍 [CACHE_LOAD] Loading predictions for '{os.path.basename(file_path)}' from {start_date} to {end_date or 'latest'}")

        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if end_date and isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # ✅ 1. 현재 파일의 캐시 목록만 가져오기
        predictions_list = get_saved_predictions_list_for_file(file_path, limit=1000)

        # ✅ 2. 날짜 범위 필터링
        filtered_predictions_info = []
        for pred_info in predictions_list:
            data_end_date = pd.to_datetime(pred_info.get('data_end_date'))
            if data_end_date >= start_date and (end_date is None or data_end_date <= end_date):
                filtered_predictions_info.append(pred_info)
        
        logger.info(f"📋 [CACHE] Found {len(filtered_predictions_info)} matching prediction files in the specified date range.")

        # ✅ 3. 각 예측 결과 파일 로드
        accumulated_results = []
        cache_dirs = get_file_cache_dirs(file_path)
        predictions_dir = cache_dirs['predictions']

        for pred_info in filtered_predictions_info:
            try:
                # data_end_date를 기준으로 예측 시작일을 다시 계산하여 정확한 파일 로드
                data_end_date = pd.to_datetime(pred_info.get('data_end_date'))
                prediction_start_date = data_end_date + pd.Timedelta(days=1)
                while prediction_start_date.weekday() >= 5 or is_holiday(prediction_start_date):
                    prediction_start_date += pd.Timedelta(days=1)

                loaded_result = load_prediction_with_attention_from_csv_in_dir(prediction_start_date, predictions_dir)

                if loaded_result.get('success'):
                    metadata = loaded_result.get('metadata', {})
                    predictions = loaded_result.get('predictions', [])
                    
                    # 누적 예측 형식에 맞게 변환
                    accumulated_item = {
                        'date': data_end_date.strftime('%Y-%m-%d'),
                        'predictions': predictions,
                        'metrics': metadata.get('metrics', {}),
                        'interval_scores': metadata.get('interval_scores', {}),
                        # ... 기타 필요한 정보 ...
                    }
                    accumulated_results.append(accumulated_item)
                else:
                    logger.warning(f"  ❌ [CACHE] Failed to load prediction for {data_end_date.strftime('%Y-%m-%d')}: {loaded_result.get('error')}")
            except Exception as e:
                logger.error(f"  ❌ Error loading individual prediction file: {str(e)}")
        
        logger.info(f"🎯 [CACHE] Successfully loaded {len(accumulated_results)} predictions from CSV cache files.")
        return accumulated_results

    except Exception as e:
        logger.error(f"Error loading accumulated predictions from CSV: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def delete_saved_prediction(prediction_date):
    """
    저장된 예측 결과를 삭제하는 함수
    
    Parameters:
    -----------
    prediction_date : str or datetime
        삭제할 예측 날짜
    
    Returns:
    --------
    dict : 삭제 결과
    """
    try:
        # 날짜 형식 통일
        if isinstance(prediction_date, str):
            pred_date = pd.to_datetime(prediction_date)
        else:
            pred_date = prediction_date
        
        date_str = pred_date.strftime('%Y%m%d')
        
        # 파일 경로들 (TARGET_DATE 방식)
        csv_filepath = os.path.join(PREDICTIONS_DIR, f"prediction_target_{date_str}.csv")
        meta_filepath = os.path.join(PREDICTIONS_DIR, f"prediction_target_{date_str}_meta.json")
        
        # 파일 삭제
        deleted_files = []
        if os.path.exists(csv_filepath):
            os.remove(csv_filepath)
            deleted_files.append(csv_filepath)
        
        if os.path.exists(meta_filepath):
            os.remove(meta_filepath)
            deleted_files.append(meta_filepath)
        
        # 🚫 레거시 인덱스 제거 기능은 파일별 캐시 시스템에서 제거됨
        # 파일별 캐시에서는 각 파일의 predictions_index.cs가 자동으로 관리됨
        logger.info("⚠️ Legacy delete_saved_prediction function called - not supported in file-based cache system")
        
        return {
            'success': True,
            'deleted_files': deleted_files,
            'message': f'Prediction for {pred_date.strftime("%Y-%m-%d")} deleted successfully'
        }
        
    except Exception as e:
        logger.error(f"Error deleting saved prediction: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }
    
def save_varmax_prediction(prediction_results: dict, prediction_date):
    """
    VARMAX 예측 결과를 파일에 저장하는 함수
    """
    try:
        file_path = prediction_state.get('current_file', None)
        if not file_path:
            logger.warning("No current file path for VARMAX prediction save")
            return False
            
        # 파일별 캐시 디렉토리 가져오기
        cache_dirs = get_file_cache_dirs(file_path)
        varmax_dir = cache_dirs['root'] / 'varmax'
        varmax_dir.mkdir(exist_ok=True)
        
        # 저장할 파일 경로
        prediction_file = varmax_dir / f"varmax_prediction_{prediction_date}.json"
        
        # JSON으로 직렬화 가능한 형태로 변환
        clean_results = {}
        for key, value in prediction_results.items():
            try:
                clean_results[key] = safe_serialize_value(value)
            except Exception as e:
                logger.warning(f"Failed to serialize {key}: {e}")
                continue
        
        # 메타데이터 추가
        clean_results['metadata'] = {
            'prediction_date': prediction_date,
            'created_at': datetime.now().isoformat(),
            'file_path': file_path,
            'model_type': 'VARMAX'
        }
        
        # 파일에 저장
        with open(prediction_file, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, ensure_ascii=False, indent=2)
        
        # 인덱스 업데이트
        update_varmax_predictions_index({
            'prediction_date': prediction_date,
            'file_path': str(prediction_file),
            'created_at': datetime.now().isoformat(),
            'original_file': file_path
        })
        
        logger.info(f"✅ VARMAX prediction saved: {prediction_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save VARMAX prediction: {e}")
        logger.error(traceback.format_exc())
        return False

def load_varmax_prediction(prediction_date):
    """
    저장된 VARMAX 예측 결과를 로드하는 함수
    """
    try:
        file_path = prediction_state.get('current_file', None)
        if not file_path:
            logger.warning("No current file path for VARMAX prediction load")
            return None
            
        # 파일별 캐시 디렉토리 가져오기
        cache_dirs = get_file_cache_dirs(file_path)
        varmax_dir = cache_dirs['root'] / 'varmax'
        
        # 로드할 파일 경로
        prediction_file = varmax_dir / f"varmax_prediction_{prediction_date}.json"
        
        if not prediction_file.exists():
            logger.info(f"VARMAX prediction file not found: {prediction_file}")
            return None
            
        # 파일에서 로드
        with open(prediction_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 🔍 로드된 데이터 타입 및 구조 확인
        logger.info(f"🔍 [VARMAX_LOAD] Loaded data type: {type(results)}")
        if isinstance(results, dict):
            logger.info(f"🔍 [VARMAX_LOAD] Loaded data keys: {list(results.keys())}")
            
            # 🔧 ma_results 필드 타입 확인 및 수정
            if 'ma_results' in results:
                ma_results = results['ma_results']
                logger.info(f"🔍 [VARMAX_LOAD] MA results type: {type(ma_results)}")
                
                if isinstance(ma_results, str):
                    logger.warning(f"⚠️ [VARMAX_LOAD] MA results is string, attempting to parse as JSON...")
                    try:
                        results['ma_results'] = json.loads(ma_results)
                        logger.info(f"🔧 [VARMAX_LOAD] Successfully parsed ma_results from string to dict")
                    except Exception as e:
                        logger.error(f"❌ [VARMAX_LOAD] Failed to parse ma_results string as JSON: {e}")
                        results['ma_results'] = {}
                elif not isinstance(ma_results, dict):
                    logger.warning(f"⚠️ [VARMAX_LOAD] MA results has unexpected type: {type(ma_results)}, setting empty dict")
                    results['ma_results'] = {}
                    
        elif isinstance(results, str):
            logger.warning(f"⚠️ [VARMAX_LOAD] Loaded data is string, not dict: {results[:100]}...")
            # 문자열인 경우 다시 JSON 파싱 시도
            try:
                results = json.loads(results)
                logger.info(f"🔧 [VARMAX_LOAD] Re-parsed string as JSON: {type(results)}")
            except:
                logger.error(f"❌ [VARMAX_LOAD] Failed to re-parse string as JSON")
                return None
        else:
            logger.warning(f"⚠️ [VARMAX_LOAD] Unexpected data type: {type(results)}")
        
        logger.info(f"✅ VARMAX prediction loaded: {prediction_file}")
        return results
        
    except Exception as e:
        logger.error(f"❌ Failed to load VARMAX prediction: {e}")
        logger.error(traceback.format_exc())
        return None

def update_varmax_predictions_index(metadata):
    """
    VARMAX 예측 인덱스를 업데이트하는 함수
    """
    try:
        file_path = prediction_state.get('current_file', None)
        if not file_path:
            return False
            
        cache_dirs = get_file_cache_dirs(file_path)
        varmax_dir = cache_dirs['root'] / 'varmax'
        varmax_dir.mkdir(exist_ok=True)
        
        index_file = varmax_dir / 'varmax_index.json'
        
        # 기존 인덱스 로드
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
        else:
            index = {'predictions': []}
        
        # 새 예측 추가 (중복 제거)
        prediction_date = metadata['prediction_date']
        index['predictions'] = [p for p in index['predictions'] if p['prediction_date'] != prediction_date]
        index['predictions'].append(metadata)
        
        # 날짜순 정렬 (최신순)
        index['predictions'].sort(key=lambda x: x['prediction_date'], reverse=True)
        
        # 최대 100개 유지
        index['predictions'] = index['predictions'][:100]
        
        # 인덱스 저장
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update VARMAX predictions index: {e}")
        return False

def get_saved_varmax_predictions_list(limit=100):
    """
    저장된 VARMAX 예측 목록을 가져오는 함수
    """
    try:
        file_path = prediction_state.get('current_file', None)
        if not file_path:
            logger.warning("No current file path for VARMAX predictions list")
            return []
            
        cache_dirs = get_file_cache_dirs(file_path)
        varmax_dir = cache_dirs['root'] / 'varmax'
        index_file = varmax_dir / 'varmax_index.json'
        
        if not index_file.exists():
            return []
            
        with open(index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)
        
        predictions = index.get('predictions', [])[:limit]
        
        logger.info(f"✅ Found {len(predictions)} saved VARMAX predictions")
        return predictions
        
    except Exception as e:
        logger.error(f"❌ Failed to get saved VARMAX predictions list: {e}")
        return []

def delete_saved_varmax_prediction(prediction_date):
    """
    저장된 VARMAX 예측을 삭제하는 함수
    """
    try:
        file_path = prediction_state.get('current_file', None)
        if not file_path:
            return False
            
        cache_dirs = get_file_cache_dirs(file_path)
        varmax_dir = cache_dirs['root'] / 'varmax'
        
        # 예측 파일 삭제
        prediction_file = varmax_dir / f"varmax_prediction_{prediction_date}.json"
        if prediction_file.exists():
            prediction_file.unlink()
        
        # 인덱스에서 제거
        index_file = varmax_dir / 'varmax_index.json'
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            
            index['predictions'] = [p for p in index['predictions'] if p['prediction_date'] != prediction_date]
            
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ VARMAX prediction deleted: {prediction_date}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to delete VARMAX prediction: {e}")
        return False

# ✅ 위에서 통합 저장소로 통합됨

def load_prediction_from_unified_storage(prediction_start_date):
    """
    통합 저장소에서 예측 결과 로드 🌟
    """
    try:
        prediction_date = pd.to_datetime(prediction_start_date)
        base_name = f"prediction_start_{prediction_date:%Y%m%d}"
        
        storage_dirs = get_unified_storage_dirs()
        predictions_dir = storage_dirs['predictions']
        
        csv_path = predictions_dir / f"{base_name}.cs"
        meta_path = predictions_dir / f"{base_name}_meta.json"
        attention_path = predictions_dir / f"{base_name}_attention.json"
        ma_path = predictions_dir / f"{base_name}_ma.json"
        
        # 타임스탬프가 붙은 파일이 있는지도 확인
        if not csv_path.exists() or not meta_path.exists():
            # 타임스탬프가 붙은 파일들 찾기
            pattern_csv = f"{base_name}_*.cs"
            pattern_meta = f"{base_name}_*_meta.json"
            
            csv_files = list(predictions_dir.glob(pattern_csv))
            meta_files = list(predictions_dir.glob(pattern_meta))
            
            if csv_files and meta_files:
                # 가장 최근 파일 사용
                csv_path = sorted(csv_files, key=lambda x: x.stat().st_mtime)[-1]
                # meta 파일명에서 타임스탬프 추출
                timestamp = csv_path.stem.split('_')[-1]
                if timestamp.isdigit() and len(timestamp) == 6:
                    meta_path = predictions_dir / f"{base_name}_{timestamp}_meta.json"
                    attention_path = predictions_dir / f"{base_name}_{timestamp}_attention.json"
                    ma_path = predictions_dir / f"{base_name}_{timestamp}_ma.json"
                else:
                    meta_path = sorted(meta_files, key=lambda x: x.stat().st_mtime)[-1]
                    
                logger.info(f"🔍 [UNIFIED_LOAD] Using timestamped files: {csv_path.name}")
        
        if not csv_path.exists() or not meta_path.exists():
            logger.warning(f"⚠️ [UNIFIED_LOAD] Prediction files not found: {base_name}")
            return None
        
        # CSV 로드
        from app.data.loader import load_csv_safe_with_fallback
        predictions_df = load_csv_safe_with_fallback(csv_path)
        
        # 컬럼명 호환성 처리
        if 'date' in predictions_df.columns:
            predictions_df['Date'] = pd.to_datetime(predictions_df['date'])
            predictions_df.drop('date', axis=1, inplace=True)
        elif 'Date' in predictions_df.columns:
            predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
        
        if 'prediction' in predictions_df.columns:
            predictions_df['Prediction'] = predictions_df['prediction']
            predictions_df.drop('prediction', axis=1, inplace=True)
        
        if 'actual' in predictions_df.columns:
            predictions_df['Actual'] = pd.to_numeric(predictions_df['actual'], errors='coerce')
            predictions_df.drop('actual', axis=1, inplace=True)
        
        if 'prediction_from' in predictions_df.columns:
            predictions_df['Prediction_From'] = pd.to_datetime(predictions_df['prediction_from'])
            predictions_df.drop('prediction_from', axis=1, inplace=True)
        elif 'Prediction_From' in predictions_df.columns:
            predictions_df['Prediction_From'] = pd.to_datetime(predictions_df['Prediction_From'])
        
        predictions = predictions_df.to_dict('records')
        
        # JSON 직렬화를 위한 안전 변환
        for pred in predictions:
            for key, value in list(pred.items()):
                if pd.isna(value):
                    pred[key] = None
                elif isinstance(value, pd.Timestamp):
                    pred[key] = value.strftime('%Y-%m-%d')
                elif isinstance(value, (np.int64, np.float64)):
                    pred[key] = float(value)
                elif hasattr(value, 'item'):
                    pred[key] = value.item()
        
        # 메타데이터 로드
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Attention 데이터 로드
        attention_data = None
        if attention_path.exists():
            try:
                with open(attention_path, 'r', encoding='utf-8') as f:
                    stored_attention = json.load(f)
                
                attention_data = {
                    'image': stored_attention.get('image_base64', ''),
                    'file_path': None,
                    'feature_importance': stored_attention.get('feature_importance', {}),
                    'temporal_importance': stored_attention.get('temporal_importance', {})
                }
                logger.info(f"✅ [UNIFIED_LOAD] Attention data loaded")
            except Exception as e:
                logger.warning(f"⚠️ [UNIFIED_LOAD] Failed to load attention data: {str(e)}")
        
        # MA 결과 로드
        ma_results = None
        if ma_path.exists():
            try:
                with open(ma_path, 'r', encoding='utf-8') as f:
                    ma_results = json.load(f)
                logger.info(f"✅ [UNIFIED_LOAD] MA results loaded ({len(ma_results)} windows)")
            except Exception as e:
                logger.warning(f"⚠️ [UNIFIED_LOAD] Failed to load MA results: {str(e)}")
        
        logger.info(f"✅ [UNIFIED_LOAD] Loaded prediction: {len(predictions)} records from unified storage")
        
        return {
            'success': True,
            'predictions': predictions,
            'metadata': metadata,
            'attention_data': attention_data,
            'ma_results': ma_results,
            'storage_system': 'unified_complete',
            'source_files': {
                'csv': str(csv_path),
                'meta': str(meta_path),
                'attention': str(attention_path) if attention_path.exists() else None,
                'ma': str(ma_path) if ma_path.exists() else None
            }
        }
        
    except Exception as e:
        logger.error(f"❌ [UNIFIED_LOAD] Error loading prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def update_unified_predictions_index(metadata):
    """통합 예측 인덱스 업데이트 - app/predictions/predictions_index.cs"""
    try:
        # 통합 예측 디렉토리 확인
        ensure_unified_predictions_dir()
        if metadata is None:
            logger.warning("⚠️ [UNIFIED_INDEX] metadata가 None입니다. 인덱스 업데이트를 건너뜁니다.")
            return False
            
        # 통합 저장소 경로 사용
        storage_dirs = get_unified_storage_dirs()
        unified_index_file = storage_dirs['predictions'] / 'predictions_index.cs'
        
        # 기존 인덱스 읽기
        index_data = []
        if unified_index_file.exists():
            with open(unified_index_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                index_data = list(reader)
        
        # 중복 제거
        prediction_start_date = metadata.get('prediction_start_date')
        if not prediction_start_date:
            logger.warning("⚠️ [UNIFIED_INDEX] metadata에 prediction_start_date가 없습니다.")
            return False
            
        index_data = [row for row in index_data 
                     if row.get('prediction_start_date') != prediction_start_date]
        
        # metrics가 None일 수도 있으므로 안전하게 처리
        metrics = metadata.get('metrics') or {}
        
        # 새 데이터 추가
        new_row = {
            'prediction_start_date': metadata.get('prediction_start_date', ''),
            'data_end_date': metadata.get('data_end_date', ''),
            'created_at': metadata.get('created_at', ''),
            'semimonthly_period': metadata.get('semimonthly_period', ''),
            'next_semimonthly_period': metadata.get('next_semimonthly_period', ''),
            'prediction_count': metadata.get('total_predictions', metadata.get('prediction_count', 0)),
            'f1_score': metrics.get('f1', 0) if isinstance(metrics, dict) else 0,
            'accuracy': metrics.get('accuracy', 0) if isinstance(metrics, dict) else 0,
            'mape': metrics.get('mape', 0) if isinstance(metrics, dict) else 0,
            'weighted_score': metrics.get('weighted_score', 0) if isinstance(metrics, dict) else 0,
            'file_content_hash': metadata.get('file_content_hash', ''),
            'source_file': metadata.get('file_path', ''),
            'storage_system': metadata.get('storage_system', 'unified'),
            'unified_storage_path': metadata.get('unified_storage_path', ''),
            'file_cache_path': metadata.get('file_cache_path', '')
        }
        index_data.append(new_row)
        
        # 날짜순 정렬 후 저장
        index_data.sort(key=lambda x: x.get('data_end_date', ''), reverse=True)
        
        if index_data:
            fieldnames = new_row.keys()
            with open(unified_index_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(index_data)
            
            logger.info(f"✅ [UNIFIED_INDEX] Updated successfully: {len(index_data)} entries")
            logger.info(f"📄 [UNIFIED_INDEX] File: {unified_index_file}")
            return True
        else:
            logger.warning("⚠️ [UNIFIED_INDEX] No data to write to index file")
            return False
        
    except Exception as e:
        logger.error(f"❌ [UNIFIED_INDEX] Error updating unified index: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def get_unified_predictions_list(limit=1000):
    from app.data.loader import load_csv_safe_with_fallback
    """
    통합 인덱스 파일에서 예측 목록을 로드하는 함수 (prediction_start_*.cs와 동일한 xlwings 포함 로직 적용)
    """
    try:
        # 인덱스 파일 경로 (.cs 사용)
        unified_index_file = UNIFIED_PREDICTIONS_DIR / 'predictions_index.cs'
        
        # 파일 존재 확인 및 재생성 시도
        if not unified_index_file.exists():
            logger.warning(f"[UNIFIED_INDEX] Index file not found: {unified_index_file}. Rebuilding...")
            rebuild_unified_predictions_index()  # 재생성
        
        # load_csv_safe_with_fallback으로 .cs 파일 읽기
        logger.info(f"[UNIFIED_INDEX] Loading index file with xlwings fallback: {unified_index_file}")
        predictions_df = load_csv_safe_with_fallback(unified_index_file)
        
        # DataFrame을 딕셔너리 리스트로 변환 (기존 로직과 호환)
        predictions = predictions_df.to_dict('records')
        
        # .cs 읽기 실패 시 레거시 .csv 대체
        if not predictions:
            legacy_index_file = UNIFIED_PREDICTIONS_DIR / 'predictions_index.csv'
            if legacy_index_file.exists():
                logger.info(f"[UNIFIED_INDEX] Falling back to legacy .csv index file")
                predictions_df = load_csv_safe_with_fallback(legacy_index_file)
                predictions = predictions_df.to_dict('records')
        
        # 정렬 및 제한 적용 (최신 순)
        predictions.sort(key=lambda x: x.get('data_end_date', ''), reverse=True)
        return predictions[:limit] if limit else predictions
    
    except Exception as e:
        logger.error(f"❌ [UNIFIED_INDEX] Error loading index: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def rebuild_unified_predictions_index():
    """
    기존 메타파일들로부터 통합 인덱스를 재생성하는 함수 🌟
    """
    try:
        # 통합 예측 디렉토리 확인
        ensure_unified_predictions_dir()
        
        # 기존 메타파일들 수집 (JSON 파일만)
        meta_files = list(UNIFIED_PREDICTIONS_DIR.glob("*_meta.json"))
        logger.info(f"📊 [UNIFIED_REBUILD] Found {len(meta_files)} meta files in unified storage")
        
        if not meta_files:
            logger.warning("[UNIFIED_REBUILD] No meta files found - nothing to rebuild")
            return False
        
        index_data = []
        
        for meta_file in meta_files:
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                metrics = metadata.get('metrics', {})
                if not isinstance(metrics, dict):
                    metrics = {}
                
                new_row = {
                    'prediction_start_date': metadata.get('prediction_start_date', ''),
                    'data_end_date': metadata.get('data_end_date', ''),
                    'semimonthly_period': metadata.get('semimonthly_period', ''),
                    'next_semimonthly_period': metadata.get('next_semimonthly_period', ''),
                    'prediction_count': metadata.get('total_predictions', metadata.get('prediction_count', 0)),
                    'f1_score': metrics.get('f1', 0) if isinstance(metrics, dict) else 0,
                    'accuracy': metrics.get('accuracy', 0) if isinstance(metrics, dict) else 0,
                    'mape': metrics.get('mape', 0) if isinstance(metrics, dict) else 0,
                    'weighted_score': metrics.get('weighted_score', 0) if isinstance(metrics, dict) else 0,
                    'file_content_hash': metadata.get('file_content_hash', ''),
                    'source_file': metadata.get('file_path', ''),
                    'storage_system': metadata.get('storage_system', 'unified'),
                    'unified_storage_path': metadata.get('unified_storage_path', ''),
                    'file_cache_path': metadata.get('file_cache_path', '')
                }
                
                index_data.append(new_row)
                logger.info(f"  ✅ [UNIFIED_REBUILD] {meta_file.name}: {new_row['prediction_start_date']}")
                
            except Exception as e:
                logger.warning(f"  ⚠️ [UNIFIED_REBUILD] Error reading {meta_file.name}: {str(e)}")
                continue
        
        if not index_data:
            logger.error("❌ [UNIFIED_REBUILD] No valid metadata found")
            return False
        
        # 날짜순 정렬
        index_data.sort(key=lambda x: x.get('data_end_date', ''), reverse=True)
        
        # CSV 파일 생성 (확장자를 .cs로 변경)
        unified_index_file = UNIFIED_PREDICTIONS_DIR / 'predictions_index.cs'
        
        fieldnames = index_data[0].keys()
        
        with open(unified_index_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(index_data)
        
        logger.info(f"✅ [UNIFIED_REBUILD] Successfully rebuilt unified index with {len(index_data)} entries")
        logger.info(f"📄 [UNIFIED_REBUILD] Index file: {unified_index_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ [UNIFIED_REBUILD] Error rebuilding unified index: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def load_prediction_from_unified_storage(prediction_start_date):
    """
    통합 예측 저장소에서 예측 결과를 로드하는 함수 🌟
    
    Parameters:
    -----------
    prediction_start_date : str or datetime
        로드할 예측 시작 날짜
        
    Returns:
    --------
    dict : 로드 결과
    """
    try:
        # 통합 예측 디렉토리 확인
        ensure_unified_predictions_dir()
        
        if isinstance(prediction_start_date, str):
            start_date = pd.to_datetime(prediction_start_date)
        else:
            start_date = prediction_start_date
        
        date_str = start_date.strftime('%Y%m%d')
        
        # 통합 폴더의 파일 경로들 (보안을 위해 .cs 확장자 사용)
        unified_csv_path = UNIFIED_PREDICTIONS_DIR / f"prediction_start_{date_str}.cs"
        unified_meta_path = UNIFIED_PREDICTIONS_DIR / f"prediction_start_{date_str}_meta.json"
        unified_attention_path = UNIFIED_PREDICTIONS_DIR / f"prediction_start_{date_str}_attention.json"
        unified_ma_path = UNIFIED_PREDICTIONS_DIR / f"prediction_start_{date_str}_ma.json"
        
        logger.info(f"🌟 [UNIFIED_LOAD] Loading from unified storage:")
        logger.info(f"  📄 CSV: {unified_csv_path.name}")
        logger.info(f"  📄 Meta: {unified_meta_path.name}")
        
        # 필수 파일 존재 확인
        if not unified_csv_path.exists() or not unified_meta_path.exists():
            logger.info(f"❌ [UNIFIED_LOAD] Required files not found in unified storage")
            return {'success': False, 'error': f'Prediction files not found for start date {start_date.strftime("%Y-%m-%d")}'}
        
        # CSV 파일 읽기 - 안전한 fallback 사용
        from app.data.loader import load_csv_safe_with_fallback
        predictions_df = load_csv_safe_with_fallback(unified_csv_path)
        
        # 컬럼명 호환성 처리 및 중복 제거
        if 'date' in predictions_df.columns:
            predictions_df['Date'] = pd.to_datetime(predictions_df['date'])
            predictions_df.drop('date', axis=1, inplace=True)  # 원본 소문자 컬럼 제거
        elif 'Date' in predictions_df.columns:
            predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
        
        if 'prediction' in predictions_df.columns:
            predictions_df['Prediction'] = predictions_df['prediction']
            predictions_df.drop('prediction', axis=1, inplace=True)  # 원본 소문자 컬럼 제거
        
        if 'prediction_from' in predictions_df.columns:
            predictions_df['Prediction_From'] = pd.to_datetime(predictions_df['prediction_from'])
            predictions_df.drop('prediction_from', axis=1, inplace=True)  # 원본 소문자 컬럼 제거
        elif 'Prediction_From' in predictions_df.columns:
            predictions_df['Prediction_From'] = pd.to_datetime(predictions_df['Prediction_From'])
        
        # actual 컬럼도 호환성 처리
        if 'actual' in predictions_df.columns:
            # actual 값이 숫자인지 확인하고 변환
            predictions_df['Actual'] = pd.to_numeric(predictions_df['actual'], errors='coerce')
            predictions_df.drop('actual', axis=1, inplace=True)  # 원본 소문자 컬럼 제거
        
        logger.info(f"📊 [UNIFIED_LOAD] DataFrame processed: {predictions_df.shape}")
        logger.info(f"📋 [UNIFIED_LOAD] Final columns: {list(predictions_df.columns)}")
        
        predictions = predictions_df.to_dict('records')
        
        # JSON 직렬화를 위한 안전한 변환
        for pred in predictions:
            for key, value in list(pred.items()):
                if pd.isna(value):
                    pred[key] = None
                elif isinstance(value, pd.Timestamp):
                    pred[key] = value.strftime('%Y-%m-%d')
                elif isinstance(value, (np.int64, np.float64)):
                    pred[key] = float(value)
                elif hasattr(value, 'item'):
                    pred[key] = value.item()
        
        # 메타데이터 읽기
        with open(unified_meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Attention 데이터 읽기 (있는 경우)
        attention_data = None
        if unified_attention_path.exists():
            try:
                with open(unified_attention_path, 'r', encoding='utf-8') as f:
                    stored_attention = json.load(f)
                
                attention_data = {
                    'image': stored_attention.get('image_base64', ''),
                    'file_path': None,  # 이미지는 base64로 저장됨
                    'feature_importance': stored_attention.get('feature_importance', {}),
                    'temporal_importance': stored_attention.get('temporal_importance', {})
                }
                logger.info(f"  🧠 Attention data loaded from unified storage")
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to load attention data: {str(e)}")
                attention_data = None

        # MA 결과 읽기 (있는 경우)
        ma_results = None
        if unified_ma_path.exists():
            try:
                with open(unified_ma_path, 'r', encoding='utf-8') as f:
                    ma_results = json.load(f)
                logger.info(f"  📊 MA results loaded from unified storage ({len(ma_results)} windows)")
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to load MA results: {str(e)}")
                ma_results = None
        
        logger.info(f"✅ [UNIFIED_LOAD] Complete unified prediction data loaded: {len(predictions)} predictions")
        
        return {
            'success': True,
            'predictions': predictions,
            'metadata': metadata,
            'attention_data': attention_data,
            'ma_results': ma_results,
            'prediction_start_date': start_date.strftime('%Y-%m-%d'),
            'data_end_date': metadata.get('data_end_date'),
            'semimonthly_period': metadata.get('semimonthly_period'),
            'next_semimonthly_period': metadata.get('next_semimonthly_period'),
            'metrics': metadata.get('metrics'),
            'interval_scores': metadata.get('interval_scores'),
            'selected_features': metadata.get('selected_features'),
            'has_cached_attention': attention_data is not None,
            'has_cached_ma': ma_results is not None,
            'storage_system': metadata.get('storage_system', 'unified'),
            'loaded_from': 'unified_storage'
        }
        
    except Exception as e:
        logger.error(f"❌ [UNIFIED_LOAD] Error loading from unified storage: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'error': str(e),
            'loaded_from': 'unified_storage_failed'
        }

def load_accumulated_predictions_from_csv(start_date, end_date=None, limit=None, file_path=None):
    """
    통합 predictions 폴더에서 누적 예측 결과를 로드하는 개선된 함수 🌟
    
    Parameters:
    -----------
    start_date : str or datetime
        시작 날짜 (데이터 기준일)
    end_date : str or datetime, optional
        종료 날짜 (데이터 기준일)
    limit : int, optional
        최대 로드할 예측 개수
    file_path : str, optional
        현재 파일 경로 (호환성을 위해 유지하지만 통합 저장소에서는 무시)
    
    Returns:
    --------
    list : 누적 예측 결과 리스트
    """
    try:
        # 통합 예측 디렉토리 확인
        ensure_unified_predictions_dir()
        logger.info(f"🌟 [UNIFIED_ACCUMULATED] Loading predictions from {start_date} to {end_date or 'latest'}")
        
        # 날짜 형식 통일
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if end_date and isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # 🌟 통합 예측 목록 조회 (Primary)
        all_predictions = get_unified_predictions_list(limit=1000)
        logger.info(f"🌟 [UNIFIED_ACCUMULATED] Found {len(all_predictions)} predictions in unified storage")
        
        # 🎯 폴백: 기존 파일별 캐시에서도 검색 (Secondary - 호환성)
        if len(all_predictions) == 0 and file_path:
            logger.info(f"🔄 [FALLBACK_ACCUMULATED] No unified predictions, trying file-specific cache...")
            try:
                # 기존 방식으로 폴백
                all_predictions = get_saved_predictions_list_for_file(file_path, limit=1000)
                logger.info(f"🎯 [FALLBACK_ACCUMULATED] Found {len(all_predictions)} predictions in file cache")
            except Exception as e:
                logger.warning(f"⚠️ [FALLBACK_ACCUMULATED] Error in file-specific search: {str(e)}")
                return []
        
        # 날짜 범위 필터링 (데이터 기준일 기준)
        filtered_predictions = []
        for pred_info in all_predictions:
            # 인덱스에서 데이터 기준일 확인
            data_end_date = pd.to_datetime(pred_info.get('data_end_date', pred_info.get('prediction_date')))
            
            # 날짜 범위 확인
            if data_end_date >= start_date:
                if end_date is None or data_end_date <= end_date:
                    filtered_predictions.append(pred_info)
            
            # 제한 개수 확인
            if limit and len(filtered_predictions) >= limit:
                break
        
        logger.info(f"📋 [UNIFIED_ACCUMULATED] Found {len(filtered_predictions)} matching prediction files in date range")
        if len(filtered_predictions) > 0:
            logger.info(f"📅 [UNIFIED_ACCUMULATED] Available cached dates:")
            for pred in filtered_predictions:
                data_end_date = pred.get('data_end_date', pred.get('prediction_date'))
                logger.info(f"    - {data_end_date}")
        
        # 각 예측 결과 로드
        accumulated_results = []
        for i, pred_info in enumerate(filtered_predictions):
            try:
                # 데이터 기준일을 사용하여 예측 시작일 계산
                data_end_date = pd.to_datetime(pred_info.get('data_end_date', pred_info.get('prediction_date')))
                
                # 데이터 기준일로부터 예측 시작일 계산
                prediction_start_date = data_end_date + pd.Timedelta(days=1)
                while prediction_start_date.weekday() >= 5:  # 주말 스킵
                    prediction_start_date += pd.Timedelta(days=1)
                
                # 🌟 통합 저장소에서 로드 시도 (Primary)
                loaded_result = load_prediction_from_unified_storage(prediction_start_date)
                
                # 🎯 폴백: 파일별 캐시에서 로드 (Secondary)
                if not loaded_result.get('success') and file_path:
                    cache_dirs = get_file_cache_dirs(file_path)
                    loaded_result = load_prediction_with_attention_from_csv_in_dir(prediction_start_date, cache_dirs['predictions'])
                
                if loaded_result.get('success'):
                    logger.info(f"  ✅ [UNIFIED_ACCUMULATED] Successfully loaded cached prediction for {data_end_date.strftime('%Y-%m-%d')}")
                    
                    # 누적 예측 형식에 맞게 변환
                    predictions = loaded_result.get('predictions', [])
                    
                    # 예측 데이터가 중첩된 딕셔너리 구조인 경우 처리
                    if isinstance(predictions, dict):
                        if 'future' in predictions:
                            predictions = predictions['future']
                        elif 'predictions' in predictions:
                            predictions = predictions['predictions']
                    
                    if not isinstance(predictions, list):
                        logger.warning(f"Loaded predictions is not a list for {data_end_date.strftime('%Y-%m-%d')}: {type(predictions)}")
                        predictions = []
                    
                    metadata = loaded_result.get('metadata', {})
                    if not isinstance(metadata, dict):
                        metadata = {}
                    
                    # 🔧 metrics 안전성 처리: None이면 기본값 설정
                    cached_metrics = metadata.get('metrics')
                    if not cached_metrics or not isinstance(cached_metrics, dict):
                        cached_metrics = {'f1': 0.0, 'accuracy': 0.0, 'mape': 0.0, 'weighted_score': 0.0}
                    
                    accumulated_item = {
                        'date': data_end_date.strftime('%Y-%m-%d'),  # 데이터 기준일
                        'prediction_start_date': loaded_result.get('prediction_start_date'),  # 예측 시작일
                        'predictions': predictions,
                        'metrics': cached_metrics,
                        'interval_scores': metadata.get('interval_scores', {}),
                        'next_semimonthly_period': metadata.get('next_semimonthly_period'),
                        'actual_business_days': metadata.get('actual_business_days'),
                        'original_interval_scores': metadata.get('interval_scores', {}),
                        'has_attention': loaded_result.get('has_cached_attention', False),
                        'storage_system': metadata.get('storage_system', 'unified'),
                        'loaded_from': loaded_result.get('loaded_from', 'unified_storage')
                    }
                    accumulated_results.append(accumulated_item)
                    logger.info(f"  ✅ [UNIFIED_ACCUMULATED] Added to results {i+1}/{len(filtered_predictions)}: {data_end_date.strftime('%Y-%m-%d')}")
                else:
                    logger.warning(f"  ❌ [UNIFIED_ACCUMULATED] Failed to load prediction {i+1}/{len(filtered_predictions)}: {loaded_result.get('error')}")
                    
            except Exception as e:
                logger.error(f"  ❌ Error loading prediction {i+1}/{len(filtered_predictions)}: {str(e)}")
                continue
        
        logger.info(f"🎯 [UNIFIED_ACCUMULATED] Successfully loaded {len(accumulated_results)} predictions from unified storage")
        return accumulated_results
        
    except Exception as e:
        logger.error(f"❌ [UNIFIED_ACCUMULATED] Error loading accumulated predictions: {str(e)}")
        logger.error(traceback.format_exc())
        return []

# 🌟 완전 통합 저장소 시스템
UNIFIED_PREDICTIONS_DIR = Path("app/predictions")
UNIFIED_HYPERPARAMETERS_DIR = Path("app/hyperparameters") 
UNIFIED_PLOTS_DIR = Path("app/plots")

# 통합 저장소 디렉토리들 초기화
def ensure_unified_storage_dirs():
    """모든 통합 저장소 디렉토리가 존재하는지 확인하고 생성"""
    try:
        dirs_to_create = [
            UNIFIED_PREDICTIONS_DIR,
            UNIFIED_HYPERPARAMETERS_DIR, 
            UNIFIED_PLOTS_DIR,
            UNIFIED_PLOTS_DIR / 'attention',
            UNIFIED_PLOTS_DIR / 'ma_plots'
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"✅ Unified directory ensured: {dir_path}")
        
        logger.info(f"🌟 All unified storage directories initialized")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create unified storage directories: {e}")
        return False

def get_unified_storage_dirs():
    """
    통합 저장소 디렉토리 구조를 반환하는 함수 🌟
    🔧 모든 예측 관련 산출물을 통합 관리
    """
    try:
        # 통합 저장소 디렉토리들 확인
        ensure_unified_storage_dirs()
        
        dirs = {
            'root': Path("app"),  # 🔧 VARMAX 호환성을 위한 root 디렉토리
            'predictions': UNIFIED_PREDICTIONS_DIR,
            'models': UNIFIED_HYPERPARAMETERS_DIR,  # 호환성을 위해 'models' 키 유지
            'hyperparameters': UNIFIED_HYPERPARAMETERS_DIR,  # 새로운 명시적 키
            'plots': UNIFIED_PLOTS_DIR,
            'attention_plots': UNIFIED_PLOTS_DIR / 'attention',
            'ma_plots': UNIFIED_PLOTS_DIR / 'ma_plots',
            'accumulated': UNIFIED_PREDICTIONS_DIR / 'accumulated'  # 누적 예측용
        }
        
        logger.debug(f"🌟 [UNIFIED_STORAGE] Using unified storage system")
        return dirs
        
    except Exception as e:
        logger.error(f"❌ Error in get_unified_storage_dirs: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

# 모듈 로드 시 통합 디렉토리 생성
ensure_unified_storage_dirs()

def ensure_unified_predictions_dir():
    """통합 예측 저장소 디렉토리가 존재하는지 확인하고 생성"""
    try:
        UNIFIED_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ [UNIFIED_DIR] Ensured unified predictions directory: {UNIFIED_PREDICTIONS_DIR}")
        return True
    except Exception as e:
        logger.error(f"❌ [UNIFIED_DIR] Failed to create unified predictions directory: {str(e)}")
        return False

# Cache directory exports  
CACHE_BASE_DIR = Path("app/cache")