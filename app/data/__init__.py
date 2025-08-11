"""
app/data 패키지
데이터 로딩, 전처리, 캐시 관리를 담당하는 모듈
"""

import logging
from pathlib import Path

# 로깅 설정
logger = logging.getLogger('app.data')

# xlwings 사용 가능 여부를 먼저 확인
from .loader import XLWINGS_AVAILABLE

# loader 모듈에서 주요 함수들 가져오기
from .loader import (
    load_data,
    load_data_safe,
    load_data_with_xlwings,
    load_holidays_from_file,
    safe_read_excel,
    load_csv_safe_with_fallback,
    variable_groups as loader_variable_groups,  # 이름 충돌 방지
    # 보안 확장자 처리
    detect_file_type_by_content,
    normalize_security_extension,
    # CSV 캐시 시스템
    get_csv_cache_path,
    create_csv_cache_metadata,
    is_csv_cache_valid,
    create_csv_cache_from_excel,
    load_csv_cache,
    load_excel_as_dataframe,
    # CSV 기반 비교
    convert_excel_to_temp_csv,
    compare_csv_files,
    check_data_extension_csv_based,
    check_dataframes_extension
)

# preprocessor 모듈에서 주요 함수들 가져오기
from .preprocessor import (
    process_excel_data_complete,
    # is_holiday,  # 이 줄을 제거 (is_holiday는 preprocessor에 없음)
    update_holidays,
    update_holidays_safe,
    get_combined_holidays,
    detect_missing_weekdays_as_holidays,
    variable_groups,
    prepare_data,
    select_features_from_groups,
    create_proper_column_names,
    clean_text_values_advanced,
    fill_missing_values_advanced,
    rename_columns_to_standard
)

# is_holiday는 holidays 모듈에서 가져오기
from app.utils.holidays import is_holiday

# cache_manager 모듈에서 주요 함수들 가져오기
from .cache_manager import (
    # 🌟 통합 저장소 시스템
    get_unified_storage_dirs,
    get_file_cache_dirs,  # 이제 통합 저장소 반환
    get_data_content_hash,
    save_prediction_simple,
    load_prediction_simple,
    load_prediction_with_attention_from_csv,
    load_prediction_from_unified_storage,
    get_saved_predictions_list,
    get_saved_predictions_list_for_file,
    get_unified_predictions_list,
    load_accumulated_predictions_from_csv,
    delete_saved_prediction,
    update_predictions_index_simple,
    update_unified_predictions_index,
    rebuild_predictions_index_from_existing_files,
    rebuild_unified_predictions_index,
    find_compatible_cache_file,
    check_data_extension,
    update_cached_prediction_actual_values,
    # VARMAX 관련
    save_varmax_prediction,
    load_varmax_prediction,
    get_saved_varmax_predictions_list,
    delete_saved_varmax_prediction,
    # 통합 예측 디렉토리 관리
    ensure_unified_predictions_dir
)

# 모듈 초기화 함수
def initialize_data_module():
    """데이터 모듈 초기화"""
    logger.info("🚀 Initializing app.data module...")
    
    # xlwings 상태 로깅
    if XLWINGS_AVAILABLE:
        logger.info("✅ xlwings is available - DRM bypass and security workarounds enabled")
    else:
        logger.warning("⚠️ xlwings is not available - some security features will be limited")
    
    # 기본 휴일 로드
    try:
        initial_holidays = load_holidays_from_file()
        logger.info(f"📅 Loaded {len(initial_holidays)} default holidays")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load default holidays: {str(e)}")
    
    # CSV 캐시 시스템 초기화 로그
    logger.info("📊 CSV cache system enabled for Excel file processing")
    logger.info("🔒 Security extension normalization (.xl, .cs, .log support)")
    logger.info("🚀 Enhanced Excel loading with xlwings + pandas fallback")
    
    logger.info("✅ app.data module initialized successfully")

# 모듈 로드 시 초기화
initialize_data_module()

# 버전 정보
__version__ = '1.0.0'

# 공개 API
__all__ = [
    # 데이터 로딩
    'load_data',
    'load_data_safe',
    'load_data_with_xlwings',
    'safe_read_excel',
    'load_csv_safe_with_fallback',
    'load_holidays_from_file',
    
    # 보안 확장자 처리
    'detect_file_type_by_content',
    'normalize_security_extension',
    
    # CSV 캐시 시스템
    'get_csv_cache_path',
    'create_csv_cache_metadata',
    'is_csv_cache_valid',
    'create_csv_cache_from_excel',
    'load_csv_cache',
    'load_excel_as_dataframe',
    
    # CSV 기반 비교 및 확장 검사
    'convert_excel_to_temp_csv',
    'compare_csv_files',
    'check_data_extension_csv_based',
    'check_dataframes_extension',
    
    # 데이터 전처리
    'process_excel_data_complete',
    'prepare_data',
    'select_features_from_groups',
    'create_proper_column_names',
    'clean_text_values_advanced',
    'fill_missing_values_advanced',
    'rename_columns_to_standard',
    
    # 휴일 관리
    'holidays',
    'is_holiday',
    'update_holidays',
    'update_holidays_safe',
    'get_combined_holidays',
    'detect_missing_weekdays_as_holidays',
    
    # 캐시 관리 (통합 저장소 시스템)
    'get_unified_storage_dirs',
    'get_file_cache_dirs',
    'get_data_content_hash',
    'save_prediction_simple',
    'load_prediction_simple',
    'load_prediction_with_attention_from_csv',
    'load_prediction_from_unified_storage',
    'get_saved_predictions_list',
    'get_saved_predictions_list_for_file',
    'get_unified_predictions_list',
    'load_accumulated_predictions_from_csv',
    'delete_saved_prediction',
    'update_predictions_index_simple',
    'update_unified_predictions_index',
    'rebuild_predictions_index_from_existing_files',
    'rebuild_unified_predictions_index',
    'find_compatible_cache_file',
    'check_data_extension',
    'update_cached_prediction_actual_values',
    
    # VARMAX 캐시
    'save_varmax_prediction',
    'load_varmax_prediction',
    'get_saved_varmax_predictions_list',
    'delete_saved_varmax_prediction',
    
    # 통합 예측 디렉토리 관리
    'ensure_unified_predictions_dir',
    
    # 상수 및 설정
    'variable_groups',
    'XLWINGS_AVAILABLE'
]

# 편의를 위한 함수 그룹화 (통합 저장소 시스템)
cache_functions = {
    'save': save_prediction_simple,
    'load': load_prediction_with_attention_from_csv,
    'load_unified': load_prediction_from_unified_storage,
    'list': get_saved_predictions_list,
    'list_unified': get_unified_predictions_list,
    'delete': delete_saved_prediction,
    'find_compatible': find_compatible_cache_file,
    'get_dirs': get_unified_storage_dirs
}

data_functions = {
    'load': load_data,
    'load_safe': load_data_safe,
    'process_excel': process_excel_data_complete,
    'read_excel': safe_read_excel
}

# CSV 캐시 관련 편의 함수
csv_cache_functions = {
    'is_valid': is_csv_cache_valid,
    'create_from_excel': create_csv_cache_from_excel,
    'load': load_csv_cache,
    'get_path': get_csv_cache_path,
    'compare_files': compare_csv_files,
    'check_extension': check_data_extension_csv_based
}

# 보안 및 파일 처리 함수
security_functions = {
    'detect_type': detect_file_type_by_content,
    'normalize_extension': normalize_security_extension,
    'convert_to_temp_csv': convert_excel_to_temp_csv
}

holiday_functions = {
    'is_holiday': is_holiday,
    'update': update_holidays,
    'update_safe': update_holidays_safe,
    'get_combined': get_combined_holidays,
    'load_from_file': load_holidays_from_file
}

logger.info(f"📦 app.data package loaded with {len(__all__)} public APIs")
