from flask import request, jsonify, send_file, make_response, Blueprint
from flask_cors import cross_origin
import os
import json
import pandas as pd
from datetime import datetime, timedelta
import io
import base64
import tempfile
import csv
import logging                          
import traceback
import shutil
import torch
from werkzeug.utils import secure_filename
import time
from pathlib import Path
import numpy as np
from threading import Thread
from app.config import PREDICTIONS_DIR

# 최상위 Flask 앱 객체 import
from app import app

# 전역 상태 변수 import
from app.core.state_manager import prediction_state

# 다른 모듈에서 필요한 함수와 변수들 import
from app.data.loader import (
    load_data, load_data_safe, load_csv_safe_with_fallback,
    # 🚀 새로운 CSV 캐시 시스템
    normalize_security_extension, detect_file_type_by_content,
    is_csv_cache_valid, create_csv_cache_from_excel, load_csv_cache,
    check_data_extension_csv_based, compare_csv_files
)
from app.data.preprocessor import variable_groups, update_holidays_safe, get_combined_holidays, select_features_from_groups, load_holidays_from_file, holidays
from app.utils.date_utils import is_holiday, get_semimonthly_period, format_date
from app.data.cache_manager import get_file_cache_dirs, get_saved_predictions_list, load_prediction_from_csv, delete_saved_prediction, update_cached_prediction_actual_values, load_accumulated_predictions_from_csv, rebuild_predictions_index_from_existing_files, find_compatible_cache_file, find_existing_cache_range, check_data_extension, check_existing_prediction, get_data_content_hash, calculate_file_hash, save_varmax_prediction, load_varmax_prediction, get_saved_varmax_predictions_list, delete_saved_varmax_prediction
from app.prediction.metrics import calculate_accumulated_purchase_reliability, calculate_prediction_consistency, calculate_moving_averages_with_history # compute_performance_metrics_improved는 plot_prediction_basic에서 사용되므로 visualization.plotter를 통해 임포트
from app.prediction.predictor import generate_predictions_compatible # generate_predictions_with_save는 background_tasks에서 호출됨
from app.visualization.plotter import plot_prediction_basic, plot_moving_average_analysis, visualize_accumulated_metrics # plot_varmax_prediction_basic, plot_varmax_moving_average_analysis는 varmax_model에서 사용
from app.visualization.attention_viz import visualize_attention_weights
from app.utils.date_utils import get_semimonthly_period
from app.utils.file_utils import process_security_file, cleanup_excel_processes
# detect_file_type_by_content, normalize_security_extension은 이제 loader에서 import
from app.utils.serialization import safe_serialize_value, clean_interval_scores_safe, convert_to_legacy_format, clean_predictions_data # clean_predictions_data, clean_cached_predictions는 cache_manager에서 사용
from app.core.gpu_manager import compare_gpu_monitoring_methods # get_gpu_info는 gpu_manager에 있음
from app.models.varmax_model import VARMAXSemiMonthlyForecaster # varmax_decision에서 사용

logger = logging.getLogger(__name__)

# 파일 해시 캐시 추가 (메모리 캐싱으로 성능 최적화)
_file_hash_cache = {}
_cache_lookup_index = {}  # 빠른 캐시 검색을 위한 인덱스

# 🚀 중복 함수 제거: cache_manager에서 import하여 사용

# 🔧 DataFrame 메모리 캐시 (중복 로딩 방지)
_dataframe_cache = {}
_cache_expiry_seconds = 120  # 2분간 캐시 유지

# 🚀 중복 함수 제거: cache_manager에서 import하여 사용

@app.route('/api/health', methods=['GET'])
def health_check():
    """서버 상태 확인 API"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'attention_endpoint_available': True
    })

@app.route('/api/test-attention', methods=['GET'])
def test_attention():
    """어텐션 맵 엔드포인트 테스트용"""
    return jsonify({
        'success': True,
        'message': 'Test attention endpoint is working',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/test', methods=['GET'])
def test_api():
    """API 연결 테스트"""
    return jsonify({
        'status': 'ok',
        'message': 'API is working!',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/test/cache-dirs', methods=['GET'])
def test_cache_dirs():
    """캐시 디렉토리 시스템 테스트"""
    try:
        # 현재 상태 확인
        current_file = prediction_state.get('current_file', None)
        
        # 파일 경로가 있으면 해당 파일로, 없으면 기본으로 테스트
        test_file = request.args.get('file_path', current_file)
        
        if test_file and not os.path.exists(test_file):
            return jsonify({
                'error': f'File does not exist: {test_file}',
                'current_file': current_file
            }), 400
        
        # 캐시 디렉토리 생성 테스트
        cache_dirs = get_file_cache_dirs(test_file)
        
        # 디렉토리 존재 여부 확인
        dir_status = {}
        for name, path in cache_dirs.items():
            dir_status[name] = {
                'path': str(path),
                'exists': path.exists(),
                'is_dir': path.is_dir() if path.exists() else False
            }
        
        return jsonify({
            'success': True,
            'test_file': test_file,
            'current_file': current_file,
            'cache_dirs': dir_status,
            'cache_root_exists': Path(CACHE_ROOT_DIR).exists()
        })
        
    except Exception as e:
        logger.error(f"Cache directory test failed: {str(e)}")
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

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

def normalize_security_extension(filename):
    """
    회사 보안정책으로 변경된 확장자를 원래 확장자로 복원
    
    Args:
        filename (str): 원본 파일명
    
    Returns:
        tuple: (정규화된 파일명, 원본 확장자, 보안 확장자인지 여부)
    """
    # 보안 확장자 매핑
    security_extensions = {
        '.cs': '.csv',     # csv -> cs
        '.xl': '.xlsx',    # xlsx -> xl  
        '.xls': '.xlsx',   # 기존 xls도 xlsx로 통일
        '.log': '.xlsx',   # log -> xlsx (보안 정책으로 Excel 파일을 log로 위장)
        '.dat': None,      # 내용 분석 필요
        '.txt': None,      # 내용 분석 필요
    }
    
    filename_lower = filename.lower()
    original_ext = os.path.splitext(filename_lower)[1]
    
    # 보안 확장자인지 확인
    if original_ext in security_extensions:
        if security_extensions[original_ext]:
            # 직접 매핑이 있는 경우
            normalized_ext = security_extensions[original_ext]
            base_name = os.path.splitext(filename)[0]
            normalized_filename = f"{base_name}{normalized_ext}"
            
            logger.info(f"🔒 [SECURITY] Extension normalization: {filename} -> {normalized_filename}")
            return normalized_filename, normalized_ext, True
        else:
            # 내용 분석이 필요한 경우
            return filename, original_ext, True
    
    # 일반 확장자인 경우
    return filename, original_ext, False

def process_security_file(temp_filepath, original_filename):
    """
    보안 정책으로 확장자가 변경된 파일을 처리
    
    Args:
        temp_filepath (str): 임시 파일 경로
        original_filename (str): 원본 파일명
    
    Returns:
        tuple: (처리된 파일 경로, 정규화된 파일명, 실제 확장자)
    """
    # 확장자 정규화
    normalized_filename, detected_ext, is_security_ext = normalize_security_extension(original_filename)
    
    if is_security_ext:
        logger.info(f"🔒 [SECURITY] Processing security file: {original_filename}")
        
        # 파일 내용으로 실제 타입 감지
        if detected_ext is None or detected_ext in ['.dat', '.txt']:
            content_type = detect_file_type_by_content(temp_filepath)
            if content_type:
                detected_ext = f'.{content_type}'
                base_name = os.path.splitext(normalized_filename)[0]
                normalized_filename = f"{base_name}{detected_ext}"
                logger.info(f"📊 [CONTENT_DETECTION] Detected file type: {content_type}")
        
        # 새로운 파일 경로 생성
        new_filepath = temp_filepath.replace(os.path.splitext(temp_filepath)[1], detected_ext)
        
        # 파일 이름 변경 (확장자 수정)
        if new_filepath != temp_filepath:
            try:
                shutil.move(temp_filepath, new_filepath)
                logger.info(f"📝 [SECURITY] File extension corrected: {os.path.basename(temp_filepath)} -> {os.path.basename(new_filepath)}")
                return new_filepath, normalized_filename, detected_ext
            except Exception as e:
                logger.warning(f"⚠️ [SECURITY] Failed to rename file: {str(e)}")
                return temp_filepath, normalized_filename, detected_ext
    
    return temp_filepath, normalized_filename, detected_ext

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """스마트 캐시 기능이 있는 데이터 파일 업로드 API (CSV, Excel 지원)"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # 🔒 보안 확장자 정규화 처리
    normalized_filename, normalized_ext, is_security_file = normalize_security_extension(file.filename)
    
    # 지원되는 파일 형식 확인 (보안 확장자 포함)
    allowed_extensions = ['.csv', '.xlsx', '.xls']
    security_extensions = ['.cs', '.xl', '.log', '.dat', '.txt']  # 보안 확장자 추가
    
    file_ext = os.path.splitext(file.filename.lower())[1]
    
    if file and (file_ext in allowed_extensions or file_ext in security_extensions):
        try:
            # 임시 파일명 생성 (원본 확장자 유지)
            original_filename = secure_filename(file.filename)
            temp_filename = secure_filename(f"temp_{int(time.time())}{file_ext}")
            temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            
            # 임시 파일로 저장
            file.save(temp_filepath)
            logger.info(f"📤 [UPLOAD] File saved temporarily: {temp_filename}")
            
            # 🔒 1단계: 보안 파일 처리 (확장자 복원) - 캐시 비교 전에 먼저 처리
            if is_security_file:
                temp_filepath, normalized_filename, actual_ext = process_security_file(temp_filepath, original_filename)
                file_ext = actual_ext  # 실제 확장자로 업데이트
                logger.info(f"🔒 [SECURITY] File processed: {original_filename} -> {normalized_filename}")
                
                # 처리된 파일이 지원되는 형식인지 재확인
                if file_ext not in allowed_extensions:
                    try:
                        os.remove(temp_filepath)
                    except:
                        pass
                    return jsonify({'error': f'보안 파일 처리 후 지원되지 않는 형식입니다: {file_ext}'}), 400
            
            # 📊 2단계: 데이터 분석 - 날짜 범위 확인 (보안 처리 완료된 파일로)
            # 🔧 데이터 로딩 캐싱을 위한 변수 초기화
            df_analysis = None
            
            try:
                if file_ext == '.csv':
                    df_analysis = pd.read_csv(temp_filepath)
                else:  # Excel 파일
                    # Excel 파일은 load_data 함수를 사용하여 고급 처리 (🔧 캐시 활성화)
                    logger.info(f"🔍 [UPLOAD] Starting data analysis for {temp_filename}")
                    df_analysis = load_data(temp_filepath, use_cache=True)
                    # 인덱스가 Date인 경우 컬럼으로 복원
                    if df_analysis.index.name == 'Date':
                        df_analysis = df_analysis.reset_index()
                if 'Date' in df_analysis.columns:
                    df_analysis['Date'] = pd.to_datetime(df_analysis['Date'])
                    start_date = df_analysis['Date'].min()
                    end_date = df_analysis['Date'].max()
                    total_records = len(df_analysis)
                    
                    # 2022년 이후 데이터 확인
                    cutoff_2022 = pd.to_datetime('2022-01-01')
                    recent_data = df_analysis[df_analysis['Date'] >= cutoff_2022]
                    recent_records = len(recent_data)
                    
                    logger.info(f"📊 [DATA_ANALYSIS] Full range: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({total_records} records)")
                    logger.info(f"📊 [DATA_ANALYSIS] 2022+ range: {recent_records} records")
                    
                    data_info = {
                        'start_date': start_date.strftime('%Y-%m-%d'),
                        'end_date': end_date.strftime('%Y-%m-%d'),
                        'total_records': total_records,
                        'recent_records_2022plus': recent_records,
                        'has_historical_data': start_date < cutoff_2022,
                        'lstm_recommended_cutoff': '2022-01-01'
                    }
                else:
                    # Date 컬럼이 없는 파일의 경우 (예: holidays.csv)
                    file_type_hint = None
                    if 'holiday' in original_filename.lower():
                        file_type_hint = "휴일 파일로 보입니다. /api/holidays/upload 엔드포인트 사용을 권장합니다."
                    data_info = {
                        'warning': 'No Date column found',
                        'file_type_hint': file_type_hint
                    }
            except Exception as e:
                logger.warning(f"Data analysis failed: {str(e)}")
                data_info = {'warning': f'Data analysis failed: {str(e)}'}
            
            # 🔧 Excel 파일 읽기 완료 후 파일 핸들 강제 해제
            import gc
            gc.collect()  # 가비지 컬렉션으로 pandas가 열어둔 파일 핸들 해제
            
            # 🔍 3단계: 캐시 호환성 확인 (보안 처리 및 데이터 분석 완료 후)
            # 사용자의 의도된 데이터 범위 추정 (기본값: 2022년부터 LSTM, 전체 데이터 VARMAX)
            # end_date가 정의되지 않은 경우를 위한 안전한 fallback
            default_end_date = datetime.now().strftime('%Y-%m-%d')
            intended_range = {
                'start_date': '2022-01-01',  # LSTM 권장 시작점
                'cutoff_date': data_info.get('end_date', default_end_date)
            }
            
            logger.info(f"🔍 [UPLOAD_CACHE] Starting cache compatibility check:")
            logger.info(f"  📁 New file: {temp_filename}")
            logger.info(f"  📅 Data range: {data_info.get('start_date')} ~ {data_info.get('end_date')}")
            logger.info(f"  📊 Total records: {data_info.get('total_records')}")
            logger.info(f"  🎯 Intended range: {intended_range}")
            
            # 🔧 이미 로딩된 데이터를 전달하여 중복 로딩 방지
            cache_result = find_compatible_cache_file(temp_filepath, intended_range, cached_df=df_analysis)
            
            logger.info(f"🎯 [UPLOAD_CACHE] Cache check result:")
            logger.info(f"  ✅ Found: {cache_result['found']}")
            logger.info(f"  🏷️ Type: {cache_result.get('cache_type')}")
            if cache_result.get('cache_files'):
                logger.info(f"  📁 Cache files: {[os.path.basename(f) for f in cache_result['cache_files']]}")
            if cache_result.get('compatibility_info'):
                logger.info(f"  ℹ️ Compatibility info: {cache_result['compatibility_info']}")
            
            response_data = {
                'success': True,
                'filepath': temp_filepath,
                'filename': os.path.basename(temp_filepath),
                'original_filename': original_filename,
                'normalized_filename': normalized_filename if is_security_file else original_filename,
                'data_info': data_info,
                'model_recommendations': {
                    'varmax': '전체 데이터 사용 권장 (장기 트렌드 분석)',
                    'lstm': '2022년 이후 데이터 사용 권장 (단기 정확도 향상)'
                },
                'security_info': {
                    'is_security_file': is_security_file,
                    'original_extension': os.path.splitext(file.filename.lower())[1] if is_security_file else None,
                    'detected_extension': file_ext if is_security_file else None,
                    'message': f"보안 파일이 처리되었습니다: {os.path.splitext(file.filename)[1]} -> {file_ext}" if is_security_file else None
                },
                'cache_info': {
                    'found': cache_result['found'],
                    'cache_type': cache_result.get('cache_type'),
                    'message': None
                }
            }
            
            if cache_result['found']:
                cache_type = cache_result['cache_type']
                cache_files = cache_result.get('cache_files', [])
                compatibility_info = cache_result.get('compatibility_info', {})
                
                if cache_type == 'exact':
                    cache_file = cache_files[0] if cache_files else None
                    response_data['cache_info']['message'] = f"동일한 데이터 발견! 기존 캐시를 활용합니다. ({os.path.basename(cache_file) if cache_file else 'Unknown'})"
                    response_data['cache_info']['compatible_file'] = cache_file
                    logger.info(f"✅ [CACHE] Exact match found: {cache_file}")
                    
                elif cache_type == 'extension':
                    cache_file = cache_files[0] if cache_files else None
                    extension_details = compatibility_info.get('extension_details', {})
                    new_rows = extension_details.get('new_rows_count', compatibility_info.get('new_rows_count', 0))
                    extension_type = extension_details.get('validation_details', {}).get('extension_type', ['데이터 확장'])
                    
                    if isinstance(extension_type, list):
                        extension_desc = ' + '.join(extension_type)
                    else:
                        extension_desc = str(extension_type)
                    
                    response_data['cache_info']['message'] = f"📈 데이터 확장 감지! {extension_desc} (+{new_rows}개 새 행). 기존 하이퍼파라미터와 캐시를 재사용할 수 있습니다."
                    response_data['cache_info']['compatible_file'] = cache_file
                    response_data['cache_info']['extension_info'] = compatibility_info
                    response_data['cache_info']['hyperparams_reusable'] = True  # 하이퍼파라미터 재사용 가능 표시
                    logger.info(f"📈 [CACHE] Extension detected from {cache_file}: {extension_desc} (+{new_rows} rows)")
                    
                elif cache_type in ['partial', 'near_complete', 'multi_cache']:
                    best_coverage = compatibility_info.get('best_coverage', 0)
                    total_caches = compatibility_info.get('total_compatible_caches', len(cache_files))
                    
                    if cache_type == 'near_complete':
                        response_data['cache_info']['message'] = f"🎯 거의 완전한 캐시 매치! ({best_coverage:.1%} 커버리지) 기존 예측 결과를 최대한 활용합니다."
                    elif cache_type == 'multi_cache':
                        response_data['cache_info']['message'] = f"🔗 다중 캐시 발견! {total_caches}개 캐시에서 {best_coverage:.1%} 커버리지로 예측을 가속화합니다."
                    else:  # partial
                        response_data['cache_info']['message'] = f"📊 부분 캐시 매치! ({best_coverage:.1%} 커버리지) 일부 예측 결과를 재활용합니다."
                    
                    response_data['cache_info']['compatible_files'] = cache_files
                    response_data['cache_info']['compatibility_info'] = compatibility_info
                    logger.info(f"🎯 [ENHANCED_CACHE] {cache_type} cache found: {total_caches} caches, {best_coverage:.1%} coverage")
                
                # 🔧 파일 처리 로직 개선: 데이터 확장 시 새 파일 사용
                if cache_type == 'exact' and cache_files:
                    # 정확히 동일한 파일인 경우에만 기존 파일 사용
                    cache_file = cache_files[0]
                    response_data['filepath'] = cache_file
                    response_data['filename'] = os.path.basename(cache_file)
                    
                    # 임시 파일 삭제 (완전히 동일한 경우만)
                    if temp_filepath != cache_file:
                        try:
                            os.remove(temp_filepath)
                            logger.info(f"🗑️ [CLEANUP] Temporary file removed (exact match): {temp_filename}")
                        except Exception as e:
                            logger.warning(f"⚠️ [CLEANUP] Failed to remove temp file {temp_filename}: {str(e)}")
                            # 실패해도 계속 진행
                            
                elif cache_type == 'extension' and cache_files:
                    # 🔄 데이터 확장의 경우: 새 파일을 사용하되, 캐시 정보는 유지
                    logger.info(f"📈 [EXTENSION] Data extension detected - using NEW file with cache info")
                    
                    # 새 파일을 정식 파일명으로 저장 (원본 확장자 유지)
                    try:
                        content_hash = get_data_content_hash(temp_filepath)
                        final_filename = f"data_{content_hash}{file_ext}" if content_hash else temp_filename
                    except Exception as hash_error:
                        logger.warning(f"⚠️ Hash calculation failed for extended file, using timestamp-based filename: {str(hash_error)}")
                        final_filename = temp_filename  # 해시 실패 시 임시 파일명 유지
                    
                    final_filepath = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)
                    
                    if temp_filepath != final_filepath:
                        # 🔧 강화된 파일 이동 로직 (Excel 파일 락 해제 대기)
                        moved_successfully = False
                        for attempt in range(3):  # 최대 3번 시도
                            try:
                                # Excel 파일 읽기 후 파일 락 해제를 위한 충분한 대기
                                import gc
                                gc.collect()  # 가비지 컬렉션으로 파일 핸들 해제
                                time.sleep(0.5 + attempt * 0.5)  # 점진적으로 대기 시간 증가
                                
                                shutil.move(temp_filepath, final_filepath)
                                logger.info(f"📝 [UPLOAD] Extended file renamed: {final_filename} (attempt {attempt + 1})")
                                moved_successfully = True
                                break
                            except OSError as move_error:
                                logger.warning(f"⚠️ Extended file move attempt {attempt + 1} failed: {str(move_error)}")
                                if attempt == 2:  # 마지막 시도
                                    logger.warning(f"⚠️ All move attempts failed, keeping original filename: {str(move_error)}")
                                    final_filepath = temp_filepath
                                    final_filename = temp_filename
                        
                        if not moved_successfully:
                            final_filepath = temp_filepath
                            final_filename = temp_filename
                    else:
                        logger.info(f"📝 [UPLOAD] Extended file already has correct name: {final_filename}")
                        
                    response_data['filepath'] = final_filepath
                    response_data['filename'] = final_filename
                    
                    # 확장 정보에 새 파일 정보 추가
                    response_data['cache_info']['new_file_used'] = True
                    response_data['cache_info']['original_cache_file'] = cache_files[0]
                    
                    # 🔑 데이터 확장 표시 - 하이퍼파라미터 재사용 가능
                    response_data['data_extended'] = True
                    response_data['hyperparams_inheritance'] = {
                        'available': True,
                        'source_file': os.path.basename(cache_files[0]),
                        'extension_type': extension_desc if 'extension_desc' in locals() else '데이터 확장',
                        'new_rows_added': new_rows if 'new_rows' in locals() else compatibility_info.get('new_rows_count', 0)
                    }
                    
                else:
                    # 새 파일은 유지 (부분/다중 캐시의 경우, 원본 확장자 유지)
                    try:
                        content_hash = get_data_content_hash(temp_filepath)
                        final_filename = f"data_{content_hash}{file_ext}" if content_hash else temp_filename
                    except Exception as hash_error:
                        logger.warning(f"⚠️ Hash calculation failed, using timestamp-based filename: {str(hash_error)}")
                        final_filename = temp_filename  # 해시 실패 시 임시 파일명 유지
                    
                    final_filepath = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)
                    
                    if temp_filepath != final_filepath:
                        # 🔧 강화된 파일 이동 로직 (Excel 파일 락 해제 대기)
                        moved_successfully = False
                        for attempt in range(3):  # 최대 3번 시도
                            try:
                                # Excel 파일 읽기 후 파일 락 해제를 위한 충분한 대기
                                import gc
                                gc.collect()  # 가비지 컬렉션으로 파일 핸들 해제
                                time.sleep(0.5 + attempt * 0.5)  # 점진적으로 대기 시간 증가
                                
                                shutil.move(temp_filepath, final_filepath)
                                logger.info(f"📝 [UPLOAD] File renamed: {final_filename} (attempt {attempt + 1})")
                                moved_successfully = True
                                break
                            except OSError as move_error:
                                logger.warning(f"⚠️ File move attempt {attempt + 1} failed: {str(move_error)}")
                                if attempt == 2:  # 마지막 시도
                                    logger.warning(f"⚠️ All move attempts failed, keeping original filename: {str(move_error)}")
                                    final_filepath = temp_filepath
                                    final_filename = temp_filename
                        
                        if not moved_successfully:
                            final_filepath = temp_filepath
                            final_filename = temp_filename
                    else:
                        logger.info(f"📝 [UPLOAD] File already has correct name: {final_filename}")
                        
                    response_data['filepath'] = final_filepath
                    response_data['filename'] = final_filename
                    # 확장 데이터가 아닌 경우에만 새로운 데이터 메시지 설정
                    if cache_type != 'extension':
                        response_data['cache_info']['message'] = "새로운 데이터입니다. 모델별로 적절한 데이터 범위를 사용하여 예측합니다."
            
            else:
                # 🆕 캐시가 없는 완전히 새로운 파일 처리
                logger.info(f"🆕 [NEW_FILE] No cache found - processing as new data file")
                
                # 새 파일을 정식 파일명으로 저장 (data 접두사 사용)
                try:
                    content_hash = get_data_content_hash(temp_filepath)
                    final_filename = f"data_{content_hash[:12]}_{os.path.splitext(normalized_filename)[0]}{file_ext}" if content_hash else temp_filename
                except Exception as hash_error:
                    logger.warning(f"⚠️ Hash calculation failed for new file, using timestamp-based filename: {str(hash_error)}")
                    final_filename = temp_filename  # 해시 실패 시 임시 파일명 유지
                
                final_filepath = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)
                
                if temp_filepath != final_filepath:
                    # 🔧 강화된 파일 이동 로직 (Excel 파일 락 해제 대기)
                    moved_successfully = False
                    for attempt in range(3):  # 최대 3번 시도
                        try:
                            # Excel 파일 읽기 후 파일 락 해제를 위한 충분한 대기
                            import gc
                            gc.collect()  # 가비지 컬렉션으로 파일 핸들 해제
                            time.sleep(0.5 + attempt * 0.5)  # 점진적으로 대기 시간 증가
                            
                            shutil.move(temp_filepath, final_filepath)
                            logger.info(f"📝 [UPLOAD] New file renamed: {final_filename} (attempt {attempt + 1})")
                            moved_successfully = True
                            break
                        except OSError as move_error:
                            logger.warning(f"⚠️ New file move attempt {attempt + 1} failed: {str(move_error)}")
                            if attempt == 2:  # 마지막 시도
                                logger.warning(f"⚠️ All move attempts failed, keeping original filename: {str(move_error)}")
                                final_filepath = temp_filepath
                                final_filename = temp_filename
                    
                    if not moved_successfully:
                        final_filepath = temp_filepath
                        final_filename = temp_filename
                else:
                    logger.info(f"📝 [UPLOAD] New file already has correct name: {final_filename}")
                    
                response_data['filepath'] = final_filepath
                response_data['filename'] = final_filename
                response_data['cache_info']['message'] = "새로운 데이터입니다. 모델별로 적절한 데이터 범위를 사용하여 예측합니다."
            
            # 🔑 업로드된 파일 경로를 전역 상태에 저장
            prediction_state['current_file'] = response_data['filepath']
            logger.info(f"📁 Set current_file in prediction_state: {response_data['filepath']}")
            
            # 🔧 성공 시 temp 파일 정리 (final_filepath와 다른 경우에만)
            if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
                final_filepath = response_data.get('filepath')
                if final_filepath and temp_filepath != final_filepath:
                    try:
                        os.remove(temp_filepath)
                        logger.info(f"🗑️ [CLEANUP] Success - temp file removed: {os.path.basename(temp_filepath)}")
                    except Exception as cleanup_error:
                        logger.warning(f"⚠️ [CLEANUP] Failed to remove temp file after success: {str(cleanup_error)}")
                else:
                    logger.info(f"📝 [CLEANUP] Temp file kept as final file: {os.path.basename(temp_filepath)}")
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Error during file upload: {str(e)}")
            # 🔧 강화된 temp 파일 정리
            try:
                if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                    logger.info(f"🗑️ [CLEANUP] Temp file removed on error: {os.path.basename(temp_filepath)}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ [CLEANUP] Failed to remove temp file on error: {str(cleanup_error)}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Only CSV and Excel files (.csv, .xlsx, .xls) are allowed'}), 400

@app.route('/api/holidays', methods=['GET'])
def get_holidays():
    """휴일 목록 조회 API"""
    try:
        # 휴일을 날짜와 설명이 포함된 딕셔너리 리스트로 변환
        holidays_list = []
        file_holidays = load_holidays_from_file()  # 파일에서 로드
        
        # 현재 전역 휴일에서 파일 휴일과 자동 감지 휴일 구분
        auto_detected = holidays - file_holidays
        
        for holiday_date in file_holidays:
            holidays_list.append({
                'date': holiday_date,
                'description': 'Holiday (from file)',
                'source': 'file'
            })
        
        for holiday_date in auto_detected:
            holidays_list.append({
                'date': holiday_date,
                'description': 'Holiday (detected from missing data)',
                'source': 'auto_detected'
            })
        
        # 날짜순으로 정렬
        holidays_list.sort(key=lambda x: x['date'])
        
        return jsonify({
            'success': True,
            'holidays': holidays_list,
            'count': len(holidays_list),
            'file_holidays': len(file_holidays),
            'auto_detected_holidays': len(auto_detected)
        })
    except Exception as e:
        logger.error(f"Error getting holidays: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'holidays': [],
            'count': 0
        }), 500

@app.route('/api/holidays/upload', methods=['POST'])
def upload_holidays():
    """휴일 목록 파일 업로드 API"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        try:
            # 임시 파일명 생성
            filename = secure_filename(f"holidays_{int(time.time())}{os.path.splitext(file.filename)[1]}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # 파일 저장
            file.save(filepath)
            
            # 휴일 정보 업데이트 - 보안 우회 기능 사용
            logger.info(f"🏖️ [HOLIDAY_UPLOAD] Processing uploaded holiday file: {filename}")
            new_holidays = update_holidays_safe(filepath)
            
            # 원본 파일을 holidays 디렉토리로 복사
            holidays_dir = 'holidays'
            if not os.path.exists(holidays_dir):
                os.makedirs(holidays_dir)
                logger.info(f"📁 Created holidays directory: {holidays_dir}")
            
            permanent_path = os.path.join(holidays_dir, 'holidays' + os.path.splitext(file.filename)[1])
            shutil.copy2(filepath, permanent_path)
            logger.info(f"📁 Holiday file copied to: {permanent_path}")
            
            # 임시 파일 정리
            try:
                os.remove(filepath)
                logger.info(f"🗑️ Temporary file removed: {filepath}")
            except:
                pass
            
            return jsonify({
                'success': True,
                'message': f'Successfully uploaded and loaded {len(new_holidays)} holidays',
                'filepath': permanent_path,
                'filename': os.path.basename(permanent_path),
                'holidays': list(new_holidays)
            })
        except Exception as e:
            logger.error(f"Error during holiday file upload: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({
        'error': 'Invalid file type. Only CSV and Excel files are allowed',
        'supported_extensions': {
            'standard': ['.csv', '.xlsx', '.xls'],
            'security': ['.cs (csv)', '.xl (xlsx)', '.log (xlsx)', '.dat (auto-detect)', '.txt (auto-detect)']
        }
    }), 400

# xlwings 대안 로더 (보안프로그램이 파일을 잠그는 경우 사용)
try:
    import xlwings as xw
    XLWINGS_AVAILABLE = True
    logger.info("✅ xlwings library available - Excel security bypass enabled")
except ImportError:
    XLWINGS_AVAILABLE = False
    logger.warning("⚠️ xlwings not available - falling back to pandas only")

@app.route('/api/holidays/reload', methods=['POST'])
def reload_holidays():
    """휴일 목록 재로드 API - 보안 우회 기능 포함"""
    try:
        filepath = request.json.get('filepath') if request.json else None
        
        logger.info(f"🔄 [HOLIDAY_RELOAD] Reloading holidays from: {filepath or 'default file'}")
        
        # 보안 우회 기능을 포함한 안전한 재로드
        new_holidays = update_holidays_safe(filepath)
        
        return jsonify({
            'success': True,
            'message': f'Successfully reloaded {len(new_holidays)} holidays',
            'holidays': list(new_holidays),
            'security_bypass_used': XLWINGS_AVAILABLE
        })
        
    except Exception as e:
        logger.error(f"❌ [HOLIDAY_RELOAD] Error reloading holidays: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to reload holidays: {str(e)}'
        }), 500

@app.route('/api/file/metadata', methods=['GET'])
def get_file_metadata():
    """파일 메타데이터 조회 API"""
    filepath = request.args.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # 파일 확장자에 따라 읽기 방식 결정
        file_ext = os.path.splitext(filepath.lower())[1]
        
        if file_ext == '.csv':
            # CSV 파일 처리
            df = pd.read_csv(filepath, nrows=5)  # 처음 5행만 읽기
            columns = df.columns.tolist()
            latest_date = None
            
            if 'Date' in df.columns:
                # 날짜 정보를 별도로 읽어서 최신 날짜 확인
                dates_df = pd.read_csv(filepath, usecols=['Date'])
                dates_df['Date'] = pd.to_datetime(dates_df['Date'])
                latest_date = dates_df['Date'].max().strftime('%Y-%m-%d')
        else:
            # Excel 파일 처리 (고급 처리 사용) - 🔧 중복 로딩 방지
            logger.info(f"🔍 [METADATA] Loading Excel data for metadata extraction...")
            df = load_data(filepath)
            # 인덱스가 Date인 경우 컬럼으로 복원
            if df.index.name == 'Date':
                full_df = df.copy()  # 🔧 전체 데이터 저장 (중복 로딩 방지)
                df = df.reset_index()
            else:
                full_df = df.copy()  # 🔧 전체 데이터 저장 (중복 로딩 방지)
            
            # 처음 5행만 선택
            df_sample = df.head(5)
            columns = df.columns.tolist()
            latest_date = None
            
            if 'Date' in df.columns:
                # 🔧 이미 로딩된 데이터에서 최신 날짜 확인 (중복 로딩 방지)
                if full_df.index.name == 'Date':
                    latest_date = pd.to_datetime(full_df.index).max().strftime('%Y-%m-%d')
                else:
                    latest_date = pd.to_datetime(full_df['Date']).max().strftime('%Y-%m-%d')
            
            # 메모리 정리
            df = df_sample
        
        return jsonify({
            'success': True,
            'rows': len(df),
            'columns': columns,
            'latest_date': latest_date,
            'sample': df.head().to_dict(orient='records')
        })
    except Exception as e:
        logger.error(f"Error reading file metadata: {str(e)}")
        return jsonify({'error': f'Error reading file: {str(e)}'}), 500
    
@app.route('/api/data/dates', methods=['GET'])
def get_available_dates():
    filepath = request.args.get('filepath')
    days_limit = int(request.args.get('limit', 999999))  # 기본값을 매우 큰 수로 설정 (모든 날짜)
    force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'  # 강제 새로고침 옵션
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # 🔄 파일의 최신 해시와 수정 시간 확인하여 변경 감지
        current_file_hash = get_data_content_hash(filepath)
        current_file_mtime = os.path.getmtime(filepath)
        
        logger.info(f"🔍 [DATE_REFRESH] Checking file status:")
        logger.info(f"  📁 File: {os.path.basename(filepath)}")
        logger.info(f"  🔑 Current hash: {current_file_hash[:12] if current_file_hash else 'None'}...")
        logger.info(f"  ⏰ Modified time: {datetime.fromtimestamp(current_file_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  🔄 Force refresh: {force_refresh}")
        
        # 파일 데이터 로드 및 분석 (파일 형식에 맞게, 항상 최신 파일 내용 확인)
        # 🔑 단일 날짜 예측용: LSTM 모델 타입 지정하여 2022년 이후 데이터만 로드
        file_ext = os.path.splitext(filepath.lower())[1]
        if file_ext == '.csv':
            # CSV 파일의 경우 다양한 구분자로 시도
            try:
                df = pd.read_csv(filepath)
                # 만약 첫 번째 컬럼에서 쉼표가 발견되면 구분자가 다를 수 있음
                if len(df.columns) == 1 and ',' in str(df.columns[0]):
                    logger.info("🔧 [CSV_PARSE] Trying different separators for CSV...")
                    # 세미콜론으로 시도
                    try:
                        df = pd.read_csv(filepath, sep=';')
                        logger.info(f"✅ [CSV_PARSE] Successfully parsed with semicolon: {df.shape}")
                    except:
                        # 탭으로 시도
                        try:
                            df = pd.read_csv(filepath, sep='\t')
                            logger.info(f"✅ [CSV_PARSE] Successfully parsed with tab: {df.shape}")
                        except:
                            logger.warning("⚠️ [CSV_PARSE] Could not parse with alternative separators")
            except Exception as e:
                logger.error(f"❌ [CSV_PARSE] Error reading CSV: {str(e)}")
                raise
        else:
            # Excel 파일인 경우 load_data 함수 사용 (LSTM 모델 타입 지정)
            df = load_data(filepath, model_type='lstm')
            # 인덱스가 Date인 경우 컬럼으로 복원
            if df.index.name == 'Date':
                df = df.reset_index()

        # Date 컬럼 존재 여부 확인 및 처리
        if 'Date' not in df.columns:
            logger.error(f"❌ [DATE_COLUMN] Date column not found. Available columns: {list(df.columns)}")
            # 첫 번째 컬럼이 날짜 형태인지 확인
            first_col = df.columns[0]
            try:
                # 첫 번째 컬럼의 첫 몇 개 값이 날짜인지 확인
                sample_values = df[first_col].head(5).dropna()
                pd.to_datetime(sample_values)
                logger.info(f"✅ [DATE_COLUMN] Using first column as Date: {first_col}")
                df['Date'] = pd.to_datetime(df[first_col])
            except:
                logger.error(f"❌ [DATE_COLUMN] First column is not date format: {first_col}")
                raise ValueError(f"Date column not found and first column '{first_col}' is not a valid date format")
        else:
            df['Date'] = pd.to_datetime(df['Date'])
        
        df = df.sort_values('Date')
        
        # 🏖️ 데이터를 로드한 후 휴일 정보 자동 업데이트 (빈 평일 감지) - 임시 비활성화
        logger.info(f"🏖️ [HOLIDAYS] Auto-detection temporarily disabled to show more dates...")
        # updated_holidays = update_holidays(df=df)
        updated_holidays = load_holidays_from_file()  # 파일 휴일만 사용
        logger.info(f"🏖️ [HOLIDAYS] Total holidays (file only): {len(updated_holidays)}")
        
        # 📊 실제 파일 데이터 범위 확인 (캐시 무시)
        total_rows = len(df)
        data_start_date = df.iloc[0]['Date']
        data_end_date = df.iloc[-1]['Date']
        
        logger.info(f"📊 [ACTUAL_DATA] File analysis results:")
        logger.info(f"  📈 Total data rows: {total_rows}")
        logger.info(f"  📅 Actual date range: {data_start_date.strftime('%Y-%m-%d')} ~ {data_end_date.strftime('%Y-%m-%d')}")
        
        # 🔍 기존 캐시와 비교 (있는 경우)
        existing_cache_range = find_existing_cache_range(filepath)
        if existing_cache_range and not force_refresh:
            # numpy.ndarray와 Timestamp 비교 에러 방지를 위해 Timestamp로 정규화
            try:
                cache_start = pd.to_datetime(existing_cache_range['start_date'])
                cache_cutoff = pd.to_datetime(existing_cache_range['cutoff_date'])
                
                # 실제 데이터 날짜들도 Timestamp로 변환하여 안전한 비교
                actual_start = pd.to_datetime(data_start_date)
                actual_end = pd.to_datetime(data_end_date)
            except Exception as e:
                logger.warning(f"⚠️ [CACHE_COMPARE] Error converting dates for comparison: {str(e)}")
                cache_start = None
                cache_cutoff = None
            
            # cache_start와 cache_cutoff가 유효한 경우에만 비교 수행
            if cache_start is not None and cache_cutoff is not None:
                # 단일 값으로 변환 (array인 경우 첫 번째 요소 사용)
                if hasattr(cache_start, '__len__') and len(cache_start) > 0:
                    cache_start = cache_start[0] if hasattr(cache_start[0], 'strftime') else pd.to_datetime(cache_start[0])
                if hasattr(cache_cutoff, '__len__') and len(cache_cutoff) > 0:
                    cache_cutoff = cache_cutoff[0] if hasattr(cache_cutoff[0], 'strftime') else pd.to_datetime(cache_cutoff[0])
                
                # 확실히 Timestamp 타입으로 변환
                cache_start = pd.Timestamp(cache_start)
                cache_cutoff = pd.Timestamp(cache_cutoff)
                data_start_date = pd.Timestamp(data_start_date)
                data_end_date = pd.Timestamp(data_end_date)
                
                logger.info(f"💾 [CACHE_COMPARISON] Found existing cache range:")
                logger.info(f"  📅 Cached range: {cache_start.strftime('%Y-%m-%d')} ~ {cache_cutoff.strftime('%Y-%m-%d')}")
                
                # 실제 데이터가 캐시된 범위보다 확장되었는지 확인
                data_extended = (
                    data_start_date < cache_start or 
                    data_end_date > cache_cutoff
                )
            else:
                # 캐시 날짜 비교에 실패한 경우 확장된 것으로 간주
                logger.warning("⚠️ [CACHE_COMPARE] Cache date comparison failed, treating as data extended")
                data_extended = True
            
            if data_extended:
                logger.info(f"📈 [DATA_EXTENSION] Data has been extended!")
                logger.info(f"  ⬅️ Start extension: {data_start_date.strftime('%Y-%m-%d')} vs cached {cache_start.strftime('%Y-%m-%d')}")
                logger.info(f"  ➡️ End extension: {data_end_date.strftime('%Y-%m-%d')} vs cached {cache_cutoff.strftime('%Y-%m-%d')}")
                logger.info(f"  🔄 Using extended data range for date calculation")
            else:
                logger.info(f"✅ [NO_EXTENSION] Data range matches cached range, proceeding with current data")
        else:
            if force_refresh:
                logger.info(f"🔄 [FORCE_REFRESH] Ignoring cache due to force refresh")
            else:
                logger.info(f"📭 [NO_CACHE] No existing cache found, using full data range")
        
        # 데이터 마지막 날짜의 다음 영업일을 계산하여 예측 시작점 설정 (실제 데이터 기준)
        # 최소 100개 행 이상의 히스토리가 있는 경우에만 예측 가능
        min_history_rows = 100
        prediction_start_index = max(min_history_rows, total_rows // 4)  # 25% 지점 또는 최소 100행 중 큰 값
        
        # 실제 예측에 사용할 수 있는 모든 날짜 (충분한 히스토리가 있는 날짜부터)
        predictable_dates = df.iloc[prediction_start_index:]['Date']
        
        # 예측 시작 임계값 계산 (참고용)
        if prediction_start_index < total_rows:
            prediction_threshold_date = df.iloc[prediction_start_index]['Date']
        else:
            prediction_threshold_date = data_end_date
        
        logger.info(f"🎯 [PREDICTION_CALC] Prediction calculation:")
        logger.info(f"  📊 Min history rows: {min_history_rows}")
        logger.info(f"  📍 Start index: {prediction_start_index} (date: {prediction_threshold_date.strftime('%Y-%m-%d')})")
        logger.info(f"  📅 Predictable dates: {len(predictable_dates)} dates available")
        
        # 예측 가능한 모든 날짜를 내림차순으로 반환 (최신 날짜부터)
        # days_limit보다 작은 경우에만 제한 적용
        if len(predictable_dates) <= days_limit:
            dates = predictable_dates.sort_values(ascending=False).dt.strftime('%Y-%m-%d').tolist()
        else:
            dates = predictable_dates.sort_values(ascending=False).head(days_limit).dt.strftime('%Y-%m-%d').tolist()
        
        logger.info(f"🔢 [FINAL_RESULT] Final date calculation:")
        logger.info(f"  📊 Available predictable dates: {len(predictable_dates)}")
        logger.info(f"  📋 Returned dates: {len(dates)}")
        logger.info(f"  📅 Latest available date: {dates[0] if dates else 'None'}")
        
        response_data = {
            'success': True,
            'dates': dates,
            'latest_date': dates[0] if dates else None,  # 첫 번째 요소가 최신 날짜 (내림차순)
            'data_start_date': data_start_date.strftime('%Y-%m-%d'),
            'data_end_date': data_end_date.strftime('%Y-%m-%d'),
            'prediction_threshold': prediction_threshold_date.strftime('%Y-%m-%d'),
            'min_history_rows': min_history_rows,
            'total_rows': total_rows,
            'file_hash': current_file_hash[:12] if current_file_hash else None,  # 추가: 파일 해시 정보
            'file_modified': datetime.fromtimestamp(current_file_mtime).strftime('%Y-%m-%d %H:%M:%S')  # 추가: 파일 수정 시간
        }
        
        logger.info(f"📡 [API_RESPONSE] Sending enhanced dates response:")
        logger.info(f"  📅 Data range: {response_data['data_start_date']} ~ {response_data['data_end_date']}")
        logger.info(f"  🎯 Prediction threshold: {response_data['prediction_threshold']}")
        logger.info(f"  📅 Available date range: {dates[-1] if dates else 'None'} ~ {dates[0] if dates else 'None'} (최신부터)")
        logger.info(f"  🔑 File signature: {response_data['file_hash']} @ {response_data['file_modified']}")
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error reading dates: {str(e)}")
        return jsonify({'error': f'Error reading dates: {str(e)}'}), 500

@app.route('/api/data/refresh', methods=['POST'])
def refresh_file_data():
    """파일 데이터 새로고침 및 캐시 갱신 API"""
    try:
        filepath = request.json.get('filepath') if request.json else request.args.get('filepath')
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # 파일 해시와 수정 시간 확인
        current_file_hash = get_data_content_hash(filepath)
        current_file_mtime = os.path.getmtime(filepath)
        
        logger.info(f"🔄 [FILE_REFRESH] Starting file data refresh:")
        logger.info(f"  📁 File: {os.path.basename(filepath)}")
        logger.info(f"  🔑 Hash: {current_file_hash[:12] if current_file_hash else 'None'}...")
        
        # 기존 캐시 확인
        existing_cache_range = find_existing_cache_range(filepath)
        refresh_needed = False
        refresh_reason = []
        
        if existing_cache_range:
            # 캐시된 메타데이터와 비교
            meta_file = existing_cache_range.get('meta_file')
            if meta_file and os.path.exists(meta_file):
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                    
                    cached_hash = meta_data.get('file_hash')
                    cached_mtime = meta_data.get('file_modified_time')
                    
                    if cached_hash != current_file_hash:
                        refresh_needed = True
                        refresh_reason.append("File content changed")
                        
                    if cached_mtime and cached_mtime != current_file_mtime:
                        refresh_needed = True
                        refresh_reason.append("File modification time changed")
                        
                except Exception as e:
                    logger.warning(f"Error reading cache metadata: {str(e)}")
                    refresh_needed = True
                    refresh_reason.append("Cache metadata error")
            else:
                refresh_needed = True
                refresh_reason.append("No cache metadata found")
        else:
            refresh_needed = True
            refresh_reason.append("No existing cache")
        
        # 파일 데이터 분석 (파일 형식에 맞게)
        file_ext = os.path.splitext(filepath.lower())[1]
        if file_ext == '.csv':
            # CSV 파일의 경우 다양한 구분자로 시도
            try:
                df = pd.read_csv(filepath)
                # 만약 첫 번째 컬럼에서 쉼표가 발견되면 구분자가 다를 수 있음
                if len(df.columns) == 1 and ',' in str(df.columns[0]):
                    logger.info("🔧 [CSV_PARSE] Trying different separators for CSV...")
                    # 세미콜론으로 시도
                    try:
                        df = pd.read_csv(filepath, sep=';')
                        logger.info(f"✅ [CSV_PARSE] Successfully parsed with semicolon: {df.shape}")
                    except:
                        # 탭으로 시도
                        try:
                            df = pd.read_csv(filepath, sep='\t')
                            logger.info(f"✅ [CSV_PARSE] Successfully parsed with tab: {df.shape}")
                        except:
                            logger.warning("⚠️ [CSV_PARSE] Could not parse with alternative separators")
            except Exception as e:
                logger.error(f"❌ [CSV_PARSE] Error reading CSV: {str(e)}")
                raise
        else:
            # Excel 파일인 경우 load_data 함수 사용
            df = load_data(filepath)
            # 인덱스가 Date인 경우 컬럼으로 복원
            if df.index.name == 'Date':
                df = df.reset_index()

        # Date 컬럼 존재 여부 확인 및 처리
        if 'Date' not in df.columns:
            logger.error(f"❌ [DATE_COLUMN] Date column not found. Available columns: {list(df.columns)}")
            # 첫 번째 컬럼이 날짜 형태인지 확인
            first_col = df.columns[0]
            try:
                # 첫 번째 컬럼의 첫 몇 개 값이 날짜인지 확인
                sample_values = df[first_col].head(5).dropna()
                pd.to_datetime(sample_values)
                logger.info(f"✅ [DATE_COLUMN] Using first column as Date: {first_col}")
                df['Date'] = pd.to_datetime(df[first_col])
            except:
                logger.error(f"❌ [DATE_COLUMN] First column is not date format: {first_col}")
                raise ValueError(f"Date column not found and first column '{first_col}' is not a valid date format")
        else:
            df['Date'] = pd.to_datetime(df['Date'])
        
        df = df.sort_values('Date')
        
        current_data_range = {
            'start_date': df.iloc[0]['Date'],
            'end_date': df.iloc[-1]['Date'],
            'total_rows': len(df)
        }
        
        # 캐시와 실제 데이터 범위 비교
        if existing_cache_range and not refresh_needed:
            cache_start = pd.to_datetime(existing_cache_range['start_date'])
            cache_cutoff = pd.to_datetime(existing_cache_range['cutoff_date'])
            
            if (current_data_range['start_date'] < cache_start or 
                current_data_range['end_date'] > cache_cutoff):
                refresh_needed = True
                refresh_reason.append("Data range extended")
        
        response_data = {
            'success': True,
            'refresh_needed': refresh_needed,
            'refresh_reasons': refresh_reason,
            'file_info': {
                'hash': current_file_hash[:12] if current_file_hash else None,
                'modified_time': datetime.fromtimestamp(current_file_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'total_rows': current_data_range['total_rows'],
                'date_range': {
                    'start': current_data_range['start_date'].strftime('%Y-%m-%d'),
                    'end': current_data_range['end_date'].strftime('%Y-%m-%d')
                }
            }
        }
        
        if existing_cache_range:
            response_data['cache_info'] = {
                'date_range': {
                    'start': existing_cache_range['start_date'],
                    'end': existing_cache_range['cutoff_date']
                },
                'meta_file': existing_cache_range.get('meta_file')
            }
        
        logger.info(f"📊 [REFRESH_ANALYSIS] File refresh analysis:")
        logger.info(f"  🔄 Refresh needed: {refresh_needed}")
        logger.info(f"  📝 Reasons: {', '.join(refresh_reason) if refresh_reason else 'None'}")
        logger.info(f"  📅 Current range: {response_data['file_info']['date_range']['start']} ~ {response_data['file_info']['date_range']['end']}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in file refresh check: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/compare-files', methods=['POST'])
def debug_compare_files():
    """두 파일을 직접 비교하여 차이점을 분석하는 디버깅 API"""
    try:
        data = request.json
        file1_path = data.get('file1_path')
        file2_path = data.get('file2_path')
        
        if not file1_path or not file2_path:
            return jsonify({'error': 'Both file paths are required'}), 400
            
        if not os.path.exists(file1_path) or not os.path.exists(file2_path):
            return jsonify({'error': 'One or both files do not exist'}), 404
        
        logger.info(f"🔍 [DEBUG_COMPARE] Comparing files:")
        logger.info(f"  📁 File 1: {file1_path}")
        logger.info(f"  📁 File 2: {file2_path}")
        
        # 파일 기본 정보
        file1_hash = get_data_content_hash(file1_path)
        file2_hash = get_data_content_hash(file2_path)
        file1_size = os.path.getsize(file1_path)
        file2_size = os.path.getsize(file2_path)
        file1_mtime = os.path.getmtime(file1_path)
        file2_mtime = os.path.getmtime(file2_path)
        
        # 데이터 분석 (파일 형식에 맞게)
        def load_file_safely(filepath):
            file_ext = os.path.splitext(filepath.lower())[1]
            if file_ext == '.csv':
                return pd.read_csv(filepath)
            else:
                # Excel 파일인 경우 load_data 함수 사용
                df = load_data(filepath)
                # 인덱스가 Date인 경우 컬럼으로 복원
                if df.index.name == 'Date':
                    df = df.reset_index()
                return df
        
        df1 = load_file_safely(file1_path)
        df2 = load_file_safely(file2_path)
        
        if 'Date' in df1.columns and 'Date' in df2.columns:
            df1['Date'] = pd.to_datetime(df1['Date'])
            df2['Date'] = pd.to_datetime(df2['Date'])
            df1 = df1.sort_values('Date')
            df2 = df2.sort_values('Date')
            
            file1_dates = {
                'start': df1['Date'].min(),
                'end': df1['Date'].max(),
                'count': len(df1)
            }
            
            file2_dates = {
                'start': df2['Date'].min(),
                'end': df2['Date'].max(),
                'count': len(df2)
            }
        else:
            file1_dates = {'error': 'No Date column'}
            file2_dates = {'error': 'No Date column'}
        
        # 확장 체크
        extension_result = check_data_extension(file1_path, file2_path)
        
        # 캐시 호환성 체크
        cache_result = find_compatible_cache_file(file2_path)
        
        response_data = {
            'success': True,
            'comparison': {
                'file1': {
                    'path': file1_path,
                    'hash': file1_hash[:12] if file1_hash else None,
                    'size': file1_size,
                    'modified': datetime.fromtimestamp(file1_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    'dates': {
                        'start': file1_dates['start'].strftime('%Y-%m-%d') if isinstance(file1_dates.get('start'), pd.Timestamp) else str(file1_dates.get('start')),
                        'end': file1_dates['end'].strftime('%Y-%m-%d') if isinstance(file1_dates.get('end'), pd.Timestamp) else str(file1_dates.get('end')),
                        'count': file1_dates.get('count')
                    } if 'error' not in file1_dates else file1_dates
                },
                'file2': {
                    'path': file2_path,
                    'hash': file2_hash[:12] if file2_hash else None,
                    'size': file2_size,
                    'modified': datetime.fromtimestamp(file2_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    'dates': {
                        'start': file2_dates['start'].strftime('%Y-%m-%d') if isinstance(file2_dates.get('start'), pd.Timestamp) else str(file2_dates.get('start')),
                        'end': file2_dates['end'].strftime('%Y-%m-%d') if isinstance(file2_dates.get('end'), pd.Timestamp) else str(file2_dates.get('end')),
                        'count': file2_dates.get('count')
                    } if 'error' not in file2_dates else file2_dates
                },
                'identical_hash': file1_hash == file2_hash,
                'size_difference': file2_size - file1_size,
                'extension_analysis': extension_result,
                'cache_analysis': cache_result
            }
        }
        
        logger.info(f"📊 [DEBUG_COMPARE] Comparison results:")
        logger.info(f"  🔑 Identical hash: {file1_hash == file2_hash}")
        logger.info(f"  📏 Size difference: {file2_size - file1_size} bytes")
        logger.info(f"  📈 Is extension: {extension_result.get('is_extension', False)}")
        logger.info(f"  💾 Cache found: {cache_result.get('found', False)}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in file comparison: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/saved', methods=['GET'])
def get_saved_predictions():
    """저장된 예측 결과 목록 조회 API"""
    try:
        limit = int(request.args.get('limit', 100))
        predictions_list = get_saved_predictions_list(limit)
        
        return jsonify({
            'success': True,
            'predictions': predictions_list,
            'count': len(predictions_list)
        })
    except Exception as e:
        logger.error(f"Error getting saved predictions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/saved/<date>', methods=['GET'])
def get_saved_prediction_by_date(date):
    """특정 날짜의 저장된 예측 결과 조회 API"""
    try:
        result = load_prediction_from_csv(date)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify({'error': result['error']}), 404
    except Exception as e:
        logger.error(f"Error loading saved prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/saved/<date>', methods=['DELETE'])
def delete_saved_prediction_api(date):
    """저장된 예측 결과 삭제 API"""
    try:
        result = delete_saved_prediction(date)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify({'error': result['error']}), 500
    except Exception as e:
        logger.error(f"Error deleting saved prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/saved/<date>/update-actual', methods=['POST'])
def update_prediction_actual_values_api(date):
    """캐시된 예측의 실제값만 업데이트하는 API - 성능 최적화"""
    try:
        # 요청 파라미터
        data = request.json or {}
        update_latest_only = data.get('update_latest_only', True)
        
        logger.info(f"🔄 [API] Updating actual values for prediction {date}")
        logger.info(f"  📊 Update latest only: {update_latest_only}")
        
        # 실제값 업데이트 실행
        result = update_cached_prediction_actual_values(date, update_latest_only)
        
        if result['success']:
            logger.info(f"✅ [API] Successfully updated {result.get('updated_count', 0)} actual values")
            return jsonify({
                'success': True,
                'updated_count': result.get('updated_count', 0),
                'message': f'Updated {result.get("updated_count", 0)} actual values',
                'predictions': result['predictions']
            })
        else:
            logger.error(f"❌ [API] Failed to update actual values: {result.get('error')}")
            return jsonify({'error': result['error']}), 500
            
    except Exception as e:
        logger.error(f"❌ [API] Error updating actual values: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/export', methods=['GET'])
def export_predictions():
    """저장된 예측 결과들을 하나의 CSV 파일로 내보내기 API"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # 날짜 범위에 따른 예측 로드
        if start_date:
            predictions = load_accumulated_predictions_from_csv(start_date, end_date)
        else:
            # 모든 저장된 예측 로드
            predictions_list = get_saved_predictions_list(limit=1000)
            predictions = []
            for pred_info in predictions_list:
                loaded = load_prediction_from_csv(pred_info['prediction_date'])
                if loaded['success']:
                    predictions.extend(loaded['predictions'])
        
        if not predictions:
            return jsonify({'error': 'No predictions found for export'}), 404
        
        # DataFrame으로 변환
        if isinstance(predictions[0], dict) and 'predictions' in predictions[0]:
            # 누적 예측 형식인 경우
            all_predictions = []
            for pred_group in predictions:
                all_predictions.extend(pred_group['predictions'])
            export_df = pd.DataFrame(all_predictions)
        else:
            # 단순 예측 리스트인 경우
            export_df = pd.DataFrame(predictions)
        
        # 임시 파일 생성 (보안을 위해 .cs 확장자 사용)
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.cs', delete=False)
        export_df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        # 파일 전송
        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name=f'predictions_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.cs',
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"Error exporting predictions: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 7. API 엔드포인트 수정 - 스마트 캐시 사용
@app.route('/api/predict', methods=['POST'])
def start_prediction_compatible():
    """호환성을 유지하는 예측 시작 API - 캐시 우선 사용 (로그 강화)"""
    from app.prediction.background_tasks import background_prediction_simple_compatible # ✅ 함수 내부에 추가
    global prediction_state
    
    if prediction_state['is_predicting']:
        return jsonify({
            'success': False,
            'error': 'Prediction already in progress',
            'progress': prediction_state['prediction_progress']
        }), 409
    
    data = request.json
    filepath = data.get('filepath')
    current_date = data.get('date')
    save_to_csv = data.get('save_to_csv', True)
    use_cache = data.get('use_cache', True)  # 기본값 True
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Invalid file path'}), 400
    
    # 🔑 파일 경로를 전역 상태에 저장 (캐시 연동용)
    prediction_state['current_file'] = filepath
    
    # ✅ 로그 강화
    logger.info(f"🚀 Prediction API called:")
    logger.info(f"  📅 Target date: {current_date}")
    logger.info(f"  📁 Data file: {filepath}")
    logger.info(f"  💾 Save to CSV: {save_to_csv}")
    logger.info(f"  🔄 Use cache: {use_cache}")
    
    # 호환성 유지 백그라운드 함수 실행 (캐시 우선 사용, 단일 예측만)
    thread = Thread(target=background_prediction_simple_compatible, 
                   args=(filepath, current_date, save_to_csv, use_cache))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Compatible prediction started (cache-first)',
        'use_cache': use_cache,
        'cache_priority': 'high',
        'features': ['Cache-first loading', 'Unified file naming', 'Enhanced logging', 'Past/Future visualization split']
    })

@app.route('/api/predict/status', methods=['GET'])
def prediction_status():
    """예측 상태 확인 API (남은 시간 추가)"""
    from app.prediction.background_tasks import calculate_estimated_time_remaining # ✅ 함수 내부에 추가
    global prediction_state
    
    status = {
        'is_predicting': prediction_state['is_predicting'],
        'progress': prediction_state['prediction_progress'],
        'error': prediction_state['error']
    }
    
    # 예측 중인 경우 남은 시간 계산
    if prediction_state['is_predicting'] and prediction_state['prediction_start_time']:
        time_info = calculate_estimated_time_remaining(
            prediction_state['prediction_start_time'], 
            prediction_state['prediction_progress']
        )
        status.update(time_info)
    
    # 예측이 완료된 경우 날짜 정보도 반환
    if not prediction_state['is_predicting'] and prediction_state['current_date']:
        status['current_date'] = prediction_state['current_date']
    
    return jsonify(status)

@app.route('/api/results', methods=['GET'])
def get_prediction_results_compatible():
    """호환성을 유지하는 예측 결과 조회 API (오류 수정)"""
    global prediction_state
    
    logger.info(f"=== API /results called (compatible version) ===")
    
    if prediction_state['is_predicting']:
        return jsonify({
            'success': False,
            'error': 'Prediction in progress',
            'progress': prediction_state['prediction_progress']
        }), 409
    
    if prediction_state['latest_predictions'] is None:
        return jsonify({'error': 'No prediction results available'}), 404

    try:
        # 예측 데이터를 기존 형태로 변환
        if isinstance(prediction_state['latest_predictions'], list):
            raw_predictions = prediction_state['latest_predictions']
        else:
            raw_predictions = prediction_state['latest_predictions']
        
        # 기존 형태로 변환
        compatible_predictions = convert_to_legacy_format(raw_predictions)
        
        logger.info(f"Converted {len(raw_predictions)} predictions to legacy format")
        logger.info(f"Sample converted prediction: {compatible_predictions[0] if compatible_predictions else 'None'}")
        
        # 메트릭 정리
        metrics = prediction_state['latest_metrics']
        cleaned_metrics = {}
        if metrics:
            for key, value in metrics.items():
                cleaned_metrics[key] = safe_serialize_value(value)
        
        # 구간 점수 안전 정리 - 오류 방지 강화
        interval_scores = prediction_state['latest_interval_scores'] or []
        
        # interval_scores 데이터 타입 검증 및 안전 처리
        if interval_scores is None:
            interval_scores = []
        elif not isinstance(interval_scores, (list, dict)):
            logger.warning(f"⚠️ Unexpected interval_scores type: {type(interval_scores)}, converting to empty list")
            interval_scores = []
        elif isinstance(interval_scores, dict) and not interval_scores:
            interval_scores = []
        
        try:
            cleaned_interval_scores = clean_interval_scores_safe(interval_scores)
        except Exception as interval_error:
            logger.error(f"❌ Error cleaning interval_scores: {str(interval_error)}")
            cleaned_interval_scores = []
        
        # MA 결과 정리 및 필요시 재계산
        ma_results = prediction_state['latest_ma_results'] or {}
        cleaned_ma_results = {}
        
        # 이동평균 결과가 없거나 비어있다면 재계산 시도
        if not ma_results or len(ma_results) == 0:
            logger.info("🔄 MA results missing, attempting to recalculate...")
            try:
                # 현재 데이터와 예측 결과를 사용하여 이동평균 재계산
                current_date = prediction_state.get('current_date')
                if current_date and prediction_state.get('latest_file_path'):
                    # 원본 데이터 로드
                    df = load_data(prediction_state['latest_file_path'])
                    if df is not None and not df.empty:
                        # 현재 날짜를 datetime으로 변환
                        if isinstance(current_date, str):
                            current_date_dt = pd.to_datetime(current_date)
                        else:
                            current_date_dt = current_date
                        
                        # 과거 데이터 추출
                        historical_data = df[df.index <= current_date_dt].copy()
                        
                        # 예측 데이터를 이동평균 계산용으로 변환
                        ma_input_data = []
                        for pred in raw_predictions:
                            try:
                                ma_item = {
                                    'Date': pd.to_datetime(pred.get('Date') or pred.get('date')),
                                    'Prediction': safe_serialize_value(pred.get('Prediction') or pred.get('prediction')),
                                    'Actual': safe_serialize_value(pred.get('Actual') or pred.get('actual'))
                                }
                                ma_input_data.append(ma_item)
                            except Exception as e:
                                logger.warning(f"⚠️ Error processing MA data item: {str(e)}")
                                continue
                        
                        # 이동평균 계산
                        if ma_input_data:
                            ma_results = calculate_moving_averages_with_history(
                                ma_input_data, historical_data, target_col='MOPJ'
                            )
                            if ma_results:
                                logger.info(f"✅ MA recalculated successfully with {len(ma_results)} windows")
                                prediction_state['latest_ma_results'] = ma_results
                            else:
                                logger.warning("⚠️ MA recalculation returned empty results")
                        else:
                            logger.warning("⚠️ No valid input data for MA calculation")
                    else:
                        logger.warning("⚠️ Unable to load original data for MA calculation")
                else:
                    logger.warning("⚠️ Missing current_date or file_path for MA calculation")
            except Exception as e:
                logger.error(f"❌ Error recalculating MA: {str(e)}")
        
        # MA 결과 정리
        for key, value in ma_results.items():
            if isinstance(value, list):
                cleaned_ma_results[key] = []
                for item in value:
                    if isinstance(item, dict):
                        cleaned_item = {}
                        for k, v in item.items():
                            cleaned_item[k] = safe_serialize_value(v)
                        cleaned_ma_results[key].append(cleaned_item)
                    else:
                        cleaned_ma_results[key].append(safe_serialize_value(item))
            else:
                cleaned_ma_results[key] = safe_serialize_value(value)
        
        # 어텐션 데이터 정리
        attention_data = prediction_state['latest_attention_data']
        cleaned_attention = None
        
        logger.info(f"📊 [ATTENTION] Processing attention data: available={bool(attention_data)}")
        if attention_data:
            logger.info(f"📊 [ATTENTION] Original keys: {list(attention_data.keys())}")
            
            cleaned_attention = {}
            for key, value in attention_data.items():
                if key == 'image' and value:
                    cleaned_attention[key] = value  # base64 이미지는 그대로
                    logger.info(f"📊 [ATTENTION] Image data preserved (length: {len(value) if isinstance(value, str) else 'N/A'})")
                elif isinstance(value, dict):
                    cleaned_attention[key] = {}
                    for k, v in value.items():
                        cleaned_attention[key][k] = safe_serialize_value(v)
                    logger.info(f"📊 [ATTENTION] Dict processed for key '{key}': {len(cleaned_attention[key])} items")
                else:
                    cleaned_attention[key] = safe_serialize_value(value)
                    logger.info(f"📊 [ATTENTION] Value processed for key '{key}': {type(value)}")
            
            logger.info(f"📊 [ATTENTION] Final cleaned keys: {list(cleaned_attention.keys())}")
        else:
            logger.warning(f"📊 [ATTENTION] No attention data available in prediction_state")
        
        # 플롯 데이터 정리
        plots = prediction_state['latest_plots'] or {}
        cleaned_plots = {}
        for key, value in plots.items():
            if isinstance(value, dict):
                cleaned_plots[key] = {}
                for k, v in value.items():
                    if k == 'image' and v:
                        cleaned_plots[key][k] = v  # base64 이미지는 그대로
                    else:
                        cleaned_plots[key][k] = safe_serialize_value(v)
            else:
                cleaned_plots[key] = safe_serialize_value(value)
        
        response_data = {
            'success': True,
            'current_date': safe_serialize_value(prediction_state['current_date']),
            'predictions': compatible_predictions,  # 호환성 유지된 형태
            'interval_scores': cleaned_interval_scores,
            'ma_results': cleaned_ma_results,
            'attention_data': cleaned_attention,
            'plots': cleaned_plots,
            'metrics': cleaned_metrics if cleaned_metrics else None,
            'selected_features': prediction_state['selected_features'] or [],
            'feature_importance': safe_serialize_value(prediction_state.get('feature_importance')),
            'semimonthly_period': safe_serialize_value(prediction_state['semimonthly_period']),
            'next_semimonthly_period': safe_serialize_value(prediction_state['next_semimonthly_period'])
        }
        
        # 🔧 강화된 JSON 직렬화 테스트
        try:
            test_json = json.dumps(response_data)
            # 직렬화된 JSON에 NaN이 포함되어 있는지 추가 확인
            if 'NaN' in test_json or 'Infinity' in test_json:
                logger.error(f"JSON contains NaN/Infinity values")
                # NaN 값들을 모두 null로 교체
                test_json_cleaned = test_json.replace('NaN', 'null').replace('Infinity', 'null').replace('-Infinity', 'null')
                response_data = json.loads(test_json_cleaned)
            logger.info(f"JSON serialization test: SUCCESS (length: {len(test_json)})")
        except Exception as json_error:
            logger.error(f"JSON serialization test: FAILED - {str(json_error)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            
            # 🔧 강화된 응급 처치: 재귀적 NaN 제거
            try:
                logger.info("🔧 Attempting emergency data cleaning...")
                
                def deep_clean_nan(obj):
                    """재귀적으로 모든 NaN 값을 제거"""
                    if obj is None:
                        return None
                    elif isinstance(obj, dict):
                        cleaned = {}
                        for k, v in obj.items():
                            cleaned[k] = deep_clean_nan(v)
                        return cleaned
                    elif isinstance(obj, list):
                        return [deep_clean_nan(item) for item in obj]
                    elif isinstance(obj, (int, float, np.number)):
                        try:
                            if pd.isna(obj) or np.isnan(obj) or np.isinf(obj):
                                return None
                            return float(obj) if isinstance(obj, (float, np.floating)) else int(obj)
                        except:
                            return None
                    elif isinstance(obj, str):
                        if obj.lower() in ['nan', 'inf', '-inf', 'infinity', '-infinity']:
                            return None
                        return obj
                    else:
                        return safe_serialize_value(obj)
                
                # 전체 응답 데이터 정리
                response_data = deep_clean_nan(response_data)
                
                # 재시도
                test_json = json.dumps(response_data)
                logger.info("✅ Emergency cleaning successful")
                
            except Exception as emergency_error:
                logger.error(f"❌ Emergency cleaning failed: {str(emergency_error)}")
                logger.error(f"❌ Original error: {str(json_error)}")
                return jsonify({
                    'success': False,
                    'error': f'Data serialization error: {str(json_error)}'
                }), 500
        
        logger.info(f"=== Compatible Response Summary ===")
        logger.info(f"Total predictions: {len(compatible_predictions)}")
        logger.info(f"Has metrics: {cleaned_metrics is not None}")
        logger.info(f"Sample prediction fields: {list(compatible_predictions[0].keys()) if compatible_predictions else 'None'}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error creating compatible response: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Error creating response: {str(e)}'}), 500

@app.route('/api/results/attention-map', methods=['GET'])
def get_attention_map():
    """어텐션 맵 데이터 조회 API"""
    global prediction_state
    
    logger.info("🔍 [ATTENTION_MAP] API call received - FINAL UPDATE")
    
    # 어텐션 데이터 확인
    attention_data = prediction_state.get('latest_attention_data')
    
    # 테스트용: 데이터가 없으면 더미 데이터 생성
    test_mode = request.args.get('test', '').lower() == 'true'
    
    if not attention_data:
        if test_mode:
            logger.info("🧪 [ATTENTION_MAP] Creating test data")
            attention_data = {
                'image': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
                'feature_importance': {
                    'Feature_1': 0.35,
                    'Feature_2': 0.25,
                    'Feature_3': 0.20,
                    'Feature_4': 0.15,
                    'Feature_5': 0.05
                },
                'temporal_importance': {
                    '2024-01-01': 0.1,
                    '2024-01-02': 0.2,
                    '2024-01-03': 0.3,
                    '2024-01-04': 0.4
                }
            }
        else:
            logger.warning("⚠️ [ATTENTION_MAP] No attention data available")
            return jsonify({
                'error': 'No attention map data available',
                'message': '예측을 먼저 실행해주세요. 예측 완료 후 어텐션 맵 데이터가 생성됩니다.',
                'suggestion': 'CSV 파일을 업로드하고 예측을 실행한 후 다시 시도해주세요.',
                'test_url': '/api/results/attention-map?test=true'
            }), 404
    
    logger.info(f"📊 [ATTENTION_MAP] Available keys: {list(attention_data.keys())}")
    
    # 어텐션 데이터 정리 및 직렬화
    cleaned_attention = {}
    
    try:
        for key, value in attention_data.items():
            if key == 'image' and value:
                cleaned_attention[key] = value  # base64 이미지는 그대로
                logger.info(f"📊 [ATTENTION_MAP] Image data preserved (length: {len(value) if isinstance(value, str) else 'N/A'})")
            elif isinstance(value, dict):
                cleaned_attention[key] = {}
                for k, v in value.items():
                    cleaned_attention[key][k] = safe_serialize_value(v)
                logger.info(f"📊 [ATTENTION_MAP] Dict processed for key '{key}': {len(cleaned_attention[key])} items")
            else:
                cleaned_attention[key] = safe_serialize_value(value)
                logger.info(f"📊 [ATTENTION_MAP] Value processed for key '{key}': {type(value)}")
        
        response_data = {
            'success': True,
            'attention_data': cleaned_attention,
            'current_date': safe_serialize_value(prediction_state.get('current_date')),
            'feature_importance': safe_serialize_value(prediction_state.get('feature_importance'))
        }
        
        # JSON 직렬화 테스트
        json.dumps(response_data)
        
        logger.info(f"✅ [ATTENTION_MAP] Response ready with keys: {list(cleaned_attention.keys())}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"💥 [ATTENTION_MAP] Error processing attention data: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Error processing attention map: {str(e)}'}), 500

@app.route('/api/features', methods=['GET'])
def get_features():
    """선택된 특성 조회 API"""
    global prediction_state
    
    if prediction_state['selected_features'] is None:
        return jsonify({'error': 'No feature information available'}), 404
    
    return jsonify({
        'success': True,
        'current_date': prediction_state['current_date'],
        'selected_features': prediction_state['selected_features'],
        'feature_importance': prediction_state['feature_importance']
    })

# 정적 파일 제공
@app.route('/static/<path:path>')
def serve_static(path):
    return send_file(os.path.join('static', path))

# 기본 라우트
@app.route('/')
def index():
    return jsonify({
        'app': 'MOPJ Prediction API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': [
            '/api/health',
            '/api/upload',
            '/api/holidays',
            '/api/holidays/upload',
            '/api/holidays/reload',
            '/api/file/metadata',
            '/api/data/dates',
            '/api/predict',
            '/api/predict/accumulated',
            '/api/predict/status',
            '/api/results',
            '/api/results/predictions',
            '/api/results/interval-scores',
            '/api/results/moving-averages',
            '/api/results/attention-map',
            '/api/results/accumulated',
            '/api/results/accumulated/interval-scores',
            '/api/results/accumulated/<date>',
            '/api/results/accumulated/report',
            '/api/results/accumulated/visualization',
            '/api/results/reliability',  # 새로 추가된 신뢰도 API
            '/api/features'
        ],
        'new_features': [
            'Prediction consistency scoring (예측 신뢰도)',
            'Purchase reliability percentage (구매 신뢰도)',
            'Holiday management system',
            'Accumulated predictions analysis'
        ]
    })

# 4. API 엔드포인트 추가 - 누적 예측 시작
@app.route('/api/predict/accumulated', methods=['POST'])
def start_accumulated_prediction():
    """여러 날짜에 대한 누적 예측 시작 API (저장/로드 기능 포함)"""
    from app.prediction.background_tasks import run_accumulated_predictions_with_save # ✅ 함수 내부에 추가
    global prediction_state
    
    if prediction_state['is_predicting']:
        return jsonify({
            'success': False,
            'error': 'Prediction already in progress',
            'progress': prediction_state['prediction_progress']
        }), 409
    
    data = request.json
    filepath = data.get('filepath')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    save_to_csv = data.get('save_to_csv', True)
    use_saved_data = data.get('use_saved_data', True)  # 저장된 데이터 활용 여부
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Invalid file path'}), 400
    
    if not start_date:
        return jsonify({'error': 'Start date is required'}), 400
    
    # 백그라운드에서 누적 예측 실행
    thread = Thread(target=run_accumulated_predictions_with_save, 
                   args=(filepath, start_date, end_date, save_to_csv, use_saved_data))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Accumulated prediction started',
        'save_to_csv': save_to_csv,
        'use_saved_data': use_saved_data,
        'status_url': '/api/predict/status'
    })

# 5. API 엔드포인트 추가 - 누적 예측 결과 조회
@app.route('/api/results/accumulated', methods=['GET'])
def get_accumulated_results():
    global prediction_state
    
    logger.info("🔍 [ACCUMULATED] API call received")
    
    if prediction_state['is_predicting']:
        logger.warning("⚠️ [ACCUMULATED] Prediction still in progress")
        return jsonify({
            'success': False,
            'error': 'Prediction in progress',
            'progress': prediction_state['prediction_progress']
        }), 409

    if not prediction_state['accumulated_predictions']:
        logger.error("❌ [ACCUMULATED] No accumulated prediction results available")
        return jsonify({'error': 'No accumulated prediction results available'}), 404

    logger.info("✅ [ACCUMULATED] Processing accumulated predictions...")
    
    # 누적 구매 신뢰도 계산 - 올바른 방식 사용
    accumulated_purchase_reliability, _ = calculate_accumulated_purchase_reliability(
        prediction_state['accumulated_predictions']
    )
    
    logger.info(f"💰 [ACCUMULATED] Purchase reliability calculated: {accumulated_purchase_reliability}")
    
    # ✅ 상세 디버깅 로깅 추가
    logger.info(f"🔍 [ACCUMULATED] Purchase reliability debugging:")
    logger.info(f"   - Type: {type(accumulated_purchase_reliability)}")
    logger.info(f"   - Value: {accumulated_purchase_reliability}")
    logger.info(f"   - Repr: {repr(accumulated_purchase_reliability)}")
    if accumulated_purchase_reliability == 100.0:
        logger.warning(f"⚠️ [ACCUMULATED] 100% reliability detected! Detailed analysis:")
        logger.warning(f"   - Total predictions: {len(prediction_state['accumulated_predictions'])}")
        for i, pred in enumerate(prediction_state['accumulated_predictions'][:3]):  # 처음 3개만
            logger.warning(f"   - Prediction {i+1}: date={pred.get('date')}, interval_scores_keys={list(pred.get('interval_scores', {}).keys())}")
    
    # 데이터 안전성 검사
    safe_interval_scores = []
    if prediction_state.get('accumulated_interval_scores'):
        safe_interval_scores = [
            item for item in prediction_state['accumulated_interval_scores'] 
            if item is not None and isinstance(item, dict) and 'days' in item and item['days'] is not None
        ]
        logger.info(f"📊 [ACCUMULATED] Safe interval scores count: {len(safe_interval_scores)}")
    else:
        logger.warning("⚠️ [ACCUMULATED] No accumulated_interval_scores found")
    
    consistency_scores = prediction_state.get('accumulated_consistency_scores', {})
    logger.info(f"🎯 [ACCUMULATED] Consistency scores keys: {list(consistency_scores.keys())}")
    
    # ✅ 캐시 통계 정보 추가
    cache_stats = prediction_state.get('cache_statistics', {
        'total_dates': 0,
        'cached_dates': 0,
        'new_predictions': 0,
        'cache_hit_rate': 0.0
    })
    
    # 🔧 NaN 값 처리 강화 - 누적 예측 결과 정리
    safe_predictions = clean_predictions_data(prediction_state['accumulated_predictions'])
    
    # 🔧 누적 메트릭스에서 NaN 값 제거
    safe_accumulated_metrics = {}
    for key, value in prediction_state.get('accumulated_metrics', {}).items():
        safe_accumulated_metrics[key] = safe_serialize_value(value)
    
    # 🔧 일관성 점수에서 NaN 값 제거
    safe_consistency_scores = {}
    for key, value in consistency_scores.items():
        safe_consistency_scores[key] = safe_serialize_value(value)
    
    # 🔧 구매 신뢰도 NaN 값 처리
    safe_purchase_reliability = safe_serialize_value(accumulated_purchase_reliability)
    if safe_purchase_reliability is None:
        safe_purchase_reliability = 0.0
    
    # 🔧 캐시 통계 NaN 값 처리
    safe_cache_stats = {}
    for key, value in cache_stats.items():
        safe_cache_stats[key] = safe_serialize_value(value)
    
    response_data = {
        'success': True,
        'prediction_dates': prediction_state.get('prediction_dates', []),
        'accumulated_metrics': safe_accumulated_metrics,
        'predictions': safe_predictions,
        'accumulated_interval_scores': safe_interval_scores,
        'accumulated_consistency_scores': safe_consistency_scores,
        'accumulated_purchase_reliability': safe_purchase_reliability,
        'cache_statistics': safe_cache_stats
    }
    
    # ✅ 최종 응답 데이터 검증 로깅
    logger.info(f"📤 [ACCUMULATED] Final response validation:")
    logger.info(f"   - accumulated_purchase_reliability in response: {response_data['accumulated_purchase_reliability']}")
    logger.info(f"   - Type in response: {type(response_data['accumulated_purchase_reliability'])}")
    
    logger.info(f"📤 [ACCUMULATED] Response summary: predictions={len(response_data['predictions'])}, metrics_keys={list(response_data['accumulated_metrics'].keys())}, reliability={response_data['accumulated_purchase_reliability']}")
    
    # 🔧 JSON 직렬화 테스트 및 NaN 값 강제 제거
    try:
        test_json = json.dumps(response_data)
        # 직렬화된 JSON에 NaN이 포함되어 있는지 추가 확인
        if 'NaN' in test_json or 'Infinity' in test_json:
            logger.error(f"🚨 [ACCUMULATED] JSON contains NaN/Infinity values")
            logger.error(f"   - JSON snippet: {test_json[:500]}...")
            test_json_cleaned = test_json.replace('NaN', 'null').replace('Infinity', 'null').replace('-Infinity', 'null')
            response_data = json.loads(test_json_cleaned)
            logger.info(f"✅ [ACCUMULATED] JSON NaN values cleaned successfully")
    except Exception as e:
        logger.error(f"❌ [ACCUMULATED] JSON serialization failed: {e}")
        logger.error(f"   - Error type: {type(e)}")
        logger.error(f"   - Error details: {str(e)}")
        # 추가 정리 시도
        for key, value in response_data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, float) and (np.isnan(sub_value) or np.isinf(sub_value)):
                        response_data[key][sub_key] = None
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        for sub_key, sub_value in item.items():
                            if isinstance(sub_value, float) and (np.isnan(sub_value) or np.isinf(sub_value)):
                                response_data[key][i][sub_key] = None
    
    # 🔧 Excel 프로세스 정리 (API 응답 전에 실행)
    cleanup_excel_processes()
    
    return jsonify(response_data)

@app.route('/api/results/accumulated/interval-scores', methods=['GET'])
def get_accumulated_interval_scores():
    global prediction_state
    scores = prediction_state.get('accumulated_interval_scores', [])
    
    # 'days' 속성이 없는 항목 필터링
    safe_scores = [
        item for item in scores 
        if item is not None and isinstance(item, dict) and 'days' in item and item['days'] is not None
    ]
    
    return jsonify(safe_scores)

# 7. 누적 보고서 API 엔드포인트
@app.route('/api/results/accumulated/report', methods=['GET'])
def get_accumulated_report():
    from app.visualization.plotter import generate_accumulated_report
    """누적 예측 결과 보고서 생성 및 다운로드 API"""
    global prediction_state
    
    # 예측 결과가 없는 경우
    if not prediction_state['accumulated_predictions']:
        return jsonify({'error': 'No accumulated prediction results available'}), 404
    
    report_file = generate_accumulated_report()
    if not report_file:
        return jsonify({'error': 'Failed to generate report'}), 500
    
    return send_file(report_file, as_attachment=True)

def return_prediction_result(pred, date, match_type):
    """
    예측 결과를 API 응답 형식으로 반환하는 헬퍼 함수
    
    Parameters:
    -----------
    pred : dict
        예측 결과 딕셔너리
    date : str
        요청된 날짜
    match_type : str
        매칭 방식 설명
    
    Returns:
    --------
    JSON response
    """
    try:
        logger.info(f"🔄 [API] Returning prediction result for date={date}, match_type={match_type}")
        
        # 예측 데이터 안전하게 추출
        predictions = pred.get('predictions', [])
        if not isinstance(predictions, list):
            logger.warning(f"⚠️ [API] predictions is not a list: {type(predictions)}")
            predictions = []
        
        # 구간 점수 안전하게 추출 및 변환
        interval_scores = pred.get('interval_scores', {})
        if isinstance(interval_scores, dict):
            # 딕셔너리를 리스트로 변환
            interval_scores_list = []
            for key, interval in interval_scores.items():
                if interval and isinstance(interval, dict) and 'days' in interval:
                    interval_scores_list.append(interval)
            interval_scores = interval_scores_list
        elif not isinstance(interval_scores, list):
            logger.warning(f"⚠️ [API] interval_scores is neither dict nor list: {type(interval_scores)}")
            interval_scores = []
        
        # 메트릭 안전하게 추출
        metrics = pred.get('metrics', {})
        if not isinstance(metrics, dict):
            logger.warning(f"⚠️ [API] metrics is not a dict: {type(metrics)}")
            metrics = {}
        
        # 🔄 이동평균 데이터 추출 (캐시된 데이터 또는 파일에서 로드)
        ma_results = pred.get('ma_results', {})
        if not ma_results:
            # 파일별 캐시에서 MA 파일 로드 시도
            try:
                current_file = prediction_state.get('current_file')
                if current_file:
                    cache_dirs = get_file_cache_dirs(current_file)
                    predictions_dir = cache_dirs['predictions']
                    
                    pred_start_date = pred.get('prediction_start_date') or date
                    ma_file_path = predictions_dir / f"prediction_start_{pd.to_datetime(pred_start_date).strftime('%Y%m%d')}_ma.json"
                else:
                    # 백업: 글로벌 캐시 사용
                    pred_start_date = pred.get('prediction_start_date') or date
                    ma_file_path = Path(PREDICTIONS_DIR) / f"prediction_start_{pd.to_datetime(pred_start_date).strftime('%Y%m%d')}_ma.json"
                
                if ma_file_path.exists():
                    with open(ma_file_path, 'r', encoding='utf-8') as f:
                        ma_results = json.load(f)
                    logger.info(f"📊 [API] MA results loaded from file for {date}: {len(ma_results)} windows")
                else:
                    logger.info(f"⚠️ [API] No MA file found for {date}: {ma_file_path}")
                    
                    # 파일이 없으면 예측 데이터에서 재계산 (히스토리컬 데이터 없이 제한적으로)
                    if predictions:
                        ma_results = calculate_moving_averages_with_history(
                            predictions, None, target_col='MOPJ', windows=[5, 10, 23]
                        )
                        logger.info(f"📊 [API] MA results recalculated for {date}: {len(ma_results)} windows")
            except Exception as e:
                logger.warning(f"⚠️ [API] Error loading/calculating MA for {date}: {str(e)}")
                ma_results = {}
        
        # 🎯 Attention 데이터 추출
        attention_data = pred.get('attention_data', {})
        if not attention_data:
            # 파일별 캐시에서 Attention 파일 로드 시도
            try:
                current_file = prediction_state.get('current_file')
                if current_file:
                    cache_dirs = get_file_cache_dirs(current_file)
                    predictions_dir = cache_dirs['predictions']
                    
                    pred_start_date = pred.get('prediction_start_date') or date
                    attention_file_path = predictions_dir / f"prediction_start_{pd.to_datetime(pred_start_date).strftime('%Y%m%d')}_attention.json"
                else:
                    # 백업: 글로벌 캐시 사용
                    pred_start_date = pred.get('prediction_start_date') or date
                    attention_file_path = Path(PREDICTIONS_DIR) / f"prediction_start_{pd.to_datetime(pred_start_date).strftime('%Y%m%d')}_attention.json"
                
                if attention_file_path.exists():
                    with open(attention_file_path, 'r', encoding='utf-8') as f:
                        attention_data = json.load(f)
                    logger.info(f"📊 [API] Attention data loaded from file for {date}")
                else:
                    logger.info(f"⚠️ [API] No attention file found for {date}: {attention_file_path}")
            except Exception as e:
                logger.warning(f"⚠️ [API] Error loading attention data for {date}: {str(e)}")
        
        # 기본 응답 데이터 구성
        response_data = {
            'success': True,
            'date': date,
            'predictions': predictions,
            'interval_scores': interval_scores,
            'metrics': metrics,
            'ma_results': ma_results,
            'attention_data': attention_data,
            'next_semimonthly_period': pred.get('next_semimonthly_period'),
            'actual_business_days': pred.get('actual_business_days'),
            'match_type': match_type,
            'data_end_date': pred.get('date'),
            'prediction_start_date': pred.get('prediction_start_date')
        }
        
        # 각 필드를 개별적으로 안전하게 직렬화
        safe_response = {}
        for key, value in response_data.items():
            safe_value = safe_serialize_value(value)
            if safe_value is not None:  # None이 아닌 경우에만 추가
                safe_response[key] = safe_value
        
        # success와 date는 항상 포함
        safe_response['success'] = True
        safe_response['date'] = date
        
        logger.info(f"✅ [API] Successfully prepared response for {date}: predictions={len(safe_response.get('predictions', []))}, interval_scores={len(safe_response.get('interval_scores', []))}, ma_windows={len(safe_response.get('ma_results', {}))}, attention_data={bool(safe_response.get('attention_data'))}")
        
        return jsonify(safe_response)
        
    except Exception as e:
        logger.error(f"💥 [API] Error in return_prediction_result for {date}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Error processing prediction result: {str(e)}',
            'date': date
        }), 500

# 8. API 엔드포인트 추가 - 특정 날짜 예측 결과 조회

@app.route('/api/results/accumulated/<date>', methods=['GET'])
def get_accumulated_result_by_date(date):
    """특정 날짜의 누적 예측 결과 조회 API"""
    global prediction_state
    
    logger.info(f"🔍 [API] Searching for accumulated result by date: {date}")
    
    if not prediction_state['accumulated_predictions']:
        logger.warning("❌ [API] No accumulated prediction results available")
        return jsonify({'error': 'No accumulated prediction results available'}), 404
    
    logger.info(f"📊 [API] Available prediction dates (data_end_date): {[p['date'] for p in prediction_state['accumulated_predictions']]}")
    
    # ✅ 1단계: 정확한 데이터 기준일 매칭 우선 확인
    logger.info(f"🔍 [API] Step 1: Looking for EXACT data_end_date match for {date}")
    for i, pred in enumerate(prediction_state['accumulated_predictions']):
        data_end_date = pred.get('date')  # 데이터 기준일
        
        logger.info(f"🔍 [API] Checking prediction {i+1}: data_end_date={data_end_date}")
        
        if data_end_date == date:
            logger.info(f"✅ [API] Found prediction by EXACT DATA END DATE match: {date}")
            logger.info(f"📊 [API] Prediction data preview: predictions={len(pred.get('predictions', []))}, interval_scores={len(pred.get('interval_scores', {}))}")
            return return_prediction_result(pred, date, "exact data end date")
    
    # ✅ 2단계: 정확한 매칭이 없으면 계산된 예측 시작일로 매칭
    logger.info(f"🔍 [API] Step 2: No exact match found. Looking for calculated prediction start date match for {date}")
    for i, pred in enumerate(prediction_state['accumulated_predictions']):
        data_end_date = pred.get('date')  # 데이터 기준일
        prediction_start_date = pred.get('prediction_start_date')  # 예측 시작일
        
        logger.info(f"🔍 [API] Checking prediction {i+1}: data_end_date={data_end_date}, prediction_start_date={prediction_start_date}")
        
        if data_end_date:
            try:
                data_end_dt = pd.to_datetime(data_end_date)
                calculated_start_date = data_end_dt + pd.Timedelta(days=1)
                
                # 주말과 휴일 건너뛰기
                while calculated_start_date.weekday() >= 5 or is_holiday(calculated_start_date):
                    calculated_start_date += pd.Timedelta(days=1)
                
                calculated_start_str = calculated_start_date.strftime('%Y-%m-%d')
                
                if calculated_start_str == date:
                    logger.info(f"✅ [API] Found prediction by CALCULATED PREDICTION START DATE: {date} (from data end date: {data_end_date})")
                    return return_prediction_result(pred, date, "calculated prediction start date from data end date")
                    
            except Exception as e:
                logger.warning(f"⚠️ [API] Error calculating prediction start date for {data_end_date}: {str(e)}")
                continue
    
    logger.error(f"❌ [API] No prediction results found for date {date}")
    return jsonify({'error': f'No prediction results for date {date}'}), 404

# 10. 누적 지표 시각화 API 엔드포인트
@app.route('/api/results/accumulated/visualization', methods=['GET'])
def get_accumulated_visualization():
    """누적 예측 지표 시각화 API"""
    global prediction_state
    
    if not prediction_state['accumulated_predictions']:
        return jsonify({'error': 'No accumulated prediction results available'}), 404
    
    filename, img_str = visualize_accumulated_metrics()
    if not filename:
        return jsonify({'error': 'Failed to generate visualization'}), 500
    
    return jsonify({
        'success': True,
        'file_path': filename,
        'image': img_str
    })

# 새로운 API 엔드포인트 추가
@app.route('/api/results/reliability', methods=['GET'])
def get_reliability_scores():
    """신뢰도 점수 조회 API"""
    global prediction_state
    
    # 단일 예측 신뢰도
    single_reliability = {}
    if prediction_state.get('latest_interval_scores') and prediction_state.get('latest_predictions'):
        try:
            # 실제 영업일 수 계산
            actual_business_days = len([p for p in prediction_state['latest_predictions'] 
                                       if p.get('Date') and not p.get('is_synthetic', False)])
            
            single_reliability = {
                'period': prediction_state['next_semimonthly_period']
            }
        except Exception as e:
            logger.error(f"Error calculating single prediction reliability: {str(e)}")
            single_reliability = {'error': 'Unable to calculate single prediction reliability'}
    
    # 누적 예측 신뢰도 (안전한 접근)
    accumulated_reliability = prediction_state.get('accumulated_consistency_scores', {})
    
    return jsonify({
        'success': True,
        'single_prediction_reliability': single_reliability,
        'accumulated_prediction_reliability': accumulated_reliability
    })

@app.route('/api/cache/clear/accumulated', methods=['POST'])
def clear_accumulated_cache():
    """누적 예측 캐시 클리어"""
    global prediction_state
    
    try:
        # 누적 예측 관련 상태 클리어
        prediction_state['accumulated_predictions'] = []
        prediction_state['accumulated_metrics'] = {}
        prediction_state['accumulated_interval_scores'] = []
        prediction_state['accumulated_consistency_scores'] = {}
        prediction_state['accumulated_purchase_reliability'] = 0
        prediction_state['prediction_dates'] = []
        
        logger.info("🧹 [CACHE] Accumulated prediction cache cleared")
        
        return jsonify({
            'success': True,
            'message': 'Accumulated prediction cache cleared successfully'
        })
        
    except Exception as e:
        logger.error(f"❌ [CACHE] Error clearing accumulated cache: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/reliability', methods=['GET'])
def debug_reliability_calculation():
    """구매 신뢰도 계산 디버깅 API"""
    global prediction_state
    
    if not prediction_state['accumulated_predictions']:
        return jsonify({'error': 'No accumulated prediction results available'}), 404
    
    predictions = prediction_state['accumulated_predictions']
    print(f"🔍 [DEBUG] Total predictions: {len(predictions)}")
    
    debug_data = {
        'prediction_count': len(predictions),
        'predictions_details': []
    }
    
    total_score = 0
    
    for i, pred in enumerate(predictions):
        pred_date = pred.get('date')
        interval_scores = pred.get('interval_scores', {})
        
        print(f"📊 [DEBUG] Prediction {i+1} ({pred_date}):")
        print(f"   - interval_scores type: {type(interval_scores)}")
        print(f"   - interval_scores keys: {list(interval_scores.keys()) if isinstance(interval_scores, dict) else 'N/A'}")
        
        pred_detail = {
            'date': pred_date,
            'interval_scores_type': str(type(interval_scores)),
            'interval_scores_keys': list(interval_scores.keys()) if isinstance(interval_scores, dict) else [],
            'individual_scores': [],
            'best_score': 0
        }
        
        if isinstance(interval_scores, dict):
            for key, score_data in interval_scores.items():
                print(f"   - {key}: {score_data}")
                if isinstance(score_data, dict) and 'score' in score_data:
                    score_value = score_data.get('score', 0)
                    pred_detail['individual_scores'].append({
                        'key': key,
                        'score': score_value,
                        'full_data': score_data
                    })
                    print(f"     -> score: {score_value}")
        
        if pred_detail['individual_scores']:
            best_score = max([s['score'] for s in pred_detail['individual_scores']])
            # 점수를 3점으로 제한
            capped_score = min(best_score, 3.0)
            pred_detail['best_score'] = best_score
            pred_detail['capped_score'] = capped_score
            total_score += capped_score
            print(f"   - Best score: {best_score:.1f}, Capped score: {capped_score:.1f}")
        
        debug_data['predictions_details'].append(pred_detail)
    
    max_possible_score = len(predictions) * 3
    reliability = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
    
    debug_data.update({
        'total_score': total_score,
        'max_possible_score': max_possible_score,
        'reliability_percentage': reliability
    })
    
    print(f"🎯 [DEBUG] CALCULATION SUMMARY:")
    print(f"   - Total predictions: {len(predictions)}")
    print(f"   - Total score: {total_score}")
    print(f"   - Max possible score: {max_possible_score}")
    print(f"   - Reliability: {reliability:.1f}%")
    
    return jsonify(debug_data)

@app.route('/api/cache/check', methods=['POST'])
def check_cached_predictions():
    """누적 예측 범위에서 캐시된 예측이 얼마나 있는지 확인"""
    data = request.get_json()
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    if not start_date or not end_date:
        return jsonify({'error': 'start_date and end_date are required'}), 400
    
    try:
        logger.info(f"🔍 [CACHE_CHECK] Checking cache availability for {start_date} to {end_date}")
        
        # 저장된 예측 확인
        cached_predictions = load_accumulated_predictions_from_csv(start_date, end_date)
        
        # 전체 범위 계산 (데이터 기준일 기준)
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # 사용 가능한 날짜 계산 (데이터 기준일)
        available_dates = []
        current_dt = start_dt
        while current_dt <= end_dt:
            # 영업일만 포함 (주말과 휴일 제외)
            if current_dt.weekday() < 5 and not is_holiday(current_dt):
                available_dates.append(current_dt.strftime('%Y-%m-%d'))
            current_dt += pd.Timedelta(days=1)
        
        # 캐시된 날짜 목록
        cached_dates = [pred['date'] for pred in cached_predictions]
        missing_dates = [date for date in available_dates if date not in cached_dates]
        
        cache_percentage = round(len(cached_predictions) / max(len(available_dates), 1) * 100, 1)
        
        logger.info(f"📊 [CACHE_CHECK] Cache status: {len(cached_predictions)}/{len(available_dates)} ({cache_percentage}%)")
        
        return jsonify({
            'success': True,
            'total_dates_in_range': len(available_dates),
            'cached_predictions': len(cached_predictions),
            'cached_dates': cached_dates,
            'missing_dates': missing_dates,
            'cache_percentage': cache_percentage,
            'will_use_cache': len(cached_predictions) > 0,
            'estimated_time_savings': f"약 {len(cached_predictions) * 3}분 절약 예상" if len(cached_predictions) > 0 else "없음"
        })
        
    except Exception as e:
        logger.error(f"❌ [CACHE_CHECK] Error checking cached predictions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/accumulated/recent', methods=['GET'])
def get_recent_accumulated_results():
    """
    페이지 로드 시 최근 누적 예측 결과를 자동으로 복원하는 API
    """
    try:
        # 저장된 예측 목록 조회 (최근 것부터)
        predictions_list = get_saved_predictions_list(limit=50)
        
        if not predictions_list:
            return jsonify({
                'success': False, 
                'message': 'No saved predictions found',
                'has_recent_results': False
            })
        
        # 날짜별로 그룹화하여 연속된 범위 찾기
        dates_by_groups = {}
        for pred in predictions_list:
            data_end_date = pred.get('data_end_date', pred.get('prediction_date'))
            if data_end_date:
                date_obj = pd.to_datetime(data_end_date)
                # 주차별로 그룹화 (같은 주의 예측들을 하나의 범위로 간주)
                week_key = date_obj.strftime('%Y-W%U')
                if week_key not in dates_by_groups:
                    dates_by_groups[week_key] = []
                dates_by_groups[week_key].append({
                    'date': data_end_date,
                    'date_obj': date_obj,
                    'pred_info': pred
                })
        
        # 가장 최근 그룹 선택
        if not dates_by_groups:
            return jsonify({
                'success': False, 
                'message': 'No valid date groups found',
                'has_recent_results': False
            })
        
        # 최근 주의 예측들 가져오기
        latest_week = max(dates_by_groups.keys())
        latest_group = dates_by_groups[latest_week]
        latest_group.sort(key=lambda x: x['date_obj'])
        
        # 연속된 날짜 범위 찾기
        start_date = latest_group[0]['date_obj']
        end_date = latest_group[-1]['date_obj']
        
        logger.info(f"🔄 [AUTO_RESTORE] Found recent accumulated predictions: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        # 기존 캐시에서 누적 결과 로드
        loaded_predictions = load_accumulated_predictions_from_csv(start_date, end_date)
        
        if not loaded_predictions:
            return jsonify({
                'success': False, 
                'message': 'Failed to load cached predictions',
                'has_recent_results': False
            })
        
        # 누적 메트릭 계산
        accumulated_metrics = {
            'f1': 0.0,
            'accuracy': 0.0,
            'mape': 0.0,
            'weighted_score': 0.0,
            'total_predictions': 0
        }
        
        for pred in loaded_predictions:
            metrics = pred.get('metrics', {})
            if isinstance(metrics, dict):
                accumulated_metrics['f1'] += metrics.get('f1', 0.0)
                accumulated_metrics['accuracy'] += metrics.get('accuracy', 0.0)
                accumulated_metrics['mape'] += metrics.get('mape', 0.0)
                accumulated_metrics['weighted_score'] += metrics.get('weighted_score', 0.0)
                accumulated_metrics['total_predictions'] += 1
        
        if accumulated_metrics['total_predictions'] > 0:
            count = accumulated_metrics['total_predictions']
            accumulated_metrics['f1'] /= count
            accumulated_metrics['accuracy'] /= count
            accumulated_metrics['mape'] /= count
            accumulated_metrics['weighted_score'] /= count
            
            # 🔧 NaN 값 처리 강화
            for metric_key in ['f1', 'accuracy', 'mape', 'weighted_score']:
                if pd.isna(accumulated_metrics[metric_key]) or np.isnan(accumulated_metrics[metric_key]) or np.isinf(accumulated_metrics[metric_key]):
                    logger.warning(f"⚠️ [CACHED_METRICS] NaN/Inf detected in {metric_key}, setting to 0.0")
                    accumulated_metrics[metric_key] = 0.0
        
        # 구간 점수 계산
        accumulated_interval_scores = {}
        for pred in loaded_predictions:
            interval_scores = pred.get('interval_scores', {})
            if isinstance(interval_scores, dict):
                for interval in interval_scores.values():
                    if not interval or not isinstance(interval, dict) or 'days' not in interval or interval['days'] is None:
                        continue
                    interval_key = f"{format_date(interval['start_date'])}-{format_date(interval['end_date'])}"
                    if interval_key in accumulated_interval_scores:
                        accumulated_interval_scores[interval_key]['score'] += interval['score']
                        accumulated_interval_scores[interval_key]['count'] += 1
                    else:
                        accumulated_interval_scores[interval_key] = interval.copy()
                        accumulated_interval_scores[interval_key]['count'] = 1
        
        # 정렬된 구간 점수 리스트
        accumulated_scores_list = [
            v for v in accumulated_interval_scores.values()
            if v is not None and isinstance(v, dict) and 'days' in v and v['days'] is not None
        ]
        accumulated_scores_list.sort(key=lambda x: x['score'], reverse=True)
        
        # 구매 신뢰도 계산
        accumulated_purchase_reliability, _ = calculate_accumulated_purchase_reliability(loaded_predictions)
        
        # 일관성 점수 계산
        unique_periods = set()
        for pred in loaded_predictions:
            if 'next_semimonthly_period' in pred and pred['next_semimonthly_period']:
                unique_periods.add(pred['next_semimonthly_period'])
        
        accumulated_consistency_scores = {}
        for period in unique_periods:
            try:
                consistency_data = calculate_prediction_consistency(loaded_predictions, period)
                # 🔧 NaN 값 처리 강화
                if consistency_data and 'consistency_score' in consistency_data:
                    consistency_score = consistency_data['consistency_score']
                    if pd.isna(consistency_score) or np.isnan(consistency_score) or np.isinf(consistency_score):
                        logger.warning(f"⚠️ [CACHED_CONSISTENCY] NaN/Inf detected for period {period}, setting to 0.0")
                        consistency_data['consistency_score'] = 0.0
                accumulated_consistency_scores[period] = consistency_data
            except Exception as e:
                logger.error(f"Error calculating consistency for period {period}: {str(e)}")
        
        # 캐시 통계
        cache_statistics = {
            'total_dates': len(loaded_predictions),
            'cached_dates': len(loaded_predictions),
            'new_predictions': 0,
            'cache_hit_rate': 100.0
        }
        
        # 🔧 NaN 값 처리 강화
        if pd.isna(cache_statistics['cache_hit_rate']) or np.isnan(cache_statistics['cache_hit_rate']) or np.isinf(cache_statistics['cache_hit_rate']):
            logger.warning(f"⚠️ [CACHED_CACHE_STATS] NaN/Inf detected in cache_hit_rate, setting to 0.0")
            cache_statistics['cache_hit_rate'] = 0.0
        
        # 전역 상태 업데이트 (선택적)
        global prediction_state
        prediction_state['accumulated_predictions'] = loaded_predictions
        prediction_state['accumulated_metrics'] = accumulated_metrics
        prediction_state['prediction_dates'] = [p['date'] for p in loaded_predictions]
        prediction_state['accumulated_interval_scores'] = accumulated_scores_list
        prediction_state['accumulated_consistency_scores'] = accumulated_consistency_scores
        prediction_state['accumulated_purchase_reliability'] = accumulated_purchase_reliability
        prediction_state['cache_statistics'] = cache_statistics
        
        logger.info(f"✅ [AUTO_RESTORE] Successfully restored {len(loaded_predictions)} accumulated predictions")
        
        return jsonify({
            'success': True,
            'has_recent_results': True,
            'restored_range': {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'prediction_count': len(loaded_predictions)
            },
            'prediction_dates': [p['date'] for p in loaded_predictions],
            'accumulated_metrics': accumulated_metrics,
            'predictions': loaded_predictions,
            'accumulated_interval_scores': accumulated_scores_list,
            'accumulated_consistency_scores': accumulated_consistency_scores,
            'accumulated_purchase_reliability': accumulated_purchase_reliability,
            'cache_statistics': cache_statistics,
            'message': f"최근 누적 예측 결과를 자동으로 복원했습니다 ({start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')})"
        })
        
    except Exception as e:
        logger.error(f"❌ [AUTO_RESTORE] Error restoring recent accumulated results: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False, 
            'error': str(e),
            'has_recent_results': False
        }), 500

@app.route('/api/cache/rebuild-index', methods=['POST'])
def rebuild_predictions_index_api():
    """예측 인덱스 재생성 API (rebuild_index.py 기능을 통합)"""
    try:
        # 현재 파일의 캐시 디렉토리 가져오기
        current_file = prediction_state.get('current_file')
        if not current_file:
            return jsonify({'success': False, 'error': '현재 업로드된 파일이 없습니다. 먼저 파일을 업로드해주세요.'})
        
        # 🔧 새로운 rebuild 함수 사용
        success = rebuild_predictions_index_from_existing_files()
        
        if success:
            cache_dirs = get_file_cache_dirs(current_file)
            index_file = cache_dirs['predictions'] / 'predictions_index.cs'
            
            # 결과 데이터 읽기
            index_data = []
            if index_file.exists():
                with open(index_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    index_data = list(reader)
            
            return jsonify({
                'success': True,
                'message': f'인덱스 파일을 성공적으로 재생성했습니다. ({len(index_data)}개 항목)',
                'file_location': str(index_file),
                'entries_count': len(index_data),
                'rebuilt_entries': [{'date': row.get('prediction_start_date', ''), 'data_end': row.get('data_end_date', '')} for row in index_data]
            })
        else:
            return jsonify({
                'success': False,
                'error': '인덱스 재생성에 실패했습니다. 로그를 확인해주세요.'
            })
        
    except Exception as e:
        logger.error(f"❌ Error rebuilding predictions index: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': f'인덱스 재생성 중 오류 발생: {str(e)}'})

@app.route('/api/cache/clear/semimonthly', methods=['POST'])
def clear_semimonthly_cache():
    """특정 반월 기간의 캐시만 삭제하는 API"""
    try:
        data = request.json
        target_date = data.get('date')
        
        if not target_date:
            return jsonify({'error': 'Date parameter is required'}), 400
        
        target_date = pd.to_datetime(target_date)
        target_semimonthly = get_semimonthly_period(target_date)
        
        logger.info(f"🗑️ [API] Clearing cache for semimonthly period: {target_semimonthly}")
        
        # 현재 파일의 캐시 디렉토리에서 해당 반월 캐시 삭제
        cache_dirs = get_file_cache_dirs()
        predictions_dir = cache_dirs['predictions']
        
        deleted_files = []
        
        if predictions_dir.exists():
            # 메타 파일 확인하여 반월 기간이 일치하는 캐시 삭제
            for meta_file in predictions_dir.glob("*_meta.json"):
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                    
                    cached_data_end_date = meta_data.get('data_end_date')
                    if cached_data_end_date:
                        cached_data_end_date = pd.to_datetime(cached_data_end_date)
                        cached_semimonthly = get_semimonthly_period(cached_data_end_date)
                        
                        if cached_semimonthly == target_semimonthly:
                            # 관련 파일들 삭제 (보안을 위해 .cs 확장자 사용)
                            base_name = meta_file.stem.replace('_meta', '')
                            files_to_delete = [
                                meta_file,
                                meta_file.parent / f"{base_name}.cs",
                                meta_file.parent / f"{base_name}_attention.json",
                                meta_file.parent / f"{base_name}_ma.json"
                            ]
                            
                            for file_path in files_to_delete:
                                if file_path.exists():
                                    file_path.unlink()
                                    deleted_files.append(str(file_path.name))
                                    logger.info(f"  🗑️ Deleted: {file_path.name}")
                            
                except Exception as e:
                    logger.warning(f"⚠️ Error processing meta file {meta_file}: {str(e)}")
                    continue
        
        return jsonify({
            'success': True,
            'message': f'Cache cleared for semimonthly period: {target_semimonthly}',
            'target_semimonthly': target_semimonthly,
            'target_date': target_date.strftime('%Y-%m-%d'),
            'deleted_files': deleted_files,
            'deleted_count': len(deleted_files)
        })
        
    except Exception as e:
        logger.error(f"❌ [API] Error clearing semimonthly cache: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

#######################################################################
# VARMAX 예측 저장/로드 시스템
#######################################################################

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

def background_varmax_prediction(file_path, current_date, pred_days, use_cache=True):
    """백그라운드에서 VARMAX 예측 작업을 수행하는 함수"""
    global prediction_state
    
    try:
        from app.utils.file_utils import set_seed
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

def plot_varmax_prediction_basic(sequence_df, sequence_start_date, start_day_value, 
                                f1, accuracy, mape, weighted_score, 
                                save_prefix=None, title_prefix="VARMAX Semi-monthly Prediction", file_path=None):
    """VARMAX 기본 예측 그래프 시각화 (기존 plot_prediction_basic과 동일한 스타일)"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import matplotlib.dates as mdates
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
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
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

#######################################################################
# VARMAX API 엔드포인트
#######################################################################

# 1) VARMAX 반월별 예측 시작
@app.route('/api/varmax/predict', methods=['POST', 'OPTIONS'])
def varmax_semimonthly_predict():
    from app.prediction.background_tasks import background_varmax_prediction # ✅ 함수 내부에 추가
    """VARMAX 반월별 예측 시작 API"""
    # 1) 먼저, OPTIONS(preflight) 요청이 들어오면 바로 200을 리턴
    if request.method == 'OPTIONS':
        # CORS(app) 로 설정해뒀으면 이미 Access-Control-Allow-Origin 등이 붙어 있을 것입니다.
        return make_response(('', 200))
    global prediction_state
    
    # 🔧 VARMAX 독립 상태 확인 - hang된 상태면 자동 리셋
    if prediction_state.get('varmax_is_predicting', False):
        current_progress = prediction_state.get('varmax_prediction_progress', 0)
        current_error = prediction_state.get('varmax_error')
        
        logger.warning(f"⚠️ [VARMAX_API] Prediction already in progress (progress: {current_progress}%, error: {current_error})")
        
        # 🔧 개선된 자동 리셋 조건: 에러가 있거나 진행률이 매우 낮은 경우만 리셋
        should_reset = False
        reset_reason = ""
        
        if current_error:
            should_reset = True
            reset_reason = f"error detected: {current_error}"
        elif current_progress > 0 and current_progress < 15:
            should_reset = True  
            reset_reason = f"very low progress stuck: {current_progress}%"
        
        if should_reset:
            logger.warning(f"🔄 [VARMAX_API] Auto-resetting stuck prediction - {reset_reason}")
            
            # 상태 리셋
            prediction_state['varmax_is_predicting'] = False
            prediction_state['varmax_prediction_progress'] = 0
            prediction_state['varmax_error'] = None
            prediction_state['varmax_predictions'] = []
            prediction_state['varmax_half_month_averages'] = []
            prediction_state['varmax_metrics'] = {}
            prediction_state['varmax_ma_results'] = {}
            prediction_state['varmax_selected_features'] = []
            prediction_state['varmax_current_date'] = None
            prediction_state['varmax_model_info'] = {}
            prediction_state['varmax_plots'] = {}
            
            logger.info(f"✅ [VARMAX_API] Stuck state auto-reset completed, proceeding with new prediction")
        else:
            # 정상적으로 진행 중인 경우 409 반환
            return jsonify({
                'success': False,
                'error': 'VARMAX prediction already in progress',
                'progress': current_progress
            }), 409
    
    data = request.get_json(force=True)
    filepath     = data.get('filepath')
    current_date = data.get('date')
    pred_days    = data.get('pred_days', 50)
    use_cache    = data.get('use_cache', True)  # 🆕 캐시 사용 여부 (기본값: True)
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Invalid file path'}), 400
    if not current_date:
        return jsonify({'error': 'Date is required'}), 400
    
    logger.info(f"🚀 [VARMAX_API] Starting VARMAX prediction (use_cache={use_cache}) for {current_date}")
    
    thread = Thread(
        target=background_varmax_prediction,
        args=(filepath, current_date, pred_days, use_cache)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'VARMAX semi-monthly prediction started',
        'status_url': '/api/varmax/status',
        'use_cache': use_cache
    })

# 2) VARMAX 예측 상태 조회
@app.route('/api/varmax/status', methods=['GET'])
def varmax_prediction_status():
    from app.prediction.background_tasks import calculate_estimated_time_remaining # ✅ 함수 내부에 추가
    """VARMAX 예측 상태 확인 API (남은 시간 추가)"""
    global prediction_state
    
    is_predicting = prediction_state.get('varmax_is_predicting', False)
    progress = prediction_state.get('varmax_prediction_progress', 0)
    error = prediction_state.get('varmax_error', None)
    
    logger.info(f"🔍 [VARMAX_STATUS] Current status - predicting: {is_predicting}, progress: {progress}%, error: {error}")
    
    status = {
        'is_predicting': is_predicting,
        'progress': progress,
        'error': error
    }
    
    # VARMAX 예측 중인 경우 남은 시간 계산
    if is_predicting and prediction_state.get('varmax_prediction_start_time'):
        time_info = calculate_estimated_time_remaining(
            prediction_state['varmax_prediction_start_time'], 
            progress
        )
        status.update(time_info)
    
    if not is_predicting and prediction_state.get('varmax_current_date'):
        status['current_date'] = prediction_state['varmax_current_date']
        logger.info(f"🔍 [VARMAX_STATUS] Prediction completed for date: {status['current_date']}")
    
    return jsonify(status)

# 3) VARMAX 전체 결과 조회
@app.route('/api/varmax/results', methods=['GET'])
def get_varmax_results():
    """VARMAX 예측 결과 조회 API"""
    global prediction_state
    
    # 🔍 상태 디버깅
    logger.info(f"🔍 [VARMAX_API] Current prediction_state keys: {list(prediction_state.keys())}")
    logger.info(f"🔍 [VARMAX_API] varmax_is_predicting: {prediction_state.get('varmax_is_predicting', 'NOT_SET')}")
    logger.info(f"🔍 [VARMAX_API] varmax_predictions available: {bool(prediction_state.get('varmax_predictions'))}")
    logger.info(f"🔍 [VARMAX_API] varmax_ma_results available: {bool(prediction_state.get('varmax_ma_results'))}")
    
    if prediction_state.get('varmax_predictions'):
        logger.info(f"🔍 [VARMAX_API] Predictions count: {len(prediction_state['varmax_predictions'])}")
    
    if prediction_state.get('varmax_ma_results'):
        logger.info(f"🔍 [VARMAX_API] MA results keys: {list(prediction_state['varmax_ma_results'].keys())}")
    
    # 🛡️ 백그라운드 스레드 완료 대기
    if prediction_state.get('varmax_is_predicting', False):
        logger.warning(f"⚠️ [VARMAX_API] Prediction still in progress: {prediction_state.get('varmax_prediction_progress', 0)}%")
        return jsonify({
            'success': False,
            'error': 'VARMAX prediction in progress',
            'progress': prediction_state.get('varmax_prediction_progress', 0)
        }), 409
    
    # 🎯 상태에 데이터가 없으면 캐시에서 직접 로드 (신뢰성 개선)
    if not prediction_state.get('varmax_predictions'):
        logger.warning(f"⚠️ [VARMAX_API] No VARMAX predictions in state, attempting direct cache load")
        logger.info(f"🔍 [VARMAX_API] Current file: {prediction_state.get('current_file')}")
        
        try:
            # 최근 저장된 VARMAX 예측 목록 가져오기
            saved_predictions = get_saved_varmax_predictions_list(limit=1)
            logger.info(f"🔍 [VARMAX_API] Found {len(saved_predictions)} saved predictions")
            
            if saved_predictions:
                latest_date = saved_predictions[0]['prediction_date']
                logger.info(f"🔧 [VARMAX_API] Loading latest prediction: {latest_date}")
                
                # 직접 로드하고 상태 복원
                cached_prediction = load_varmax_prediction(latest_date)
                if cached_prediction and cached_prediction.get('predictions'):
                    logger.info(f"✅ [VARMAX_API] Successfully loaded from cache ({len(cached_prediction.get('predictions', []))} predictions)")
                    
                    # 🔑 즉시 상태 복원 (더 안전하게)
                    prediction_state['varmax_predictions'] = cached_prediction.get('predictions', [])
                    prediction_state['varmax_half_month_averages'] = cached_prediction.get('half_month_averages', [])
                    prediction_state['varmax_metrics'] = cached_prediction.get('metrics', {})
                    prediction_state['varmax_ma_results'] = cached_prediction.get('ma_results', {})
                    prediction_state['varmax_selected_features'] = cached_prediction.get('selected_features', [])
                    prediction_state['varmax_current_date'] = cached_prediction.get('current_date')
                    prediction_state['varmax_model_info'] = cached_prediction.get('model_info', {})
                    prediction_state['varmax_plots'] = cached_prediction.get('plots', {})
                    
                    logger.info(f"🎯 [VARMAX_API] State restored from cache - {len(prediction_state['varmax_predictions'])} predictions")
                    
                    return jsonify({
                        'success': True,
                        'current_date': cached_prediction.get('current_date'),
                        'predictions': cached_prediction.get('predictions', []),
                        'half_month_averages': cached_prediction.get('half_month_averages', []),
                        'metrics': cached_prediction.get('metrics', {}),
                        'ma_results': cached_prediction.get('ma_results', {}),
                        'selected_features': cached_prediction.get('selected_features', []),
                        'model_info': cached_prediction.get('model_info', {}),
                        'plots': cached_prediction.get('plots', {})
                    })
                else:
                    logger.warning(f"⚠️ [VARMAX_API] Cached prediction is empty or invalid")
            else:
                logger.warning(f"⚠️ [VARMAX_API] No saved predictions found")
                
        except Exception as e:
            logger.error(f"❌ [VARMAX_API] Direct cache load failed: {e}")
            import traceback
            logger.error(f"❌ [VARMAX_API] Cache load traceback: {traceback.format_exc()}")
        
        # 캐시 로드도 실패한 경우 명확한 메시지
        logger.error(f"❌ [VARMAX_API] No VARMAX results available in state or cache")
        return jsonify({
            'success': False,
            'error': 'No VARMAX prediction results available. Please run a new prediction.'
        }), 404
    
    logger.info(f"✅ [VARMAX_API] Returning VARMAX results successfully from state")
    return jsonify({
        'success': True,
        'current_date':      prediction_state.get('varmax_current_date'),
        'predictions':       prediction_state.get('varmax_predictions', []),
        'half_month_averages': prediction_state.get('varmax_half_month_averages', []),
        'metrics':           prediction_state.get('varmax_metrics', {}),
        'ma_results':        prediction_state.get('varmax_ma_results', {}),
        'selected_features': prediction_state.get('varmax_selected_features', []),
        'model_info':        prediction_state.get('varmax_model_info', {}),
        'plots':             prediction_state.get('varmax_plots', {})
    })

# 4) VARMAX 예측값만 조회
@app.route('/api/varmax/predictions', methods=['GET'])
def get_varmax_predictions_only():
    """VARMAX 예측 값만 조회 API"""
    global prediction_state
    
    if not prediction_state.get('varmax_predictions'):
        return jsonify({'error': 'No VARMAX prediction results available'}), 404
    
    return jsonify({
        'success': True,
        'current_date':      prediction_state['varmax_current_date'],
        'predictions':       prediction_state['varmax_predictions'],
        'model_info':        prediction_state['varmax_model_info']
    })

# 5) VARMAX 이동평균 조회 - 즉석 계산 방식
@app.route('/api/varmax/moving-averages', methods=['GET'])
def get_varmax_moving_averages():
    """VARMAX 이동평균 조회 API - 예측 결과로 즉석 계산"""
    global prediction_state
    
    # 🎯 상태에 MA 데이터가 있으면 바로 반환
    if prediction_state.get('varmax_ma_results'):
        return jsonify({
            'success': True,
            'current_date': prediction_state['varmax_current_date'],
            'ma_results': prediction_state['varmax_ma_results']
        })
    
    # 🚀 예측 결과가 있으면 즉석에서 MA 계산
    varmax_predictions = prediction_state.get('varmax_predictions')
    current_date = prediction_state.get('varmax_current_date')
    current_file = prediction_state.get('current_file')
    
    # 상태에 예측 결과가 없으면 캐시에서 로드
    if not varmax_predictions or not current_date:
        logger.info(f"🔧 [VARMAX_MA_API] No predictions in state, loading from cache")
        try:
            saved_predictions = get_saved_varmax_predictions_list(limit=1)
            if saved_predictions:
                latest_date = saved_predictions[0]['prediction_date']
                cached_prediction = load_varmax_prediction(latest_date)
                if cached_prediction and cached_prediction.get('predictions'):
                    varmax_predictions = cached_prediction.get('predictions')
                    current_date = cached_prediction.get('current_date', latest_date)
                    # current_file은 prediction_state에서 가져오거나 추정
                    if not current_file:
                        current_file = prediction_state.get('current_file')
                    logger.info(f"✅ [VARMAX_MA_API] Loaded predictions from cache: {len(varmax_predictions)} items")
        except Exception as e:
            logger.error(f"❌ [VARMAX_MA_API] Failed to load from cache: {e}")
    
    # 예측 결과가 없으면 에러
    if not varmax_predictions or not current_date:
        return jsonify({
            'success': False,
            'error': 'No VARMAX predictions available for MA calculation'
        }), 404
    
    # 🎯 즉석에서 MA 계산
    try:
        logger.info(f"🔄 [VARMAX_MA_API] Calculating MA on-the-fly for {len(varmax_predictions)} predictions")
        
        # VARMAX 클래스 인스턴스 생성 (MA 계산용)
        if not current_file or not os.path.exists(current_file):
            logger.error(f"❌ [VARMAX_MA_API] File not found: {current_file}")
            return jsonify({
                'success': False,
                'error': 'Original data file not available for MA calculation'
            }), 404
            
        forecaster = VARMAXSemiMonthlyForecaster(current_file, pred_days=50)
        forecaster.load_data()  # 과거 데이터 로드
        
        # MA 계산
        ma_results = forecaster.calculate_moving_averages_varmax(
            varmax_predictions, 
            current_date, 
            windows=[5, 10, 20, 30]
        )
        
        logger.info(f"✅ [VARMAX_MA_API] MA calculation completed: {len(ma_results)} windows")
        
        # 상태에 저장 (다음번 요청을 위해)
        prediction_state['varmax_ma_results'] = ma_results
        
        return jsonify({
            'success': True,
            'current_date': current_date,
            'ma_results': ma_results
        })
        
    except Exception as e:
        logger.error(f"❌ [VARMAX_MA_API] MA calculation failed: {e}")
        logger.error(f"❌ [VARMAX_MA_API] Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'MA calculation failed: {str(e)}'
        }), 500

# 6) VARMAX 의사결정 조회
@app.route('/api/varmax/saved', methods=['GET'])
def get_saved_varmax_predictions():
    """저장된 VARMAX 예측 목록을 반환하는 API"""
    try:
        limit = request.args.get('limit', 100, type=int)
        predictions = get_saved_varmax_predictions_list(limit=limit)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'count': len(predictions)
        })
        
    except Exception as e:
        logger.error(f"Error getting saved VARMAX predictions: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/varmax/saved/<date>', methods=['GET'])
def get_saved_varmax_prediction_by_date(date):
    """특정 날짜의 저장된 VARMAX 예측을 반환하는 API"""
    global prediction_state
    
    try:
        prediction = load_varmax_prediction(date)
        
        if prediction is None:
            return jsonify({
                'success': False,
                'error': f'Prediction not found for date: {date}'
            }), 404
        
        # 🔍 로드된 예측 데이터 타입 확인
        logger.info(f"🔍 [VARMAX_API_LOAD] Prediction data type: {type(prediction)}")
        
        if not isinstance(prediction, dict):
            logger.error(f"❌ [VARMAX_API_LOAD] Prediction is not a dictionary: {type(prediction)}")
            return jsonify({
                'success': False,
                'error': f'Invalid prediction data format: expected dict, got {type(prediction).__name__}'
            }), 500
        
        # 🔧 백엔드 prediction_state 복원
        logger.info(f"🔄 [VARMAX_LOAD] Restoring prediction_state for date: {date}")
        logger.info(f"🔍 [VARMAX_LOAD] Available prediction keys: {list(prediction.keys())}")
        
        # VARMAX 상태 복원 (안전한 접근)
        prediction_state['varmax_is_predicting'] = False
        prediction_state['varmax_prediction_progress'] = 100
        prediction_state['varmax_error'] = None
        prediction_state['varmax_current_date'] = prediction.get('current_date', date)
        prediction_state['varmax_predictions'] = prediction.get('predictions', [])
        prediction_state['varmax_half_month_averages'] = prediction.get('half_month_averages', [])
        prediction_state['varmax_metrics'] = prediction.get('metrics', {})
        prediction_state['varmax_ma_results'] = prediction.get('ma_results', {})
        prediction_state['varmax_selected_features'] = prediction.get('selected_features', [])
        prediction_state['varmax_model_info'] = prediction.get('model_info', {})
        prediction_state['varmax_plots'] = prediction.get('plots', {})
        
        logger.info(f"✅ [VARMAX_LOAD] prediction_state restored successfully")
        logger.info(f"🔍 [VARMAX_LOAD] Restored predictions count: {len(prediction_state['varmax_predictions'])}")
        logger.info(f"🔍 [VARMAX_LOAD] MA results keys: {list(prediction_state['varmax_ma_results'].keys()) if prediction_state['varmax_ma_results'] else 'None'}")
        
        return jsonify({
            'success': True,
            'prediction': prediction
        })
        
    except Exception as e:
        logger.error(f"Error getting saved VARMAX prediction for {date}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/varmax/saved/<date>', methods=['DELETE'])
def delete_saved_varmax_prediction_api(date):
    """특정 날짜의 저장된 VARMAX 예측을 삭제하는 API"""
    try:
        success = delete_saved_varmax_prediction(date)
        
        if not success:
            return jsonify({
                'success': False,
                'error': f'Failed to delete prediction for date: {date}'
            }), 404
        
        return jsonify({
            'success': True,
            'message': f'Prediction for {date} deleted successfully'
        })
        
    except Exception as e:
        logger.error(f"Error deleting saved VARMAX prediction for {date}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# 6) VARMAX 의사결정 조회
# 7) VARMAX 상태 리셋 API (새로 추가)
@app.route('/api/varmax/reset', methods=['POST', 'OPTIONS'])
@cross_origin()
def reset_varmax_state():
    """VARMAX 예측 상태를 리셋하는 API (hang된 예측 해결용)"""
    global prediction_state
    
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    try:
        logger.info("🔄 [VARMAX_RESET] Resetting VARMAX prediction state...")
        
        # VARMAX 상태 완전 리셋
        prediction_state['varmax_is_predicting'] = False
        prediction_state['varmax_prediction_progress'] = 0
        prediction_state['varmax_error'] = None
        prediction_state['varmax_predictions'] = []
        prediction_state['varmax_half_month_averages'] = []
        prediction_state['varmax_metrics'] = {}
        prediction_state['varmax_ma_results'] = {}
        prediction_state['varmax_selected_features'] = []
        prediction_state['varmax_current_date'] = None
        prediction_state['varmax_model_info'] = {}
        prediction_state['varmax_plots'] = {}
        
        logger.info("✅ [VARMAX_RESET] VARMAX state reset completed")
        
        return jsonify({
            'success': True,
            'message': 'VARMAX state reset successfully',
            'current_state': {
                'is_predicting': prediction_state.get('varmax_is_predicting', False),
                'progress': prediction_state.get('varmax_prediction_progress', 0),
                'error': prediction_state.get('varmax_error')
            }
        })
        
    except Exception as e:
        logger.error(f"❌ [VARMAX_RESET] Error resetting VARMAX state: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to reset VARMAX state: {str(e)}'
        }), 500

@app.route('/api/varmax/decision', methods=['POST', 'OPTIONS'])
@cross_origin() 
def get_varmax_decision():
    """VARMAX 의사 결정 조회 API"""
    # 1) OPTIONS(preflight) 요청 처리
    if request.method == 'OPTIONS':
        return make_response('', 200)
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '파일이 없습니다.'}), 400

    file = request.files['file']
    # 파일 저장 경로 설정
    save_dir = '/path/to/models'
    os.makedirs(save_dir, exist_ok=True)  # 폴더가 없으면 생성
    filepath = os.path.join(save_dir, secure_filename(file.filename))
    file.save(filepath)

    logger.info("POST /api/varmax/decision 로 진입")
    #data = request.get_json()
    #filepath = data.get('filepath')
    """# 유효성 검사
    if not filepath or not os.path.exists(os.path.normpath(filepath)):
        return jsonify({'success': False, 'error': 'Invalid file path'}), 400"""

    results = varmax_decision(filepath)
    logger.info("결과 데이터프레임 형성 완료")
    column_order1 = ["조건1", "조건2", "샘플 수", "음수 비율 [%]"]
    column_order2 = ["조건1", "조건2", "샘플 수", "상위 변동성 확률 [%]"]

    return jsonify({
        'success': True,
        'filepath': filepath,  # ← 파일 경로 추가
        'filename': file.filename,
        'columns1': column_order1,
        'columns2': column_order2,
        'case_1':      results['case_1'],
        'case_2':      results['case_2'],
    })

@app.route('/api/market-status', methods=['GET'])
def get_market_status():
    """최근 30일간의 시장 가격 데이터를 카테고리별로 반환하는 API"""
    try:
        # 파일 경로 가져오기
        file_path = request.args.get('file_path')
        if not file_path:
            return jsonify({
                'success': False,
                'error': 'File path is required'
            }), 400
        
        # URL 디코딩 및 파일 경로 정규화 (Windows 백슬래시 처리)
        import urllib.parse
        file_path = urllib.parse.unquote(file_path)  # URL 디코딩
        file_path = os.path.normpath(file_path)
        logger.info(f"📊 [MARKET_STATUS] Normalized file path: {file_path}")
        
        # 파일 존재 여부 확인
        if not os.path.exists(file_path):
            logger.error(f"❌ [MARKET_STATUS] File not found: {file_path}")
            return jsonify({
                'success': False,
                'error': f'File not found: {file_path}'
            }), 400
        
        # 원본 데이터 직접 로드 (Date 컬럼 유지를 위해) - Excel/CSV 파일 모두 지원
        try:
            file_ext = os.path.splitext(file_path.lower())[1]
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
                logger.info(f"📊 [MARKET_STATUS] CSV data loaded: {df.shape}")
            elif file_ext in ['.xlsx', '.xls']:
                # Excel 파일의 경우 보안 문제를 고려한 안전한 로딩 사용
                df = load_data_safe(file_path, use_cache=True, use_xlwings_fallback=True)
                # 인덱스가 Date인 경우 컬럼으로 복원
                if df.index.name == 'Date':
                    df = df.reset_index()
                logger.info(f"📊 [MARKET_STATUS] Excel data loaded with security bypass: {df.shape}")
            else:
                logger.error(f"❌ [MARKET_STATUS] Unsupported file format: {file_ext}")
                return jsonify({
                    'success': False,
                    'error': f'Unsupported file format: {file_ext}. Only CSV and Excel files are supported.'
                }), 400
        except Exception as e:
            logger.error(f"❌ [MARKET_STATUS] Failed to load data file: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Failed to load data file: {str(e)}'
            }), 400
        
        if df is None or df.empty:
            logger.error(f"❌ [MARKET_STATUS] No data available or empty dataframe")
            return jsonify({
                'success': False,
                'error': 'No data available'
            }), 400
        
        # 날짜 컬럼 확인 및 정렬
        logger.info(f"📊 [MARKET_STATUS] Columns in dataframe: {list(df.columns)}")
        if 'Date' not in df.columns:
            logger.error(f"❌ [MARKET_STATUS] Date column not found. Available columns: {list(df.columns)}")
            return jsonify({
                'success': False,
                'error': 'Date column not found in data'
            }), 400
        
        # 날짜로 정렬
        df = df.sort_values('Date')
        
        # 휴일 정보 로드
        holidays = get_combined_holidays(df=df)
        holiday_dates = set([h['date'] if isinstance(h, dict) else h for h in holidays])
        
        # 영업일만 필터링
        def is_business_day(date_str):
            date_obj = pd.to_datetime(date_str).date()
            weekday = date_obj.weekday()  # 0=월요일, 6=일요일
            return weekday < 5 and date_str not in holiday_dates  # 월~금 & 휴일 아님
        
        logger.info(f"📊 [MARKET_STATUS] Total rows before business day filtering: {len(df)}")
        logger.info(f"📊 [MARKET_STATUS] Holiday dates count: {len(holiday_dates)}")
        
        business_days_df = df[df['Date'].apply(is_business_day)]
        logger.info(f"📊 [MARKET_STATUS] Business days after filtering: {len(business_days_df)}")
        
        if business_days_df.empty:
            logger.error(f"❌ [MARKET_STATUS] No business days found after filtering")
            return jsonify({
                'success': False,
                'error': 'No business days found in data'
            }), 400
        
        # 최근 30일 영업일 데이터 추출
        recent_30_days = business_days_df.tail(30)

        # Crack 변수 계산 (MOPJ - Brent_Singapore * 7.5)
        if 'MOPJ' in recent_30_days.columns and 'Brent_Singapore' in recent_30_days.columns:
            recent_30_days = recent_30_days.copy()  # 경고 방지를 위한 복사본 생성
            recent_30_days['Crack'] = recent_30_days['MOPJ'] - (recent_30_days['Brent_Singapore'] * 7.5)
            logger.info(f"📊 [MARKET_STATUS] Crack variable calculated: MOPJ - Brent_Singapore * 7.5")
        elif 'MOPJ' in recent_30_days.columns and 'Brent' in recent_30_days.columns:
            # Brent_Singapore가 없으면 Brent로 대체 계산
            recent_30_days = recent_30_days.copy()
            recent_30_days['Crack'] = recent_30_days['MOPJ'] - (recent_30_days['Brent'] * 7.5)
            logger.info(f"📊 [MARKET_STATUS] Crack variable calculated with Brent fallback: MOPJ - Brent * 7.5")
        else:
            logger.warning(f"⚠️ [MARKET_STATUS] Cannot calculate Crack: Missing MOPJ or Brent columns")

        # 카테고리별 컬럼 분류 (실제 데이터 컬럼명에 맞게 수정)
        categories = {
            '원유 가격': [
                'WTI', 'Brent', 'Dubai', 'Crack'
            ],
            '가솔린 가격': [
                'Gasoline_92RON', 'Gasoline_95RON', 'Europe_M.G_10ppm', 'RBOB (NYMEX)_M1'
            ],
            '나프타 가격': [
                'MOPJ', 'MOPAG', 'MOPS', 'Europe_CIF NWE'
            ],
            'LPG 가격': [
                'C3_LPG', 'C4_LPG'
            ],
            '경제지표': [
                'Dow_Jones', 'Euro', 'Gold', 'Exchange'
            ],
            '석유화학 제품 가격': [
                'EL_CRF NEA', 'EL_CRF SEA', 'PL_FOB Korea', 'BZ_FOB Korea', 'BZ_FOB SEA', 'BZ_FOB US M1', 'BZ_FOB US M2', 
                'TL_FOB Korea', 'TL_FOB US M1', 'TL_FOB US M2','MX_FOB Korea', 'PX_FOB Korea', 'SM_FOB Korea', 'RPG Value_FOB PG', 
                'FO_HSFO 180 CST', 'MTBE_FOB Singapore'
            ]
        }
        
        # 실제 존재하는 컬럼만 필터링
        available_columns = set(recent_30_days.columns)
        filtered_categories = {}
        
        logger.info(f"📊 [MARKET_STATUS] Available columns: {sorted(available_columns)}")
        
        for category, columns in categories.items():
            existing_columns = [col for col in columns if col in available_columns]
            if existing_columns:
                filtered_categories[category] = existing_columns
                logger.info(f"📊 [MARKET_STATUS] Category '{category}': found {len(existing_columns)} columns: {existing_columns}")
            else:
                logger.warning(f"⚠️ [MARKET_STATUS] Category '{category}': no matching columns found from {columns}")
        
        if not filtered_categories:
            logger.error(f"❌ [MARKET_STATUS] No categories found! Expected columns don't match available columns")
            return jsonify({
                'success': False,
                'error': 'No matching columns found for market status categories',
                'debug_info': {
                    'available_columns': sorted(available_columns),
                    'expected_categories': categories
                }
            }), 400
        
        # 카테고리별 데이터 구성
        result = {
            'success': True,
            'date_range': {
                'start_date': recent_30_days['Date'].iloc[0],
                'end_date': recent_30_days['Date'].iloc[-1],
                'total_days': len(recent_30_days)
            },
            'categories': {}
        }
        
        for category, columns in filtered_categories.items():
            category_data = {
                'columns': columns,
                'data': []
            }
            
            for _, row in recent_30_days.iterrows():
                data_point = {
                    'date': row['Date'],
                    'values': {}
                }
                
                for col in columns:
                    if pd.notna(row[col]):
                        data_point['values'][col] = float(row[col])
                    else:
                        data_point['values'][col] = None
                
                category_data['data'].append(data_point)
            
            result['categories'][category] = category_data
        
        logger.info(f"✅ [MARKET_STATUS] Returned {len(recent_30_days)} business days data for {len(filtered_categories)} categories")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ [MARKET_STATUS] Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get market status: {str(e)}'
        }), 500

@app.route('/api/gpu-info', methods=['GET'])
def get_gpu_info():
    """GPU 및 디바이스 정보를 반환하는 API"""
    try:
        from app.core.gpu_manager import get_detailed_gpu_utilization
        # 실시간 GPU 테스트 여부 확인
        run_test = request.args.get('test', 'false').lower() == 'true'
        
        # GPU 정보 수집
        device_info = {
            'cuda_available': torch.cuda.is_available(),
            'pytorch_version': torch.__version__,
            'default_device': str(DEFAULT_DEVICE),
            'current_device_info': {},
            'test_performed': False,
            'test_results': {}
        }
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            
            # 실시간 GPU 활용률 확인 (상세 버전)
            gpu_utilization_stats = get_detailed_gpu_utilization()
            
            device_info.update({
                'gpu_count': gpu_count,
                'current_gpu_device': current_device,
                'cudnn_version': torch.backends.cudnn.version(),
                'cudnn_enabled': torch.backends.cudnn.enabled,
                'detailed_utilization': gpu_utilization_stats,
                'gpus': []
            })
            
            # 각 GPU 정보
            for i in range(gpu_count):
                gpu_props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                total = gpu_props.total_memory / 1024**3
                
                # PyTorch 버전 호환성을 위한 안전한 속성 접근
                gpu_info = {
                    'device_id': i,
                    'name': getattr(gpu_props, 'name', 'Unknown GPU'),
                    'total_memory_gb': round(total, 2),
                    'allocated_memory_gb': round(allocated, 2),
                    'cached_memory_gb': round(cached, 2),
                    'memory_usage_percent': round((allocated / total) * 100, 2),
                    'compute_capability': f"{getattr(gpu_props, 'major', 0)}.{getattr(gpu_props, 'minor', 0)}",
                    'is_current': i == current_device
                }
                
                # 선택적 속성들 (PyTorch 버전에 따라 존재하지 않을 수 있음)
                if hasattr(gpu_props, 'multiprocessor_count'):
                    gpu_info['multiprocessor_count'] = gpu_props.multiprocessor_count
                elif hasattr(gpu_props, 'multi_processor_count'):
                    gpu_info['multiprocessor_count'] = gpu_props.multi_processor_count
                else:
                    gpu_info['multiprocessor_count'] = 'N/A'
                
                # 추가 GPU 속성들 (존재하는 경우에만)
                optional_attrs = {
                    'max_threads_per_block': 'max_threads_per_block',
                    'max_threads_per_multiprocessor': 'max_threads_per_multiprocessor',
                    'warp_size': 'warp_size',
                    'memory_clock_rate': 'memory_clock_rate'
                }
                
                for attr_name, prop_name in optional_attrs.items():
                    if hasattr(gpu_props, prop_name):
                        gpu_info[attr_name] = getattr(gpu_props, prop_name)
                
                device_info['gpus'].append(gpu_info)
            
            # 현재 디바이스 상세 정보
            current_gpu_props = torch.cuda.get_device_properties(current_device)
            device_info['current_device_info'] = {
                'name': current_gpu_props.name,
                'total_memory_gb': round(current_gpu_props.total_memory / 1024**3, 2),
                'allocated_memory_gb': round(torch.cuda.memory_allocated(current_device) / 1024**3, 2),
                'cached_memory_gb': round(torch.cuda.memory_reserved(current_device) / 1024**3, 2)
            }
            
            # GPU 테스트 수행 (요청된 경우)
            if run_test:
                try:
                    logger.info("🧪 API에서 GPU 테스트 수행 중...")
                    
                    # 테스트 전 메모리 상태
                    memory_before = {
                        'allocated': torch.cuda.memory_allocated(current_device) / 1024**3,
                        'cached': torch.cuda.memory_reserved(current_device) / 1024**3
                    }
                    
                    # 간단한 GPU 연산 테스트
                    test_size = 500
                    test_tensor = torch.randn(test_size, test_size, device=current_device, dtype=torch.float32)
                    test_result = torch.matmul(test_tensor, test_tensor.T)
                    computation_result = torch.sum(test_result).item()
                    
                    # 테스트 후 메모리 상태
                    memory_after = {
                        'allocated': torch.cuda.memory_allocated(current_device) / 1024**3,
                        'cached': torch.cuda.memory_reserved(current_device) / 1024**3
                    }
                    
                    # 메모리 사용량 차이 계산
                    memory_diff = {
                        'allocated_diff': memory_after['allocated'] - memory_before['allocated'],
                        'cached_diff': memory_after['cached'] - memory_before['cached']
                    }
                    
                    device_info['test_performed'] = True
                    device_info['test_results'] = {
                        'test_tensor_size': f"{test_size}x{test_size}",
                        'computation_result': round(computation_result, 4),
                        'memory_before_gb': {
                            'allocated': round(memory_before['allocated'], 4),
                            'cached': round(memory_before['cached'], 4)
                        },
                        'memory_after_gb': {
                            'allocated': round(memory_after['allocated'], 4),
                            'cached': round(memory_after['cached'], 4)
                        },
                        'memory_diff_gb': {
                            'allocated': round(memory_diff['allocated_diff'], 4),
                            'cached': round(memory_diff['cached_diff'], 4)
                        },
                        'test_success': True
                    }
                    
                    # 테스트 텐서 정리
                    del test_tensor, test_result
                    torch.cuda.empty_cache()
                    
                    # 정리 후 메모리 상태
                    memory_final = {
                        'allocated': torch.cuda.memory_allocated(current_device) / 1024**3,
                        'cached': torch.cuda.memory_reserved(current_device) / 1024**3
                    }
                    
                    device_info['test_results']['memory_after_cleanup_gb'] = {
                        'allocated': round(memory_final['allocated'], 4),
                        'cached': round(memory_final['cached'], 4)
                    }
                    
                    logger.info(f"✅ GPU 테스트 완료: 메모리 사용량 변화 {memory_diff['allocated_diff']:.4f}GB")
                    
                except Exception as test_e:
                    logger.error(f"❌ GPU 테스트 실패: {str(test_e)}")
                    device_info['test_performed'] = True
                    device_info['test_results'] = {
                        'test_success': False,
                        'error': str(test_e)
                    }
        else:
            device_info.update({
                'gpu_count': 0,
                'reason': 'CUDA not available - using CPU'
            })
        
        # 로그에도 정보 출력
        logger.info(f"🔍 GPU Info API 호출:")
        logger.info(f"  🔧 CUDA 사용 가능: {device_info['cuda_available']}")
        logger.info(f"  ⚡ 기본 디바이스: {device_info['default_device']}")
        if device_info['cuda_available']:
            logger.info(f"  🎮 GPU 개수: {device_info.get('gpu_count', 0)}")
            if 'current_gpu_device' in device_info:
                logger.info(f"  🎯 현재 GPU: {device_info['current_gpu_device']}")
        
        # 테스트 결과 로깅
        if device_info.get('test_performed', False):
            test_results = device_info.get('test_results', {})
            if test_results.get('test_success', False):
                logger.info(f"  ✅ GPU 테스트 성공")
            else:
                logger.warning(f"  ❌ GPU 테스트 실패: {test_results.get('error', 'Unknown error')}")
        
        return jsonify({
            'success': True,
            'device_info': device_info,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ GPU 정보 API 오류: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get GPU info: {str(e)}'
        }), 500

@app.route('/api/gpu-monitoring-comparison', methods=['GET'])
def get_gpu_monitoring_comparison():
    """다양한 GPU 모니터링 방법을 비교하는 API"""
    try:
        comparison_data = compare_gpu_monitoring_methods()
        
        # 추가적인 설명 정보
        explanation = {
            'why_different_readings': [
                "Windows 작업 관리자는 주로 3D 그래픽 엔진 활용률을 표시합니다",
                "nvidia-smi는 CUDA 연산 활용률을 측정하므로 ML/AI 작업에 더 정확합니다",
                "측정 시점의 차이로 인해 순간적인 값이 다를 수 있습니다",
                "GPU는 여러 엔진(Compute, 3D, Encoder, Decoder)을 가지고 있어 각각 다른 활용률을 보입니다"
            ],
            'recommendations': [
                "ML/AI 작업: nvidia-smi의 GPU 활용률 확인",
                "게임/3D 렌더링: Windows 작업 관리자의 3D 활용률 확인", 
                "비디오 처리: nvidia-smi의 Encoder/Decoder 활용률 확인",
                "메모리 사용량: PyTorch CUDA 정보와 nvidia-smi 모두 확인"
            ],
            'task_manager_vs_nvidia_smi': {
                "작업 관리자 GPU": "주로 3D 그래픽 워크로드 (DirectX, OpenGL)",
                "nvidia-smi GPU": "CUDA 연산 워크로드 (ML, AI, GPGPU)",
                "왜 다른가": "서로 다른 GPU 엔진을 측정하기 때문",
                "어느 것이 정확한가": "작업 유형에 따라 다름 - ML/AI는 nvidia-smi가 정확"
            }
        }
        
        # 현재 상황 분석
        current_analysis = {
            'status': 'monitoring_successful',
            'notes': []
        }
        
        if comparison_data.get('nvidia_smi'):
            nvidia_util = comparison_data['nvidia_smi'].get('gpu_utilization', '0')
            try:
                util_value = float(nvidia_util)
                if util_value < 10:
                    current_analysis['notes'].append(f"현재 CUDA 활용률이 매우 낮습니다 ({util_value}%)")
                    current_analysis['notes'].append("이는 정상적일 수 있습니다 - ML 작업이 진행 중이 아닐 때")
                elif util_value > 50:
                    current_analysis['notes'].append(f"현재 CUDA 활용률이 높습니다 ({util_value}%)")
                    current_analysis['notes'].append("ML/AI 작업이 활발히 진행 중입니다")
            except:
                pass
        
        if comparison_data.get('torch_cuda'):
            memory_usage = comparison_data['torch_cuda'].get('memory_usage_percent', 0)
            if memory_usage > 1:
                current_analysis['notes'].append(f"PyTorch가 GPU 메모리를 사용 중입니다 ({memory_usage:.1f}%)")
            else:
                current_analysis['notes'].append("PyTorch가 현재 GPU 메모리를 거의 사용하지 않습니다")
        
        return jsonify({
            'success': True,
            'comparison_data': comparison_data,
            'explanation': explanation,
            'current_analysis': current_analysis,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ GPU 모니터링 비교 API 오류: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to compare GPU monitoring methods: {str(e)}'
        }), 500

# 메인 실행 부분 업데이트
if __name__ == '__main__':
    # 필요한 패키지 설치 확인
    try:
        import optuna
    except ImportError:
        logger.warning("Optuna 패키지가 설치되어 있지 않습니다. 하이퍼파라미터 최적화를 위해 https://inthiswork.com/archives/226539설치가 필요합니다.")
        logger.warning("pip install optuna 명령으로 설치할 수 있습니다.")
    
    # 🎯 파일별 캐시 시스템 - 레거시 디렉토리 및 인덱스 파일 생성 제거
    # 모든 데이터는 이제 파일별 캐시 디렉토리에 저장됩니다
    logger.info("🚀 Starting with file-based cache system - no legacy directories needed")
    
    # 라우트 등록 확인을 위한 디버깅
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule.endpoint}: {rule.rule} {list(rule.methods or [])}")
    
    print("Starting Flask app with attention-map endpoint...")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
