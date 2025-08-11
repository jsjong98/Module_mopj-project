import os
import shutil
import time
import random
import numpy as np
import torch
import logging
import pandas as pd
from werkzeug.utils import secure_filename
import psutil

# logger 정의를 파일 상단에 추가
logger = logging.getLogger(__name__)

# SEED import 또는 정의
try:
    from app.config import SEED
except ImportError:
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
        # 레거시 지원 제거 (deprecated)
    except Exception as e:
        logger.debug(f"Optuna 시드 설정 생략: {e}")
    
    logger.debug(f"🎯 랜덤 시드 {seed}로 고정됨")

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

def cleanup_excel_processes():
    """
    남은 Excel 프로세스들을 정리하는 함수
    """
    try:
        import psutil
        excel_processes = []
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] and 'excel' in proc.info['name'].lower():
                    excel_processes.append(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if excel_processes:
            logger.info(f"🔧 [EXCEL_CLEANUP] Found {len(excel_processes)} Excel processes to clean up")
            for pid in excel_processes:
                try:
                    proc = psutil.Process(pid)
                    proc.terminate()
                    proc.wait(timeout=3)
                    logger.debug(f"🔧 [EXCEL_CLEANUP] Terminated Excel process {pid}")
                except:
                    pass
    except ImportError:
        # psutil이 없으면 무시
        pass
    except Exception as e:
        logger.debug(f"🔧 [EXCEL_CLEANUP] Error during cleanup: {e}")