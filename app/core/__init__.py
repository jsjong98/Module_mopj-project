"""
app/core 패키지
핵심 기능과 상태 관리를 담당하는 모듈
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# 로깅 설정
def setup_logger():
    """애플리케이션 로거 설정"""
    logger = logging.getLogger('app.core')
    
    # 이미 핸들러가 있으면 중복 방지
    if not logger.handlers:
        # 로그 레벨 설정 (환경변수로 제어 가능)
        log_level = os.environ.get('LOG_LEVEL', 'INFO')
        logger.setLevel(getattr(logging, log_level))
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포맷터 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        # 파일 핸들러 (선택적)
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / 'app_core.log', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # 핸들러 추가
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger

# 로거 초기화
logger = setup_logger()

# state_manager에서 가져오기
from .state_manager import prediction_state

# gpu_manager에서 주요 함수들 가져오기
from .gpu_manager import (
    check_gpu_availability,
    get_gpu_utilization,
    get_detailed_gpu_utilization,
    compare_gpu_monitoring_methods,
    log_device_usage
)

# 환경 설정 및 상수
DEFAULT_DEVICE = None  # 초기화 시 설정됨
CACHE_ROOT_DIR = os.environ.get('CACHE_ROOT_DIR', 'cache')
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
PREDICTIONS_DIR = os.path.join(CACHE_ROOT_DIR, 'predictions')

# 디렉토리 생성
for directory in [CACHE_ROOT_DIR, UPLOAD_FOLDER, PREDICTIONS_DIR]:
    os.makedirs(directory, exist_ok=True)
    logger.info(f"📁 Ensured directory exists: {directory}")

# GPU 초기화
def initialize_gpu():
    """GPU를 초기화하고 기본 디바이스 설정"""
    global DEFAULT_DEVICE
    device, cuda_available = check_gpu_availability()
    DEFAULT_DEVICE = device
    logger.info(f"🚀 Core module initialized with device: {DEFAULT_DEVICE}")
    return device, cuda_available

# 모듈 로드 시 GPU 초기화
DEFAULT_DEVICE, CUDA_AVAILABLE = initialize_gpu()

# 버전 정보
__version__ = '1.0.0'
__author__ = 'MOPJ Prediction Team'

# 공개 API
__all__ = [
    # 상태 관리
    'prediction_state',
    
    # GPU 관리
    'check_gpu_availability',
    'get_gpu_utilization',
    'get_detailed_gpu_utilization',
    'compare_gpu_monitoring_methods',
    'log_device_usage',
    
    # 설정 및 상수
    'DEFAULT_DEVICE',
    'CUDA_AVAILABLE',
    'CACHE_ROOT_DIR',
    'UPLOAD_FOLDER',
    'PREDICTIONS_DIR',
    
    # 유틸리티
    'logger',
    'initialize_gpu'
]

logger.info("✅ app.core package initialized successfully")
