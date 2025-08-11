# app/__init__.py
from flask import Flask
from flask_cors import CORS
import logging

# Flask 앱 인스턴스 생성
app = Flask(__name__)

# CORS 설정
# app_rev.py에서는 resources={r"/api/*": {"origins": "*"}}, supports_credentials=False 로 설정되어 있습니다.
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=False)

# config.py에서 설정 값 로드 및 적용
from .config import UPLOAD_FOLDER, MAX_CONTENT_LENGTH, SEED, setup_logging
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 로깅 설정 초기화 (config.py의 함수 호출)
setup_logging()
logger = logging.getLogger(__name__)

# 랜덤 시드 설정 (utils.file_utils의 함수 호출)
from .utils.file_utils import set_seed
set_seed(SEED)

# GPU 정보 확인 및 기본 디바이스 설정 (core.gpu_manager의 함수 호출)
from .core.gpu_manager import check_gpu_availability
global DEFAULT_DEVICE, CUDA_AVAILABLE
DEFAULT_DEVICE, CUDA_AVAILABLE = check_gpu_availability()
logger.info(f"기본 디바이스 설정 완료: {DEFAULT_DEVICE}")

# API 라우트 등록을 위해 app_routes 모듈 임포트 (주로 마지막에 위치)
from .core import app_routes # app 객체가 완전히 구성된 후 임포트하여 라우트 등록
