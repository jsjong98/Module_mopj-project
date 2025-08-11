import logging
import os

# 로깅 설정 (초기화 시 호출)
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("app.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
logger = logging.getLogger(__name__) # 설정 후 로거 생성 (로깅 파일에 기록)


# 디렉토리 설정
UPLOAD_FOLDER = 'uploads'
HOLIDAY_DIR = 'holidays'
CACHE_ROOT_DIR = 'cache'
PREDICTIONS_DIR = 'predictions' # (하위 호환성용)

# 기본 디렉토리 생성
for d in [UPLOAD_FOLDER, CACHE_ROOT_DIR, PREDICTIONS_DIR]:
    os.makedirs(d, exist_ok=True)

# Flask 설정
MAX_CONTENT_LENGTH = 32 * 1024 * 1024

# 랜덤 시드 설정
SEED = 42