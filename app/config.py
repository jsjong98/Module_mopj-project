import logging
import os
from pathlib import Path

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

# 캐시 하위 디렉토리들
CACHE_PREDICTIONS_DIR = os.path.join(CACHE_ROOT_DIR, 'predictions')
CACHE_HYPERPARAMETERS_DIR = os.path.join(CACHE_ROOT_DIR, 'hyperparameters')
CACHE_PLOTS_DIR = os.path.join(CACHE_ROOT_DIR, 'plots')
CACHE_PROCESSED_CSV_DIR = os.path.join(CACHE_ROOT_DIR, 'processed_csv')
CACHE_VARMAX_DIR = os.path.join(CACHE_ROOT_DIR, 'varmax')

# 플롯 하위 디렉토리들
CACHE_ATTENTION_PLOTS_DIR = os.path.join(CACHE_PLOTS_DIR, 'attention')
CACHE_MA_PLOTS_DIR = os.path.join(CACHE_PLOTS_DIR, 'ma_plots')

# 기본 디렉토리 생성
for d in [UPLOAD_FOLDER, CACHE_ROOT_DIR, CACHE_PREDICTIONS_DIR, 
          CACHE_HYPERPARAMETERS_DIR, CACHE_PLOTS_DIR, CACHE_PROCESSED_CSV_DIR, 
          CACHE_VARMAX_DIR, CACHE_ATTENTION_PLOTS_DIR, CACHE_MA_PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Flask 설정
MAX_CONTENT_LENGTH = 32 * 1024 * 1024

# 랜덤 시드 설정
SEED = 42
