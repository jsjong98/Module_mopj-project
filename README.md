## MOPJ 가격 예측 시스템

MOPJ(나프타) 가격을 예측하고 구매 구간을 추천하는 End-to-End 시스템입니다. 백엔드는 Flask(Python, PyTorch LSTM + VARMAX), 프론트엔드는 React로 구성되어 있으며, 예측 결과와 지표는 PNG로 저장되어 대시보드에서 시각화됩니다.

### 주요 기능
- **단기 LSTM 예측(23 영업일)**: Attention 기반 개선 LSTM 모델, 반월(SM1/SM2) 기준 하이퍼파라미터 캐시(k-fold) 재사용
- **장기 VARMAX 예측(반월 단위)**: 선택 변수 기반 VARMAX + 잔차(RandomForest) 보정, 반월 평균/이동평균 분석
- **구매 구간 추천**: 다음 반월 영업일에 대해 윈도우별 평균가를 계산하고 전역 랭크로 최적 구간 선정
- **누적 예측 분석**: 날짜별 예측을 축적하여 일관성/신뢰도/지표(F1, 방향정확도, MAPE) 집계 시각화
- **DRM/보안 확장자 우회 로딩**: xlwings로 Excel 프로세스를 경유하여 데이터 추출(.cs→.csv, .xl→.xlsx, .log→.xlsx 등 처리)
- **시각화 PNG 저장**: 모든 플롯은 PNG로 저장(인터랙티브 창 미출력)

---

## 빠른 시작

### 1) 요구 사항
- Python 3.9+
- Node.js 16+ (프론트엔드)
- Microsoft Excel(Windows, xlwings 사용 시)
- CUDA GPU(선택 사항, PyTorch CUDA)

Python 패키지(예시):
```bash
pip install flask flask-cors pandas numpy torch scikit-learn matplotlib seaborn statsmodels xlwings optuna
```

또는 환경 스크립트(선택):
```bash
python app/scripts/setup_environment.py
```

### 2) 백엔드 실행(Flask)
프로젝트 루트에서:
```bash
cd app
python run.py --host 127.0.0.1 --port 5000 --debug
# 또는 프로덕션 바인딩
python run.py --production --port 5000
```
Windows 배치 스크립트(선택):
```cmd
app\scripts\start_server.bat
```

서버가 기동되면:
- API Base: `http://localhost:5000/api`
- 상태: `GET /api/health`, `GET /api/test`

### 3) 프론트엔드 실행(React)
```bash
cd mopj-dashboard
npm install
npm start
# 개발 서버: http://localhost:3000 (백엔드: http://localhost:5000)
```

---

## 데이터 업로드와 전처리

### 지원 형식 및 보안 확장자
- 원본: `.xlsx`, `.xls`, `.csv`
- 보안 확장자 자동 처리: `.cs → .csv`, `.xl → .xlsx`, `.log → .xlsx`, `.dat/.txt → 내용 기반 판별`
- DRM/보안 적용 파일: xlwings로 Excel 프로세스를 경유하여 안전 추출(Windows/Excel 필요)

### 시트 및 날짜 처리
- 기본 시트: `29 Nov 2010 till todate`(없을 시 첫 시트 사용)
- 날짜 컬럼 표준화: `Date` 인덱스 사용
- 결측/이상치 처리: 전/후방 보간, 쉼표 소수점, TBA/Q 값 보정 등 고급 정제 파이프라인

### CSV 캐시(성능 최적화)
- 업로드 파일은 완전 전처리 후 `app/cache/processed_csv/`에 `.cs`(CSV)와 `metadata_*.json`으로 캐시
- 데이터 확장(구간 추가) 감지 시 캐시 부분 갱신/재생성

---

## 모델 개요

### LSTM(단기)
- 모듈: `app/models/lstm_model.py`
- 특징: 계층 LSTM + 듀얼 어텐션(Temporal/Feature), Prev value encoder, Conv feature extractor
- 손실: `DirectionalLoss(α=0.7, β=0.2)`(MSE+방향성+연속성)
- 하이퍼파라미터: 반월(SM1/SM2) 기준 시계열 k-fold 최적화 + 캐시 재사용(`app/cache/hyperparameters/`)
- 학습 데이터: 기본 2022-01-01 이후 최근 구간 우선(부족 시 전체)

### VARMAX(장기)
- 모듈: `app/models/varmax_model.py`
- 절차: 변수군 상관 기반 선택 → VARMAX(p=7,q=0) 적합 → 잔차(RandomForest)로 보정
- 출력: 예측 시퀀스, 반월 평균, 이동평균, 성능 지표(R² train/test, F1/Acc/MAPE 등)

---

## 예측·시각화·저장

### 단일 예측(LSTM)
1) 파일 업로드 후 날짜 선택
2) 예측 시작일 다음 영업일부터 23영업일 예측(휴일/주말 제외)
3) 다음 반월 대상 영업일 집합을 별도 계산하여 구매 구간 점수화(Global Rank)
4) 결과 저장: `app/cache/predictions/`

저장물(예시):
- `prediction_start_YYYYMMDD_meta.json`(메타/지표/구간점수)
- `prediction_start_YYYYMMDD_ma.json`(이동평균 결과)
- `prediction_start_YYYYMMDD_attention.json`(어텐션/특성중요도)
- `prediction_start_YYYYMMDD.png`(기본 그래프)
- `ma_analysis_YYYYMMDD.png`(이동평균 그래프)

주의: 모든 플롯은 PNG로 저장되며, 화면 창을 띄우지 않습니다.

### 누적 예측(LSTM)
- 기간 지정(시작~종료) 후 날짜별 예측을 수행/캐시 재활용
- 누적 지표, 일관성 점수, 구매 신뢰도(상위 3구간 점수 누적/정규화) 산출 및 시각화

### VARMAX 예측
- 기준일과 변수 수를 자동 탐색하여 MAPE 최소 모델 선택 후 예측/시각화
- 결과 및 플롯을 캐시에 보관, 저장 목록 조회/개별 조회/삭제 지원

---

## API 요약(일부)

백엔드 Base: `http://localhost:5000/api`

### 공통/상태
- `GET /health`, `GET /test`
- `GET /predict/status`

### 데이터/업로드
- `POST /upload` (파일 업로드)
- `GET /data/dates?filepath=...` (가용 날짜)
- `POST /data/refresh` (파일-캐시 비교/갱신 판단)

### LSTM 예측
- `POST /predict` (단일 예측 시작)
- `GET /results` (최신 결과)
- `GET /results/predictions` / `.../interval-scores` / `.../moving-averages` / `.../attention-map`
- 누적: `POST /predict/accumulated`, `GET /results/accumulated`, `GET /results/accumulated/{date}`

### VARMAX
- `POST /varmax/decision` (의사결정용 CSV 업로드)
- `POST /varmax/predict`, `GET /varmax/status`, `GET /varmax/results`
- `GET /varmax/predictions`, `GET /varmax/moving-averages`
- `GET /varmax/saved?limit=...`, `GET/DELETE /varmax/saved/{date}`, `POST /varmax/reset`

### 휴일 관리
- `GET /holidays`, `POST /holidays/upload`, `POST /holidays/reload`
- 기본 파일: `holidays/holidays.csv` (없으면 자동 생성 + 데이터 빈 평일 감지 병합)

### 캐시 유틸
- `POST /cache/check` (기간 캐시 확인), `POST /cache/clear/accumulated`
- `GET /results/accumulated/recent`, `GET /results/accumulated/report`

프론트엔드 `mopj-dashboard/src/services/api.js`에 전체 엔드포인트 사용 예가 포함되어 있습니다.

---

## 캐시/디렉토리 구조(핵심)
프로젝트 루트 기준:

```
app/
  run.py                      # Flask 서버 진입점
  config.py                   # 폴더/캐시 설정 및 로깅
  core/                       # GPU/상태/라우트
  data/                       # 로더/전처리/캐시 매니저
  models/                     # LSTM, VARMAX, 손실함수
  prediction/                 # 예측 파이프라인/지표/백그라운드 작업
  visualization/              # 플롯 생성(모두 PNG 저장)
  cache/
    processed_csv/            # 전처리 CSV(.cs) + 메타
    predictions/              # 예측 결과/지표/플롯
    plots/attention|ma_plots  # 주의/이동평균 플롯
    hyperparameters/          # 반월별 LSTM HP 캐시
    varmax/                   # VARMAX 예측 캐시
  uploads/                    # 업로드 파일(원본/보안확장자 포함)
holidays/                     # 휴일 파일(없으면 생성)
mopj-dashboard/               # React 대시보드
```

---

## 대시보드(프론트엔드)
- 업로드(일반/VARMAX), 날짜 선택, 진행률 표시, 예측/이동평균/어텐션 시각화, 누적 분석 카드 제공
- 기본 API URL은 `http://localhost:5000/api`로 설정되어 있습니다.

실행:
```bash
cd mopj-dashboard
npm install
npm start
```

---

## 팁 & 문제 해결

- **Excel/DRM 문제**: Windows + Excel이 설치되어 있어야 xlwings 우회가 동작합니다. 실패 시 pandas fallback을 시도합니다.
- **CSV 파싱 오류**: 구분자 자동 탐지 재시도(, ; \t) 및 xlwings fallback을 사용합니다.
- **GPU 활용률**: `nvidia-smi`의 CUDA 활용률과 작업 관리자 3D 지표는 다를 수 있습니다. 로그에 CUDA/메모리 상세가 출력됩니다.
- **플롯 미표시**: 서버 환경에서는 GUI 백엔드를 사용하지 않으며, 모든 플롯은 PNG로 저장됩니다.
- **대시보드 404/프록시 이슈**: 프론트의 API Base는 직접 URL(`http://localhost:5000/api`)로 설정되어 있습니다. 백엔드가 5000 포트에서 실행 중인지 확인하세요.

---

## 라이선스
© 2025 MOPJ Prediction Team. 내부 과제/연구용.


