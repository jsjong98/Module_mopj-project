# MOPJ 가격 예측 시스템 - 프론트엔드

## 📋 개요
React 기반 MOPJ 가격 예측 시스템의 사용자 인터페이스입니다. LSTM 딥러닝 모델과 VARMAX 시계열 통계 모델을 활용한 예측 결과를 인터랙티브한 차트와 직관적인 대시보드를 통해 시각화합니다.

## 🔧 환경 요구사항
- Node.js 16.0 이상
- npm 8.0 이상 또는 yarn 1.22 이상
- 모던 웹 브라우저 (Chrome, Firefox, Safari, Edge)

## 📦 설치 방법

### 1. 의존성 패키지 설치
```bash
# npm 사용
npm install

# 또는 yarn 사용
yarn install
```

### 2. 환경 설정
```bash
# .env 파일 생성 (선택사항)
REACT_APP_API_URL=http://localhost:5000
REACT_APP_VERSION=1.0.0
```

## 🚀 실행 방법

### 개발 모드
```bash
# npm 사용
npm start

# 또는 yarn 사용
yarn start
```
개발 서버가 `http://localhost:3000`에서 시작됩니다.

### 프로덕션 빌드
```bash
# npm 사용
npm run build

# 또는 yarn 사용
yarn build
```

### 테스트 실행
```bash
# npm 사용
npm test

# 또는 yarn 사용
yarn test
```

## 🎯 주요 기능

### 1. 파일 업로드 인터페이스
- 드래그 앤 드롭으로 CSV 파일 업로드
- 파일 검증 및 미리보기
- 업로드 진행률 표시
- LSTM용 날짜 포함 데이터와 VARMAX용 구매 의사결정 데이터 구분 업로드

### 2. 예측 설정 및 실행
- **LSTM 예측**:
  - 단일 예측: 날짜 선택기를 통한 예측 시작일 설정
  - 누적 예측: 시작일과 종료일 범위 설정
- **VARMAX 예측**:
  - 반월별 시계열 예측 (15일 단위)
  - 다변량 통계 모델 기반 예측
  - 구매 의사결정 지원 분석
- 실시간 예측 진행률 모니터링

### 3. 결과 시각화
- **LSTM 예측 시각화**:
  - Recharts 기반 인터랙티브 차트
  - Attention 메커니즘 히트맵
  - 이동평균 분석 (5일, 10일, 23일)
- **VARMAX 예측 시각화**:
  - 반월별 예측 결과 차트
  - 이동평균 분석 (5일, 10일, 20일, 30일)
  - 구매 구간 추천 시각화
- **공통 기능**:
  - 성능 지표 실시간 표시 (F1 Score, MAPE, 정확도)
  - 반응형 차트 디자인

### 4. 대시보드 및 분석
- **LSTM 누적 분석 대시보드**:
  - 날짜별 예측 비교 테이블
  - 추이 분석 및 일관성 점수
  - 구매 신뢰도 지표
- **VARMAX 의사결정 대시보드**:
  - 구매 추천 구간 분석
  - 시계열 패턴 분석
  - 모델 성능 지표

### 5. 저장된 예측 관리
- 예측 결과 자동 저장 및 불러오기
- 날짜별 예측 기록 관리
- 예측 삭제 및 관리 기능

## 🏗️ 컴포넌트 구조

```
src/
├── App.js                          # 메인 애플리케이션 컴포넌트
├── components/                     # 재사용 가능한 컴포넌트
│   ├── FileUploader.js            # 통합 파일 업로드 컴포넌트
│   ├── DateSelector.js            # 날짜 선택 컴포넌트
│   ├── ProgressBar.js             # 진행률 표시 바
│   │
│   ├── LSTM 관련 컴포넌트/
│   ├── PredictionChart.js         # LSTM 예측 차트
│   ├── MovingAverageChart.js      # LSTM 이동평균 분석 차트
│   ├── AttentionMap.js            # Attention 가중치 시각화
│   ├── IntervalScoresTable.js     # 구간 점수 테이블
│   ├── AccumulatedResultsTable.js # 누적 결과 테이블
│   ├── AccumulatedMetricsChart.js # 누적 지표 차트
│   ├── AccumulatedSummary.js      # 누적 예측 요약
│   │
│   ├── VARMAX 관련 컴포넌트/
│   ├── VarmaxFileUploader.js      # VARMAX 전용 파일 업로드
│   ├── VarmaxPredictionChart.js   # VARMAX 예측 차트
│   ├── VarmaxMovingAverageChart.js # VARMAX 이동평균 분석
│   ├── VarmaxModelInfo.js         # VARMAX 모델 정보
│   ├── VarmaxResult.js            # VARMAX 결과 표시
│   └── VarmaxAlgorithm.js         # VARMAX 알고리즘 분석
│
├── services/                       # API 서비스
│   └── api.js                     # 백엔드 API 호출 (LSTM + VARMAX)
└── utils/                         # 유틸리티 함수
    └── formatting.js              # 데이터 포맷팅 함수
```

## 🎨 사용된 라이브러리

### 핵심 라이브러리
- **React 18.2.0**: UI 프레임워크
- **Recharts 2.15.2**: 차트 및 데이터 시각화
- **Lucide React 0.487.0**: 아이콘 라이브러리
- **Axios 1.8.4**: HTTP 클라이언트

### 유틸리티 라이브러리
- **React Modal 3.16.3**: 모달 다이얼로그
- **HTTP Proxy Middleware 3.0.5**: 개발 서버 프록시

### 테스팅 라이브러리
- **React Testing Library**: 컴포넌트 테스트
- **Jest DOM**: DOM 테스트 유틸리티

## 🎯 주요 상태 관리

### 전역 상태 (App.js)
```javascript
// 파일 및 데이터 상태
const [fileInfo, setFileInfo] = useState(null);
const [predictableStartDates, setPredictableStartDates] = useState([]);

// LSTM 예측 상태
const [isPredicting, setIsPredicting] = useState(false);
const [progress, setProgress] = useState(0);
const [predictionData, setPredictionData] = useState([]);
const [intervalScores, setIntervalScores] = useState([]);
const [maResults, setMaResults] = useState(null);
const [attentionImage, setAttentionImage] = useState(null);

// VARMAX 예측 상태
const [varmaxResults, setVarmaxResults] = useState(null);
const [varmaxPredictions, setVarmaxPredictions] = useState(null);
const [varmaxMaResults, setVarmaxMaResults] = useState(null);
const [isVarmaxPredicting, setIsVarmaxPredicting] = useState(false);
const [varmaxProgress, setVarmaxProgress] = useState(0);

// 누적 예측 상태 (LSTM)
const [accumulatedResults, setAccumulatedResults] = useState(null);
const [selectedAccumulatedDate, setSelectedAccumulatedDate] = useState(null);

// 저장된 예측 관리
const [savedPredictions, setSavedPredictions] = useState([]);
const [savedVarmaxPredictions, setSavedVarmaxPredictions] = useState([]);
```

## 📊 API 통신

### LSTM 관련 API (`services/api.js`)
```javascript
// 파일 업로드 및 예측
export const uploadFile = (file) => { /* ... */ };
export const startPrediction = (filepath, currentDate) => { /* ... */ };
export const startAccumulatedPrediction = (filepath, startDate, endDate) => { /* ... */ };

// 결과 조회
export const getPredictionResults = () => { /* ... */ };
export const getAccumulatedResults = () => { /* ... */ };
export const getPredictionStatus = () => { /* ... */ };
```

### VARMAX 관련 API (`services/api.js`)
```javascript
// VARMAX 파일 업로드 및 예측
export const uploadVarmaxFile = (file) => { /* ... */ };
export const startVarmaxPrediction = (filepath, date, predDays) => { /* ... */ };

// VARMAX 결과 조회
export const getVarmaxResults = () => { /* ... */ };
export const getVarmaxPredictions = () => { /* ... */ };
export const getVarmaxMovingAverages = () => { /* ... */ };
export const getVarmaxStatus = () => { /* ... */ };

// 저장된 VARMAX 예측 관리
export const getSavedVarmaxPredictions = (limit) => { /* ... */ };
export const getSavedVarmaxPredictionByDate = (date) => { /* ... */ };
export const deleteSavedVarmaxPrediction = (date) => { /* ... */ };
```

## 🎨 스타일링

### 인라인 스타일 시스템
```javascript
const styles = {
  card: {
    backgroundColor: 'white',
    borderRadius: '0.5rem',
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
    padding: '1rem',
    marginBottom: '1.5rem'
  },
  // VARMAX 전용 스타일
  varmaxCard: {
    backgroundColor: '#f8f9fa',
    border: '1px solid #e2e8f0',
    borderRadius: '0.5rem',
    padding: '1rem'
  }
};
```

### 반응형 디자인
- **모바일 퍼스트**: 768px 브레이크포인트 기준
- **플렉시블 레이아웃**: CSS Flexbox 및 Grid 활용
- **적응형 차트**: 화면 크기에 따른 차트 크기 조정
- **탭 기반 UI**: LSTM과 VARMAX 기능 분리

## 🔧 개발 도구

### 디버깅
```javascript
// 콘솔 로그 시스템
console.log(`🔄 [LSTM] Starting fetchResults...`);
console.log(`📊 [VARMAX] VARMAX results received:`, data);
console.log(`✅ [STATE] States updated successfully`);
console.error(`❌ [ERROR] Prediction failed:`, error);
```

### 성능 모니터링
- React DevTools 호환
- 컴포넌트 렌더링 최적화
- 메모리 사용량 모니터링
- 차트 렌더링 성능 최적화

## 🚀 배포

### 정적 파일 생성
```bash
npm run build
```

### 배포 옵션
1. **Netlify**: `build` 폴더를 드래그 앤 드롭
2. **Vercel**: GitHub 연동 자동 배포
3. **AWS S3**: S3 버킷에 정적 호스팅
4. **Nginx**: 역프록시 설정으로 백엔드와 통합
5. **Docker**: 컨테이너 기반 배포

## 🐛 문제 해결

### 1. 패키지 설치 오류

#### 일반적인 설치 오류
```bash
# npm 캐시 정리
npm cache clean --force

# node_modules 재설치
rm -rf node_modules package-lock.json
npm install
```

#### SSL 인증서 오류 (회사/기관 네트워크)
회사나 학교 네트워크에서 `npm install` 실행 시 "self-signed certificate in certificate chain" 오류가 발생하는 경우:

**방법 1: SSL 검증 비활성화 (권장)**
```bash
# SSL 검증 비활성화
npm config set strict-ssl false

# 캐시 정리 후 재설치
npm cache clean --force
npm install
```

**방법 2: 레지스트리 변경**
```bash
# HTTP 레지스트리 사용
npm config set registry http://registry.npmjs.org/
npm install

# 또는 HTTPS 레지스트리로 복구
npm config set registry https://registry.npmjs.org/
```

**방법 3: 환경변수 설정 (Windows)**
```cmd
set NODE_TLS_REJECT_UNAUTHORIZED=0
npm install
```

**방법 4: .npmrc 파일 생성**
프로젝트 루트에 `.npmrc` 파일을 생성하고 다음 내용 추가:
```
strict-ssl=false
registry=https://registry.npmjs.org/
```

**주의사항**: 
- `strict-ssl=false` 설정은 보안상 권장되지 않으므로 설치 완료 후 원복하세요
- 설정 원복: `npm config set strict-ssl true`
- 회사 네트워크에서는 네트워크 관리자에게 프록시 설정 문의

### 2. 백엔드 연결 오류 (404 NOT FOUND)

#### 증상
```
Failed to load resource: the server responded with a status of 404 (NOT FOUND)
:3000/api/holidays:1
:3000/api/results/attention-map:1
```

#### 원인 및 해결방법

**단계 1: 백엔드 서버 실행 확인**
```bash
# backend 폴더에서 서버 실행
cd backend
python app.py
```

**확인사항:**
- 서버가 `http://localhost:5000`에서 실행되는지
- `Starting server on http://localhost:5000` 메시지 확인
- 오류 메시지 없이 실행되는지

**단계 2: 백엔드 직접 연결 테스트**
브라우저에서 다음 URL에 직접 접속:
```
http://localhost:5000/api/health
```

**예상 응답:**
```json
{
  "status": "ok",
  "timestamp": "2025-01-09T...",
  "attention_endpoint_available": true
}
```

**단계 3: 프론트엔드 디버깅**
브라우저 개발자 도구(F12) → Console 탭에서 다음 명령어 실행:
```javascript
// API 연결 상태 확인
testAPI()

// 또는
debugBackend()
```

**단계 4: 포트 충돌 확인**
```bash
# Windows에서 포트 사용 확인
netstat -ano | findstr :5000
netstat -ano | findstr :3000

# Linux/Mac에서 포트 사용 확인
lsof -i :5000
lsof -i :3000
```

**단계 5: 방화벽/보안 프로그램 확인**
- Windows Defender 방화벽에서 Python, Node.js 허용 확인
- 회사 보안 프로그램에서 localhost 접속 차단 여부 확인

**단계 6: 프록시 설정 재확인**
프론트엔드를 완전히 재시작:
```bash
# 기존 프로세스 종료 (Ctrl+C)
cd mopj-dashboard
npm start
```

**단계 7: 캐시 정리**
```bash
# npm 캐시 정리
npm cache clean --force

# 브라우저 캐시 정리 (Ctrl+Shift+Delete)
# 또는 시크릿/프라이빗 브라우저에서 테스트
```

#### 고급 문제 해결

**임시 해결책: 직접 연결 모드**
`mopj-dashboard/src/services/api.js` 파일을 수정:
```javascript
// 임시로 프록시 대신 직접 연결 사용
const API_BASE_URL = 'http://localhost:5000/api';
```

**영구 해결책: 환경별 설정**
`mopj-dashboard/.env` 파일 생성:
```
REACT_APP_API_URL=http://localhost:5000/api
```

#### 네트워크 환경별 대응

**회사/기관 네트워크:**
```bash
# 프록시 서버 설정 (네트워크 관리자에게 문의)
npm config set proxy http://프록시주소:포트
npm config set https-proxy http://프록시주소:포트
```

**개인 PC (방화벽 문제):**
```bash
# Windows 방화벽에서 Python 허용
# 제어판 → 시스템 및 보안 → Windows Defender 방화벽 → 앱 허용
```

### 3. 종합 문제 해결 체크리스트

#### ✅ 필수 확인사항
1. [ ] 백엔드 서버가 포트 5000에서 실행 중
2. [ ] 프론트엔드가 포트 3000에서 실행 중
3. [ ] `http://localhost:5000/api/health` 직접 접속 가능
4. [ ] 브라우저 콘솔에서 `testAPI()` 성공
5. [ ] 방화벽/보안 프로그램에서 허용 설정

#### 🔧 고급 디버깅
```bash
# 백엔드 로그 확인
cd backend
python app.py > backend.log 2>&1

# 프론트엔드 상세 로그 확인
cd mopj-dashboard
REACT_APP_LOG_LEVEL=debug npm start
```

#### 📞 지원 요청 시 필요 정보
1. 운영체제 (Windows/Mac/Linux)
2. 브라우저 종류 및 버전
3. 네트워크 환경 (회사/개인/학교)
4. 브라우저 콘솔의 전체 오류 메시지
5. `testAPI()` 실행 결과
6. 백엔드 서버 실행 로그

### 4. 차트 렌더링 문제
- 브라우저 콘솔에서 JavaScript 오류 확인
- Recharts 버전 호환성 확인
- 데이터 구조 검증

### 5. VARMAX 기능 관련 문제
- VARMAX API 연결 상태 확인
- 업로드한 CSV 파일 형식 검증
- 백엔드 statsmodels 패키지 설치 확인

## 📞 지원
프론트엔드 관련 문제 발생 시 다음 순서로 디버깅하세요:
1. 브라우저 개발자 도구(F12) → Console 탭에서 `testAPI()` 실행
2. Network 탭에서 실패한 요청 확인
3. 백엔드 서버 로그 확인
4. 위 트러블슈팅 가이드 참조

---
© 2025 MOPJ 가격 예측 시스템