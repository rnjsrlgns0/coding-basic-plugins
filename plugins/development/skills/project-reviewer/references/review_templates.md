# 프로젝트 리뷰 템플릿

이 파일은 각 리뷰 문서에 대한 상세한 템플릿과 예시를 제공합니다.

## 템플릿 1: project_brief.md

### 구조
```markdown
# [프로젝트 이름]

## 목적
- [주요 목적을 한 줄로]

## 기술 스택
- **언어**: [주요 언어]
- **프레임워크**: [주요 프레임워크/라이브러리]
- **데이터베이스**: [해당하는 경우 데이터베이스]
- **주요 의존성**: [2-3개의 중요한 의존성]

## 핵심 기능
- [기능 1]
- [기능 2]
- [기능 3]
- [기능 4 - 선택사항]
- [기능 5 - 선택사항]

## 사용 사례
- [누가 사용하며 왜 사용하는지]
```

### 예시
```markdown
# TaskFlow API

## 목적
- 실시간 협업을 지원하는 작업 관리용 RESTful API

## 기술 스택
- **언어**: Python 3.11
- **프레임워크**: FastAPI
- **데이터베이스**: PostgreSQL + Redis
- **주요 의존성**: SQLAlchemy, Pydantic, WebSocket

## 핵심 기능
- 역할 기반 접근 제어를 갖춘 작업 CRUD 작업
- WebSocket을 통한 실시간 작업 업데이트
- 사용자 인증 (JWT)
- 작업 할당 및 알림
- 활동 로깅 및 감사 추적

## 사용 사례
- 즉각적인 업데이트가 필요한 팀 협업 작업 추적
```

---

## 템플릿 2: project_structure.md

### 구조
```markdown
# 프로젝트 구조

## 디렉토리 트리

\`\`\`
project-root/
├── [디렉토리1]/              # [목적]
│   ├── [하위디렉토리]/        # [목적]
│   └── [주요파일]             # [목적]
├── [디렉토리2]/              # [목적]
│   ├── [모듈1]/               # [목적]
│   └── [모듈2]/               # [목적]
├── [설정파일]                 # [목적]
└── [진입점]                   # [목적]
\`\`\`

## 주요 디렉토리

### [디렉토리 1]
- **목적**: [포함 내용]
- **주요 파일**: [중요한 파일과 그 역할]

### [디렉토리 2]
- **목적**: [포함 내용]
- **주요 파일**: [중요한 파일과 그 역할]

## 중요 파일

- **[파일1]**: [역할 및 중요성]
- **[파일2]**: [역할 및 중요성]
```

### 예시
```markdown
# 프로젝트 구조

## 디렉토리 트리

\`\`\`
taskflow-api/
├── src/                 # 애플리케이션 소스 코드
│   ├── api/             # API 엔드포인트 및 라우트
│   │   ├── v1/          # API 버전 1
│   │   └── deps.py      # 공유 의존성
│   ├── core/            # 핵심 설정 및 보안
│   │   ├── config.py    # 앱 설정
│   │   └── security.py  # 인증 핸들러
│   ├── models/          # 데이터베이스 모델 (SQLAlchemy)
│   ├── schemas/         # Pydantic 스키마
│   └── services/        # 비즈니스 로직
├── tests/               # 테스트 스위트
│   ├── unit/            # 단위 테스트
│   └── integration/     # 통합 테스트
├── alembic/             # 데이터베이스 마이그레이션
├── main.py              # 애플리케이션 진입점
├── requirements.txt     # Python 의존성
└── .env.example         # 환경 변수 템플릿
\`\`\`

## 주요 디렉토리

### src/api/
- **목적**: API 라우트 핸들러 및 엔드포인트 정의
- **주요 파일**: `deps.py` (의존성 주입), `v1/tasks.py` (작업 엔드포인트)

### src/models/
- **목적**: 데이터베이스 테이블을 위한 SQLAlchemy ORM 모델
- **주요 파일**: `task.py`, `user.py`, `activity.py`

### src/services/
- **목적**: API와 데이터베이스 간의 비즈니스 로직 계층
- **주요 파일**: `task_service.py`, `notification_service.py`

## 중요 파일

- **main.py**: FastAPI 앱 초기화, 미들웨어 설정, 라우트 등록
- **src/core/config.py**: Pydantic BaseSettings를 사용한 중앙 집중식 설정
- **requirements.txt**: 재현 가능한 빌드를 위한 고정된 의존성
```

---

## 템플릿 3: project_flow.md

### 구조
```markdown
# 프로젝트 흐름

## 데이터 흐름 다이어그램

\`\`\`mermaid
[데이터 흐름을 보여주는 Mermaid 다이어그램]
\`\`\`

## 모듈 상호작용 다이어그램

\`\`\`mermaid
[모듈 관계를 보여주는 Mermaid 다이어그램]
\`\`\`

## [특정 프로세스 흐름 - 해당하는 경우]

\`\`\`mermaid
[인증, 요청 처리 등 특정 프로세스를 위한 Mermaid 다이어그램]
\`\`\`

## 주요 상호작용

- **[모듈 A] → [모듈 B]**: [모듈 간 흐르는 데이터/호출]
- **[모듈 C] → [모듈 D]**: [모듈 간 흐르는 데이터/호출]
```

### 예시
```markdown
# 프로젝트 흐름

## 요청 흐름 다이어그램

\`\`\`mermaid
sequenceDiagram
    participant Client
    participant API
    participant Auth
    participant Service
    participant DB
    participant WebSocket

    Client->>API: POST /api/v1/tasks
    API->>Auth: JWT 토큰 검증
    Auth-->>API: 사용자 인증됨
    API->>Service: 작업 생성
    Service->>DB: 작업 레코드 삽입
    DB-->>Service: 작업 생성됨
    Service->>WebSocket: 작업 업데이트 브로드캐스트
    WebSocket-->>Client: 실시간 알림
    Service-->>API: 작업 응답
    API-->>Client: 201 Created
\`\`\`

## 모듈 상호작용 다이어그램

\`\`\`mermaid
graph TB
    API[API 라우트] --> Services[비즈니스 서비스]
    API --> Auth[인증 미들웨어]
    Services --> Models[데이터베이스 모델]
    Services --> Schemas[Pydantic 스키마]
    Models --> DB[(PostgreSQL)]
    Services --> Cache[(Redis 캐시)]
    Services --> WS[WebSocket 매니저]

    Auth -.-> Cache
    WS --> Client[연결된 클라이언트]
\`\`\`

## 인증 흐름

\`\`\`mermaid
flowchart LR
    A[로그인 요청] --> B{자격증명 검증}
    B -->|유효| C[JWT 생성]
    B -->|무효| D[401 에러]
    C --> E[Redis에 저장]
    E --> F[토큰 반환]
    F --> G[보호된 엔드포인트]
    G --> H{JWT 검증}
    H -->|유효| I[요청 처리]
    H -->|무효| D
\`\`\`

## 주요 상호작용

- **API → Services**: 비즈니스 로직 위임, 검증된 스키마 전달
- **Services → Models**: SQLAlchemy를 통한 ORM 작업 (CRUD)
- **Services → WebSocket**: 연결된 클라이언트에게 실시간 업데이트 브로드캐스트
- **Auth 미들웨어 → Cache**: 빠른 조회를 위해 Redis에서 JWT 검증
```

---

## 템플릿 4: quick_start.md

### 구조
```markdown
# 빠른 시작 가이드

## 사전 요구사항
- [버전이 포함된 요구사항 1]
- [버전이 포함된 요구사항 2]
- [버전이 포함된 요구사항 3]

## 환경 설정

### 1. [단계 1 제목]
\`\`\`bash
[명령어]
\`\`\`
**예상**: [발생할 결과]

### 2. [단계 2 제목]
\`\`\`bash
[명령어]
\`\`\`
**예상**: [발생할 결과]

[모든 설정 단계 계속...]

## 구성

### [구성 측면 1]
\`\`\`bash
[명령어 또는 파일 편집]
\`\`\`

### [구성 측면 2]
\`\`\`bash
[명령어 또는 파일 편집]
\`\`\`

## 애플리케이션 실행

### 개발 모드
\`\`\`bash
[명령어]
\`\`\`
**출력**: [예상 출력 또는 URL]

### [해당하는 경우 대체 실행 모드]
\`\`\`bash
[명령어]
\`\`\`
**출력**: [예상 출력]

## 일반 명령어

| 명령어 | 목적 |
|---------|---------|
| \`[명령어]\` | [수행 작업] |
| \`[명령어]\` | [수행 작업] |
| \`[명령어]\` | [수행 작업] |

## 검증

- [ ] [확인 1]
- [ ] [확인 2]
- [ ] [확인 3]

## 테스트된 환경
- **OS**: [테스트된 OS]
- **Python/Node**: [버전]
- **날짜**: [테스트 날짜]
```

### 예시
```markdown
# 빠른 시작 가이드

## 사전 요구사항
- Python 3.11+
- PostgreSQL 14+
- Redis 6+
- pip 또는 uv 패키지 매니저

## 환경 설정

### 1. 클론 및 이동
\`\`\`bash
cd taskflow-api
\`\`\`

### 2. 가상 환경 생성
\`\`\`bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows
\`\`\`
**예상**: 프롬프트에 `(.venv)` 표시

### 3. 의존성 설치
\`\`\`bash
pip install -r requirements.txt
\`\`\`
**예상**: 약 45개 패키지 설치됨, 에러 없음

### 4. 필수 서비스 시작
\`\`\`bash
# PostgreSQL (실행 중이 아닌 경우)
brew services start postgresql@14  # Mac
# sudo service postgresql start    # Linux

# Redis
brew services start redis          # Mac
# sudo service redis-server start  # Linux
\`\`\`
**예상**: 기본 포트(5432, 6379)에서 서비스 실행 중

## 구성

### 데이터베이스 설정
\`\`\`bash
# 데이터베이스 생성
createdb taskflow_db

# 환경 변수 템플릿 복사
cp .env.example .env

# 데이터베이스 자격증명으로 .env 편집
# DATABASE_URL=postgresql://user:password@localhost:5432/taskflow_db
\`\`\`

### 마이그레이션 실행
\`\`\`bash
alembic upgrade head
\`\`\`
**예상**: "Running upgrade ... -> ..., create initial tables"

## 애플리케이션 실행

### 개발 모드
\`\`\`bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
\`\`\`
**출력**:
\`\`\`
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
\`\`\`
**접속**: API 문서는 http://localhost:8000/docs

### 프로덕션 모드
\`\`\`bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
\`\`\`

## 일반 명령어

| 명령어 | 목적 |
|---------|---------|
| \`pytest\` | 모든 테스트 실행 |
| \`pytest tests/unit\` | 단위 테스트만 실행 |
| \`alembic revision --autogenerate -m "msg"\` | 새 마이그레이션 생성 |
| \`alembic upgrade head\` | 마이그레이션 적용 |
| \`python -m src.scripts.seed_data\` | 샘플 데이터 시드 |

## 검증

- [ ] 서버가 에러 없이 시작됨
- [ ] http://localhost:8000/docs 에서 API 문서 접근 가능
- [ ] 데이터베이스에 예상 테이블 존재 (tasks, users, activities)
- [ ] WebSocket 연결 작동 (/ws 엔드포인트 확인)
- [ ] 헬스 체크 200 반환: http://localhost:8000/health

## 문제 해결

**포트가 이미 사용 중**:
\`\`\`bash
lsof -ti:8000 | xargs kill -9
\`\`\`

**데이터베이스 연결 실패**:
- PostgreSQL 실행 여부 확인: \`pg_isready\`
- .env 파일의 자격증명 확인

## 테스트된 환경
- **OS**: macOS Sonoma 14.2, Ubuntu 22.04
- **Python**: 3.11.6
- **PostgreSQL**: 14.10
- **Redis**: 6.2.7
- **날짜**: 2025-01-15
```

---

## 참고사항

- 모든 콘텐츠를 간결하고 훑어보기 쉽게 유지
- 문서화하기 전에 모든 명령어 테스트
- 프로젝트의 실제 파일/디렉토리 이름 사용
- 해당하는 경우 버전 번호 포함
- 자리 표시자가 아닌 실제 예시 제공
