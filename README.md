# Coding Basic Skills

> **Version 0.1** - 

Claude Code의 강력한 확장 기능을 쉽게 사용할 수 있도록 구성된 Plugin Marketplace 입니다.

## 📦 플러그인 구조

이 프로젝트는 두 가지 주요 플러그인으로 구성되어 있습니다:

### 1. Development Plugin (`plugins/development/`)
- **Agents**: 6개 (Backend API, Frontend UI, UI/UX, Web App Tester, Debugger, Documentation)
- **Commands**: 5개 (workflow, plan, branch, complete, update-memory)
- **Skills**: 5개 (task-planner, branch-manager, memory-bank-updater, project-reviewer, mcp-builder)
- **MCP Servers**: 3개 (context7-mcp, playwright-mcp, serena-mcp)

### 2. Data Science Plugin (`plugins/data-science/`)
⚠️ **현재 에이전트 및 스킬 최적화 중 - 사용 미권장**
- **Agents**: 6개 (Data Scientist, Cleaning, Visualization, Feature Engineering, ML Modeling, Model Evaluation)
- **Skills**: 2개 (EDA, Data-cleansing) - 각 스킬별 상세 레퍼런스 포함
- **MCP Servers**: 4개 (context7-mcp, serena-mcp, jupyter-mcp-server, antv-chart-mcp)
- **상태**: 베타 테스트 단계, 안정화 작업 진행 중


## 🚀 Quick Start

### Claude Plugin Marketplace에서 설치
0. MCP 의존성 설치
  ```bash
  # home brew
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" 

  # node
  brew install node

  # uv 
  brew install uv
  ```
 
1. **Claude Code 실행**
  본 프로젝트 루트 디렉토리에서
   ```bash
   # Claude Code 앱 실행
   claude or claude --dangerously-skip-permissions # <- 전체 권한 부여모드
   ```

2. **플러그인 마켓플레이스 접속**
   - Claude Code 내에서 `/plugin` 명령어 실행
   - "Add Marketplace" 선택
   - 경로 입력(프로젝트 루트 경로라면 `./`)

3. **플러그인 검색 및 설치**
   ```
   # /plugin > Browse and install plugins 선택
   - "development" 선택
   - 설치하고자 하는 plugin 선택
   ```

4. **Claude Code 재시작**
   - 플러그인 설치 후 자동으로 재시작 안내
   - 재시작하여 플러그인 활성화

5. **플러그인 확인**
   ```bash
   # 설치된 플러그인 확인
   /plugin -> Manage and uninstall plugins
   ```

## 🤖 Agents

### Development Plugin

#### backend-api-developer
- **주요 기능**:
  - RESTful API 및 GraphQL 설계/구현
  - 데이터베이스 스키마 설계 및 최적화
  - JWT 인증, 보안, 캐싱 전략 구현
- **활성화 상황**: API 개발, 데이터베이스 작업, 서버 로직 구현 시
- **사용 MCP**: serena-mcp (코드베이스 작업), context7-mcp (라이브러리 조회)

#### frontend-ui-developer
- **주요 기능**:
  - 디자인 명세를 프로덕션 코드로 변환
  - 반응형 레이아웃 및 컴포넌트 구현
  - 접근성 표준(ARIA) 준수 구현
- **활성화 상황**: UI 컴포넌트 구현, 반응형 디자인, 와이어프레임 코드화 시
- **사용 MCP**: serena-mcp, context7-mcp

#### ui-ux-designer
- **주요 기능**:
  - 사용자 흐름도, 와이어프레임, 정보 구조도 생성
  - 접근성 및 사용성 평가
  - 디자인 시스템 및 컴포넌트 가이드 작성
- **활성화 상황**: UI 설계, 사용자 경험 개선, 인터페이스 분석 시
- **사용 MCP**: serena-mcp, context7-mcp (UI 라이브러리 조회)

#### web-app-tester
- **주요 기능**:
  - Playwright 기반 자동화 테스트 실행
  - 회귀 테스트, 성능 모니터링, UI/UX 검증
  - Backend/Frontend 에이전트 참조용 상세 보고서 작성
- **활성화 상황**: 기능 배포 후, 성능 이슈, 크로스 디바이스 테스트 시
- **사용 MCP**: playwright-mcp (브라우저 자동화), serena-mcp

#### code-debugger
- **주요 기능**:
  - 버그 검출 시스템 결과 검증 및 분석
  - 근본 원인 분석 및 포괄적 수정안 제시
  - 테스트 전략 및 모니터링 제안
- **활성화 상황**: 버그 발생, 에러 추적, 코드 품질 개선 시
- **사용 MCP**: serena-mcp, supabase-mcp (DB 디버깅), playwright-mcp

#### technical-documentation-writer
- **주요 기능**:
  - memory-bank 파일 자동 관리 및 업데이트
  - API 문서, 기술 컨텍스트, 데이터 흐름 문서화
  - 프로젝트 진행 상황 및 기술 결정 기록
- **활성화 상황**: 기능 완료, 아키텍처 변경, 테스트 결과 발생 시
- **사용 MCP**: serena-mcp

---

### Data Science Plugin

#### data-scientist
- **주요 기능**:
  - 통계 분석 (가설 검정, 회귀 분석, 시계열 분석)
  - 머신러닝 파이프라인 구축 (전처리, 모델 선택, 평가)
  - 탐색적 데이터 분석 (EDA) 및 인사이트 도출
- **활성화 상황**: 데이터 분석 프로젝트, 통계 모델링, 예측 분석 시
- **사용 MCP**: serena-mcp

#### data-cleaning-specialist
- **주요 기능**:
  - 결측치, 이상치, 중복 데이터 처리
  - 데이터 타입 변환 및 정규화
  - 데이터 품질 검증 및 보고서 작성
- **활성화 상황**: 원시 데이터 수집 후, 데이터 품질 이슈 발견 시
- **사용 MCP**: serena-mcp

#### data-visualization-specialist
- **주요 기능**:
  - 데이터 탐색용 시각화 (분포, 상관관계, 추세)
  - 인터랙티브 대시보드 구축
  - 비즈니스 프레젠테이션용 차트 생성
- **활성화 상황**: 데이터 분석 결과 전달, EDA, 보고서 작성 시
- **사용 MCP**: serena-mcp

#### feature-engineering-specialist
- **주요 기능**:
  - 도메인 지식 기반 특성 생성
  - 특성 선택 및 차원 축소
  - 범주형/수치형 특성 변환
- **활성화 상황**: 모델 성능 개선, 데이터 전처리, 특성 최적화 시
- **사용 MCP**: serena-mcp

#### ml-modeling-specialist
- **주요 기능**:
  - 지도/비지도 학습 모델 구현
  - 하이퍼파라미터 튜닝 및 교차 검증
  - 앙상블 모델 및 AutoML 활용
- **활성화 상황**: 예측 모델 구축, 분류/회귀 문제, 모델 최적화 시
- **사용 MCP**: serena-mcp

#### model-evaluation-specialist
- **주요 기능**:
  - 모델 성능 지표 계산 (정확도, F1, AUC-ROC)
  - 교차 검증 및 A/B 테스트 설계
  - 모델 해석 (SHAP, LIME) 및 비교 분석
- **활성화 상황**: 모델 성능 평가, 모델 선택, 프로덕션 배포 전
- **사용 MCP**: serena-mcp

## 📝 Commands

### /workflow
- **주요 기능**:
  - 전체 개발 워크플로우 자동화
  - 계획 → 브랜치 생성 → 작업 수행 → 메모리 업데이트 → 브랜치 병합
- **연관 요소**: task-planner, branch-manager, memory-bank-updater 스킬
- **동작 Flow**:
  1. task-planner로 작업 계획 수립
  2. branch-manager로 feature 브랜치 생성
  3. 계획에 따라 코드 작업 수행
  4. memory-bank-updater로 진행 상황 문서화
  5. branch-manager로 develop 브랜치에 병합

### /plan
- **주요 기능**:
  - 사용자 요구사항 분석 및 subtask 분해
  - 서브에이전트 할당 및 병렬 실행 계획
  - .serena/memories 컨텍스트 활용한 계획 수립
- **연관 요소**: task-planner 스킬, 모든 agents
- **동작 Flow**:
  1. .serena/memories에서 프로젝트 컨텍스트 수집
  2. 사용자 요구사항 및 제공 지침 분석
  3. 작업을 관리 가능한 subtask로 분해
  4. 각 subtask에 적절한 서브에이전트 할당
  5. 병렬 실행 가능한 작업 그룹화
  6. 계획 문서 생성 및 memories에 저장

### /branch
- **주요 기능**:
  - Git 브랜치 생명주기 관리 (생성, 병합, 삭제)
  - AI 작업(leaf)와 인간 작업(core) 브랜치 분리
  - 브랜치 명명 규칙 자동 적용
- **연관 요소**: branch-manager 스킬
- **동작 Flow**:
  - **create**: develop 기준 → 기능명 추출 → feat-<name>/leaf-<component> 생성
  - **merge**: 변경사항 커밋 확인 → develop 병합 → 테스트 → 브랜치 삭제
  - **delete**: develop 전환 → 로컬/원격 브랜치 삭제
  - **status**: 현재 브랜치 및 상태 확인

### /complete
- **주요 기능**:
  - 작업 완료 후 메모리 업데이트 및 브랜치 정리
  - 성공/실패 경로 자동 처리
  - 프로젝트 문서화 완성
- **연관 요소**: memory-bank-updater, branch-manager 스킬
- **동작 Flow**:
  - **success 경로**:
    1. memory-bank-updater로 모든 관련 메모리 파일 업데이트
    2. 최종 변경사항 커밋
    3. develop 브랜치에 병합
    4. 통합 테스트 실행
    5. 병합된 브랜치 삭제
  - **fail 경로**:
    1. 시도한 접근 방식 문서화
    2. 실패 원인 기록
    3. develop로 전환 및 실패 브랜치 삭제

### /update-memory
- **주요 기능**:
  - .serena/memories/ 파일 자동 업데이트
  - 작업 유형에 따른 적절한 파일 선택
  - 중복 방지 및 시간순 정렬
- **연관 요소**: memory-bank-updater 스킬, technical-documentation-writer agent
- **동작 Flow**:
  1. 완료된 작업 분석 (파일 변경, 기술 결정, 데이터 흐름)
  2. 업데이트 대상 파일 결정 (progress.md, techContext.md, dataflow.md 등)
  3. 각 파일에 한글로 내용 작성
  4. activeContext.md 요약하여 progress.md로 이동
  5. 중복 검증 및 시간순 정렬 확인

## 🛠️ Skills

### task-planner
- **주요 기능**:
  - .serena/memories 컨텍스트 기반 계획 수립
  - 사용자 요청만 포함하는 정확한 범위 설정
  - 병렬 실행 최적화 및 도구 사용 명세
- **활성화 순간**:
  - 사용자가 "plan this task", "create a plan", "계획 세워줘" 요청 시
  - /plan 또는 /workflow 명령어 실행 시
- **동작 Flow**:
  1. mcp__serena-mcp__list_memories로 사용 가능한 메모리 확인
  2. 관련 메모리 읽기 (이전 작업, 기술 스택, 제약사항)
  3. 사용자 요구사항 및 제공 지침 추출
  4. 작업을 subtask로 분해
  5. 각 subtask에 서브에이전트 할당
  6. 독립적인 작업 식별하여 병렬 실행 배치 구성
  7. serena 도구 사용 명세 및 검증 체크리스트 작성
  8. 계획 문서를 .serena/memories에 저장

### branch-manager
- **주요 기능**:
  - Git 브랜치 구조 관리 (main → develop → feat/leaf)
  - AI 작업(leaf)과 인간 작업(core) 분리
  - 브랜치 생명주기 자동화
- **활성화 순간**:
  - 코딩 작업 시작 전 (브랜치 생성)
  - 작업 완료 후 (병합 또는 삭제)
  - task-planner, memory-bank-updater 스킬 이후
  - 사용자가 "브랜치 관리해줘" 요청 시
- **동작 Flow**:
  - **생성**: 기능명 추출 → leaf/core 타입 결정 → feat-<name>/leaf-<component> 형식 브랜치 생성 → 검증
  - **병합**: 커밋 확인 → develop 업데이트 → 현재 브랜치 병합 → 테스트 → 브랜치 삭제
  - **삭제**: develop 전환 → 로컬/원격 브랜치 삭제 → 상태 확인

### memory-bank-updater
- **주요 기능**:
  - .serena/memories/ 파일 자동 관리
  - 작업 내용, 기술 결정, 데이터 흐름 문서화
  - activeContext → progress 이동 자동화
- **활성화 순간**:
  - 주요 작업 완료 후
  - 기술적 결정 (라이브러리, 아키텍처 선택) 시
  - 테스트 완료 후
  - 데이터 흐름 변경 시
  - /update-memory, /complete 명령어 실행 시
- **동작 Flow**:
  1. 완료된 작업 분석 (변경 파일, 기술 결정, 데이터 흐름, 테스트 결과)
  2. 작업 유형별 대상 파일 결정:
     - 새 기능 → progress.md
     - 기술 스택 변경 → techContext.md
     - 데이터 구조 변경 → dataflow.md, techContext.md
     - 테스트 → test_results/
  3. serena 도구로 각 파일 한글 작성
  4. activeContext.md 요약하여 progress.md 상단 추가
  5. 중복 검증 (같은 내용이 여러 파일에 없는지)
  6. 시간순 정렬 확인 (progress.md는 역순)

### project-reviewer
- **주요 기능**:
  - 프로젝트 온보딩 문서 자동 생성
  - 코드베이스 구조, 흐름, 빠른 시작 가이드 작성
  - serena MCP 도구로 심볼 분석 및 관계 매핑
- **활성화 순간**:
  - 사용자가 "프로젝트 리뷰", "project review" 요청 시
  - 새로운 팀원 온보딩 시
  - 프로젝트 문서화 필요 시
- **동작 Flow**:
  1. mcp__serena-mcp__list_memories 및 list_dir로 프로젝트 구조 파악
  2. get_symbols_overview로 주요 진입점 및 모듈 식별
  3. find_referencing_symbols로 모듈 간 의존성 분석
  4. /review/ 디렉토리에 4개 문서 생성:
     - project_brief.md: 프로젝트 개요 (< 200단어)
     - project_structure.md: 디렉토리 트리 및 주석
     - project_flow.md: Mermaid 다이어그램 (데이터 흐름, 모듈 상호작용)
     - quick_start.md: 사전 테스트된 설정 가이드
  5. 모든 명령어 실제 테스트 후 문서화
  6. 불릿 포인트 형식으로 간결하게 작성

### mcp-builder
- **주요 기능**:
  - 고품질 MCP 서버 개발 가이드 제공
  - TypeScript (MCP SDK) 및 Python (FastMCP) 지원
  - 외부 API/서비스 통합을 위한 도구 설계
- **활성화 순간**:
  - MCP 서버 개발 요청 시
  - 외부 서비스 통합 필요 시
  - LLM 도구 확장 개발 시
- **동작 Flow**:
  1. Phase 1: API 분석 및 MCP 프로토콜 연구
  2. Phase 2: 프로젝트 구조 설정 및 핵심 인프라 구현
  3. Phase 3: 도구 구현 (입력/출력 스키마, 에러 처리)
  4. Phase 4: 코드 품질 검토 및 테스트
  5. 평가 질문 10개 생성 및 검증

## 🔌 MCP Servers

### Development Plugin MCP Servers

| 서버명 | 유형 | 목적 |
|--------|------|------|
| context7-mcp | Node.js (npx) | 라이브러리 문서 실시간 조회 |
| playwright-mcp | Node.js (npx) | 브라우저 자동화 및 웹 테스팅 |
| serena-mcp | Python (uvx) | 코드베이스 분석 및 심볼 기반 편집 |

### Data Science Plugin MCP Servers

| 서버명 | 유형 | 목적 |
|--------|------|------|
| context7-mcp | Node.js (npx) | 라이브러리 문서 실시간 조회 |
| serena-mcp | Python (uvx) | 코드베이스 분석 및 심볼 기반 편집 |
| jupyter-mcp-server | Python (uvx) | Jupyter 노트북 연동 및 실행 |
| antv-chart-mcp | Node.js (npx) | 데이터 시각화 차트 생성 |


