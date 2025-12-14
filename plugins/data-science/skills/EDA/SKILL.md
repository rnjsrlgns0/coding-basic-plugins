---
name: eda
description: 데이터셋을 탐색하고 주요 특성, 패턴, 품질 이슈를 체계적으로 식별하는 스킬
license: Apache 2.0
---

# EDA (Exploratory Data Analysis)

## 개요

탐색적 데이터 분석(EDA)은 데이터셋을 시각화하고 요약하여 주요 특성과 패턴을 발견하는 체계적 접근법입니다. 데이터 사이언스 프로젝트의 필수 단계로, 모델링 전에 데이터를 깊이 있게 이해하고 잠재적 문제를 조기에 식별합니다.

## 사용 시점

1. 새로운 데이터셋을 처음 받았을 때
2. 모델링 전 데이터 품질 및 특성을 파악해야 할 때
3. 비즈니스 인사이트를 도출해야 할 때
4. 특성 엔지니어링 아이디어가 필요할 때
5. 데이터 기반 의사결정을 위한 근거가 필요할 때

## 핵심 워크플로우

**1단계: 데이터 로드 및 초기 검증** - 데이터 구조, 타입, 메모리 확인  
**2단계: 자동 프로파일링** - 통계 요약, 결측값, 품질 경고 생성  
**3단계: 기술 통계 분석** - 변수별 분포, 통계량, 정규성 검정  
**4단계: 관계 분석** - 변수 간 상관관계, 다중공선성, 상호작용 탐지  
**5단계: 시각화** - 분포, 관계, 패턴을 효과적으로 표현  
**6단계: 특성 중요도** - 예측 모델 관점에서 중요 변수 식별  
**7단계: 통계 검정** - 유의미한 차이, 관계, 가설 검증  
**8단계: 문서화** - 발견사항 정리, 액션 아이템 도출  

## 상황별 레퍼런스 가이드

| 상황 | 레퍼런스 | 설명 |
|------|---------|------|
| 데이터 처음 받음 | `references/01-data-loading-and-validation.md` | 다양한 형식 로딩, 기본 정보 확인, 초기 검증 |
| 자동 분석 필요 | `references/02-automated-profiling.md` | YData-Profiling, Great Expectations로 빠른 리포트 생성 |
| 변수 분포 확인 | `references/04-univariate-analysis.md` | 수치형/범주형 분포, 통계량, 정규성 검정 |
| 변수 간 관계 파악 | `references/05-bivariate-analysis.md` | 산점도, 상관계수, 그룹별 비교, 검정 |
| 다변량 분석 | `references/06-multivariate-analysis.md` | Pairplot, 3D 시각화, 차원 축소 (PCA, t-SNE, UMAP) |
| 시각화 방법 | `references/07-visualization-patterns.md` | 차트 유형, 고급 기법, 베스트 프랙티스 |
| 시계열 데이터 | `references/08-time-series-analysis.md` | 트렌드 분해, 정상성, 계절성, 이상 탐지 |
| 특성 중요도 | `references/09-feature-importance.md` | Random Forest, 순열 중요도, SHAP values |
| 상관관계 심층 분석 | `references/10-correlation-analysis.md` | 부분 상관, VIF, 비선형 관계, 인과관계 |
| 가설 검정 | `references/11-hypothesis-testing.md` | t-test, ANOVA, 카이제곱, 효과 크기 |
| 통계 추론 | `references/12-statistical-inference.md` | 신뢰구간, p-value, 검정력, 다중 비교 보정 |
| 리포트 작성 | `references/13-eda-documentation.md` | 구조화된 리포트, 자동 생성, 액션 아이템 |
| 심화 분석 | `references/14-advanced-segmentation.md` | 세그먼트, 코호트, 민감도, A/B 테스트, 생존 분석 |
| 자동화 구축 | `references/15-automation-guide.md` | 파이프라인 설계, 에이전트 매핑, 오류 처리 |

## 관련 에이전트

- **data-scientist**: 전체 EDA 프로세스 오케스트레이션, 통계 분석, 데이터 로딩, 프로파일링, 가설 검정
- **data-visualization-specialist**: 분포 시각화, 관계 시각화, 인터랙티브 플롯, 시각화 최적화
- **feature-engineering-specialist**: 특성 중요도 평가, 상호작용 탐지, Feature selection
- **data-cleaning-specialist**: 결측값 처리, 이상치 처리, 데이터 정제
- **ml-modeling-specialist**: 모델 기반 인사이트, 특성 평가
- **technical-documentation-writer**: EDA 결과 문서화, 리포트 작성

## 참고 자료

`references/` 디렉토리의 15개 문서를 참조하여 각 상황에 맞는 상세 가이드, 실행 가능한 코드, 시각화 예시, 베스트 프랙티스를 확인하세요.
