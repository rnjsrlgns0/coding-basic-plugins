---
name: data-cleansing
description: 데이터셋에서 부정확하거나 손상된 데이터를 식별하고 수정하여 분석과 모델링에 적합한 고품질 데이터로 변환하는 프로세스
license: Apache 2.0
---

# Data-cleansing

## 개요

데이터 클렌징(Data Cleansing)은 데이터 분석 및 머신러닝 프로젝트의 필수 단계입니다. 결측값, 이상치, 중복, 형식 불일치 등의 품질 이슈를 체계적으로 해결하여 신뢰할 수 있는 데이터를 확보합니다. 데이터 사이언스 프로젝트 시간의 약 60-80%가 데이터 준비와 클렌징에 소요되므로, 효율적인 클렌징은 전체 프로젝트의 성공을 좌우합니다.

## 사용 시점

1. **새 데이터셋 수신**: 데이터 품질 현황을 파악하고 클렌징이 필요한 이슈 식별
2. **결측값 처리**: NaN, None 등의 빈 값을 분석하고 적절한 전략으로 대체
3. **이상치 탐지 및 처리**: 극단값을 식별하고 제거 또는 보정 결정
4. **데이터 표준화**: 형식 불일치, 공백, 대소문자 등의 일관성 문제 해결
5. **중복 제거**: 완전 중복이나 키 기반 중복 레코드 제거
6. **ML 전처리**: 모델 학습을 위한 수치형 정규화 및 범주형 인코딩
7. **데이터 검증**: 클렌징 후 품질 확인 및 비즈니스 규칙 검증
8. **자동화 파이프라인**: 반복적인 클렌징 작업 자동화 및 모니터링

## 핵심 워크플로우

**1단계: 품질 평가** → 현재 데이터의 품질 상태를 정량적으로 파악합니다.

**2단계: 이슈 식별** → 결측값, 이상치, 중복, 형식 불일치 등의 구체적인 문제를 목록화합니다.

**3단계: 전략 수립** → 데이터 특성과 비즈니스 요구사항에 맞는 처리 방법을 결정합니다.

**4단계: 클렌징 실행** → 각 이슈별로 적절한 기법을 적용하여 데이터를 정제합니다.

**5단계: 검증** → 클렌징된 데이터가 필요한 품질 기준을 충족하는지 확인합니다.

**6단계: 문서화** → 변환 과정과 의사결정을 기록하여 재현성을 보장합니다.

## 상황별 레퍼런스 가이드

| 상황 | 레퍼런스 | 설명 |
|------|---------|------|
| 데이터 품질 평가 | `references/01-data-quality-assessment.md` | 종합 프로파일링, 타입 검증, 품질 메트릭 |
| 결측값 분석 | `references/02-missing-data-patterns.md` | 결측값 메커니즘 이해, 패턴 시각화 |
| 결측값 처리 | `references/03-imputation-strategies.md` | mean/median, KNN, MICE 등 다양한 대체 전략 |
| 통계적 이상치 탐지 | `references/04-statistical-outlier-detection.md` | Z-Score, IQR, Modified Z-Score 방법 |
| ML 이상치 탐지 | `references/05-ml-outlier-detection.md` | Isolation Forest, LOF, One-Class SVM 등 고급 기법 |
| 이상치 처리 | `references/06-outlier-treatment.md` | Capping, 변환, 제거, 분리 모델링 |
| 중복 제거 | `references/07-duplicate-handling.md` | 완전/부분 중복 탐지, Fuzzy matching, 병합 전략 |
| 형식 표준화 | `references/08-data-standardization.md` | 날짜, 문자열, 범주형, 특수 형식 통일 |
| 수치형 정규화 | `references/09-data-normalization.md` | StandardScaler, MinMaxScaler, RobustScaler 선택 가이드 |
| 범주형 인코딩 | `references/10-categorical-encoding.md` | Label, One-Hot, Ordinal, Target Encoding 비교 |
| 데이터 검증 | `references/11-data-validation.md` | 교차 필드 검증, 참조 무결성, 비즈니스 규칙 검증 |
| 품질 리포트 | `references/12-quality-reporting.md` | 전후 비교 리포트, HTML 리포트 생성 |
| 데이터 리니지 | `references/13-data-lineage.md` | 변환 이력 추적, 재현성 보장 |
| 자동 검증 프레임워크 | `references/14-great-expectations-guide.md` | Great Expectations를 활용한 자동화 검증 |
| 자동화 파이프라인 | `references/15-automation-pipeline.md` | 클렌징 파이프라인 설계, 에이전트 오케스트레이션 |

## 관련 에이전트

- **data-cleaning-specialist**: 모든 클렌징 작업의 주담당. 결측값, 이상치, 중복, 표준화 처리 및 검증 수행
- **data-scientist**: 데이터 품질 평가, 클렌징 전략 수립, 비즈니스 로직 검증
- **data-visualization-specialist**: 클렌징 전후 비교 시각화, 품질 이슈 시각화
- **feature-engineering-specialist**: 데이터 변환, 정규화, 범주형 인코딩
- **technical-documentation-writer**: 클렌징 리포트 작성, 프로세스 문서화

## 참고 자료

`references/` 디렉토리에 15개의 상세 가이드가 준비되어 있습니다. 각 문서는 독립적으로 학습할 수 있으며, 실제 클렌징 작업에 필요한 구체적인 코드, 함수, 선택 가이드를 포함하고 있습니다.
