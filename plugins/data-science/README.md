# Data Science Plugin

데이터 분석, 머신러닝, 과학 컴퓨팅을 위한 종합 플러그인입니다.

## 🎯 목적

데이터 사이언스 워크플로우 전체를 지원하는 에이전트와 50개 이상의 과학 컴퓨팅 스킬을 제공합니다.

## 🤖 포함된 에이전트

### 데이터 분석
- **data-scientist**: 탐색적 데이터 분석(EDA), 통계 모델링, 가설 검정
- **data-cleaning-specialist**: 결측치 처리, 이상치 탐지, 데이터 검증

### 시각화
- **data-visualization-specialist**: 차트, 플롯, 대시보드 생성 (matplotlib, seaborn, plotly)

### 특성 엔지니어링
- **feature-engineering-specialist**: 특성 생성, 인코딩, 스케일링, 차원 축소

### 머신러닝
- **ml-modeling-specialist**: 모델 선택, 학습, 하이퍼파라미터 튜닝, AutoML
- **model-evaluation-specialist**: 성능 메트릭, 혼동 행렬, ROC 곡선, 모델 비교

## 🔬 포함된 Scientific Skills (50개)

### 시계열 분석
- **aeon**: 시계열 머신러닝 (분류, 회귀, 클러스터링, 이상 탐지)

### 생물정보학 & 의학 데이터베이스
- **chembl-database**: 생물활성 분자 및 약물 발견 데이터
- **clinicaltrials-database**: 임상시험 정보 조회
- **clinvar-database**: 유전변이 임상 의미
- **cosmic-database**: 암 돌연변이 데이터
- **drugbank-database**: 약물 정보 및 상호작용
- **ena-database**: DNA/RNA 서열 데이터
- **ensembl-database**: 게놈 데이터베이스 (250+ 종)
- **gene-database**: NCBI 유전자 정보
- **geo-database**: 유전자 발현 데이터
- **gwas-database**: SNP-질병 연관성
- **hmdb-database**: 인간 대사체 데이터베이스
- **pdb-database**: 단백질 구조 데이터
- **pubchem-database**: 화학 물질 정보
- **pubmed-database**: 의학 문헌 검색
- **uniprot-database**: 단백질 서열 및 기능
- **zinc-database**: 화합물 라이브러리

### 문서 처리
- **docx**: Word 문서 생성/편집
- **pdf**: PDF 조작 및 데이터 추출
- **pptx**: PowerPoint 프레젠테이션
- **markitdown**: 다양한 형식을 Markdown으로 변환
- **reportlab**: PDF 보고서 생성

### 시각화
- **matplotlib**: 기본 플로팅 라이브러리
- **seaborn**: 통계 시각화
- **networkx**: 네트워크/그래프 시각화

### 머신러닝 & 딥러닝
- **scikit-learn**: 전통적 머신러닝
- **pytorch-lightning**: PyTorch 학습 프레임워크
- **transformers**: Hugging Face 트랜스포머 모델
- **stable-baselines3**: 강화학습
- **pufferlib**: 강화학습 유틸리티
- **shap**: 모델 해석성

### 통계 & 수학
- **statsmodels**: 통계 모델링
- **sympy**: 기호 수학
- **pymc**: 베이지안 추론
- **pymoo**: 다목적 최적화

### 생존 분석
- **scikit-survival**: 생존 분석 및 시간-이벤트 모델링

### 차원 축소
- **umap-learn**: 차원 축소 및 시각화

### 연구 도구
- **literature-review**: 체계적 문헌 검토
- **peer-review**: 논문 피어 리뷰
- **scientific-writing**: 과학 논문 작성
- **paper-2-web**: 논문을 웹 콘텐츠로 변환

### 기타 데이터베이스
- **datacommons-client**: 공공 통계 데이터
- **fda-database**: FDA 규제 정보
- **kegg-database**: 대사 경로 데이터
- **metabolomics-workbench**: 대사체학 데이터
- **opentargets-database**: 질병-타겟 연관성
- **reactome-database**: 생물학적 경로
- **string-database**: 단백질 상호작용
- **uspto-database**: 특허 정보

## 📋 사용 시나리오

### 머신러닝 파이프라인
```
1. data-cleaning-specialist로 데이터 전처리
2. data-visualization-specialist로 EDA 수행
3. feature-engineering-specialist로 특성 생성
4. ml-modeling-specialist로 모델 학습
5. model-evaluation-specialist로 성능 평가
```

### 생물정보학 분석
```
1. ensembl-database로 유전자 정보 조회
2. clinvar-database로 변이 의미 확인
3. data-scientist로 통계 분석
4. data-visualization-specialist로 결과 시각화
```

### 논문 작성 워크플로우
```
1. literature-review로 문헌 조사
2. data-scientist로 데이터 분석
3. data-visualization-specialist로 그래프 생성
4. scientific-writing로 논문 작성
5. reportlab로 PDF 보고서 생성
```

### 시계열 예측
```
1. data-cleaning-specialist로 데이터 정제
2. aeon 스킬로 시계열 분석
3. data-visualization-specialist로 트렌드 시각화
4. model-evaluation-specialist로 예측 성능 평가
```

## 🔧 설정

### 플러그인 활성화
`.claude-plugin/marketplace.json`에서:
```json
{
  "plugins": [
    {
      "name": "data-science",
      "enabled": true
    }
  ]
}
```

### 프로필 사용
`.claude/profiles.json`에서 `data-science` 프로필 선택

## 📂 디렉토리 구조

```
data-science/
├── .claude-plugin/
│   └── marketplace.json       # 플러그인 매니페스트
├── agents/                    # 데이터 사이언스 에이전트
│   ├── data-scientist.md
│   ├── data-cleaning-specialist.md
│   ├── data-visualization-specialist.md
│   ├── feature-engineering-specialist.md
│   ├── ml-modeling-specialist.md
│   └── model-evaluation-specialist.md
├── scientific-skills/         # 50개 과학 스킬 (symlink)
│   └── scientific-skills/
│       ├── aeon/
│       ├── matplotlib/
│       ├── scikit-learn/
│       └── ... (47 more)
└── README.md                  # 이 파일
```

## 🚀 시작하기

1. 프로필을 `data-science`로 설정
2. 데이터 전처리: `data-cleaning-specialist` 호출
3. 탐색적 분석: `data-scientist` + `data-visualization-specialist`
4. 모델링: `feature-engineering-specialist` → `ml-modeling-specialist`
5. 평가: `model-evaluation-specialist`

## 💡 팁

- 복잡한 ML 파이프라인은 에이전트를 순차적으로 활용
- 생물정보학 작업 시 관련 database 스킬 먼저 확인
- 시각화는 matplotlib → seaborn 순서로 시도
- 문서 작성 시 reportlab이나 scientific-writing 활용
- 통계 분석은 statsmodels나 pymc 사용

## 🔗 참고 자료

- [K-Dense Scientific Skills](https://github.com/K-Dense/claude-scientific-skills)
- 각 스킬의 상세 문서는 `scientific-skills/scientific-skills/{skill-name}/SKILL.md` 참조
