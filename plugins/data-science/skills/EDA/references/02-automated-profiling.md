# Reference 02: Automated Profiling

**Version**: 1.0  
**Last Updated**: 2025-01-25  
**Workflow Phase**: Phase 2 - Automated Profiling  
**Estimated Reading Time**: 20-25 minutes

---

## 1. 개요 (Overview)

### 1.1 목적 (Purpose)

자동화된 프로파일링(Automated Profiling)은 최소한의 코드로 데이터셋의 포괄적인 특성을 분석하고 시각화하는 프로세스입니다. 수동으로 수십 줄의 코드를 작성하는 대신, 단 한 줄의 명령으로 데이터의 전체적인 모습을 파악할 수 있습니다.

**주요 목적**:
- 빠른 데이터 이해 (단 몇 분 내)
- 데이터 품질 이슈 자동 탐지
- 변수 간 관계 자동 분석
- 인터랙티브 HTML 리포트 생성
- 반복적인 EDA 작업 자동화

### 1.2 적용 시기 (When to Apply)

자동화된 프로파일링은 다음과 같은 상황에서 특히 유용합니다:

1. **신규 데이터셋 첫 분석**: 데이터의 전반적 특성을 빠르게 파악
2. **클라이언트 미팅 전**: 데이터 품질 리포트 사전 준비
3. **데이터 품질 모니터링**: 주기적인 데이터 검증
4. **다수의 데이터셋 비교**: 여러 데이터셋의 특성 빠른 비교
5. **프로젝트 초기 단계**: 심층 분석 방향 설정

### 1.3 도구 선택 가이드

| 도구 | 장점 | 단점 | 추천 시나리오 |
|------|------|------|---------------|
| **YData Profiling** | 가장 포괄적, 상관관계 분석 우수 | 대용량 데이터 느림 | 중소 규모 데이터, 포괄적 분석 |
| **Great Expectations** | 데이터 품질 검증 특화, 자동화 우수 | 시각화 부족 | 프로덕션 파이프라인, 품질 검증 |
| **Sweetviz** | 타겟 변수 분석, 데이터셋 비교 | 커스터마이징 제한적 | 타겟 중심 분석, A/B 비교 |
| **DataPrep** | 속도 빠름, 심플한 인터페이스 | 상세 분석 부족 | 대용량 데이터, 빠른 개요 |

---

## 2. 이론적 배경 (Theoretical Background)

### 2.1 자동화 프로파일링의 가치

**시간 절약**: 수동으로 작성하면 100-200 줄의 코드가 필요한 분석을 1-2줄로 실행

**일관성**: 표준화된 분석 프로세스로 누락 방지

**포괄성**: 사람이 놓칠 수 있는 패턴과 이슈 자동 탐지

**재현성**: 동일한 분석을 여러 데이터셋에 일관되게 적용

### 2.2 프로파일링 구성 요소

자동화된 프로파일링은 일반적으로 다음 요소를 포함합니다:

1. **Overview (개요)**
   - 데이터셋 크기 (행/열)
   - 변수 타입 분포
   - 결측값 통계
   - 중복 행 비율
   - 메모리 사용량

2. **Variables (변수별 분석)**
   - 기술 통계량 (평균, 중앙값, 표준편차 등)
   - 분포 시각화 (히스토그램, KDE)
   - 고유값 개수
   - 결측값 비율
   - 이상치 탐지

3. **Interactions (상호작용)**
   - 산점도 매트릭스
   - 변수 쌍별 관계

4. **Correlations (상관관계)**
   - Pearson, Spearman, Kendall 상관계수
   - 히트맵 시각화
   - 높은 상관관계 경고

5. **Missing Values (결측값)**
   - 결측값 패턴 분석
   - 히트맵 시각화
   - 결측값 간 상관관계

6. **Sample (샘플 데이터)**
   - 상위/하위 데이터
   - 무작위 샘플

7. **Alerts (경고)**
   - 높은 상관관계
   - 높은 카디널리티
   - 높은 왜도
   - 불균형 데이터
   - 상수 값
   - 고유값

### 2.3 일반적인 시나리오

**시나리오 1: 초기 데이터 탐색**
- 새로운 프로젝트 시작 시
- 데이터의 전반적 구조와 품질 파악
- 심층 분석 방향 설정

**시나리오 2: 데이터 품질 검증**
- 데이터 파이프라인 검증
- 정기적 품질 모니터링
- 예상 스키마와 실제 비교

**시나리오 3: 타겟 변수 중심 분석**
- 분류/회귀 문제의 타겟 변수 분석
- 타겟과 피처 간 관계 파악
- 불균형 탐지

**시나리오 4: 데이터셋 비교**
- Train vs Test 데이터 비교
- Before vs After 데이터 비교
- 데이터 드리프트 탐지

---

## 3. 구현 (Implementation)

### 3.1 YData Profiling (Pandas Profiling)

#### 3.1.1 기본 프로파일링

```python
import pandas as pd
from ydata_profiling import ProfileReport
import warnings
warnings.filterwarnings('ignore')

def generate_ydata_profile(
    df: pd.DataFrame,
    title: str = "Data Profile Report",
    output_file: str = "profile_report.html",
    minimal: bool = False,
    explorative: bool = False,
    config: dict = None
) -> ProfileReport:
    """
    YData Profiling을 사용한 포괄적 데이터 프로파일링
    
    Parameters:
    -----------
    df : pd.DataFrame
        프로파일링할 데이터프레임
    title : str, default "Data Profile Report"
        리포트 제목
    output_file : str, default "profile_report.html"
        출력 HTML 파일 경로
    minimal : bool, default False
        최소 모드 (빠른 실행)
    explorative : bool, default False
        탐색 모드 (더 많은 시각화)
    config : dict, optional
        커스텀 설정
    
    Returns:
    --------
    ProfileReport
        생성된 프로파일 리포트 객체
    
    Examples:
    ---------
    >>> profile = generate_ydata_profile(df, title="Sales Data Profile")
    >>> profile.to_file("sales_profile.html")
    """
    print(f"YData Profiling 시작...")
    print(f"  데이터 크기: {df.shape}")
    
    # 기본 설정
    if config is None:
        config = {}
    
    # 모드별 설정
    if minimal:
        config.update({
            "correlations": None,  # 상관관계 비활성화
            "missing_diagrams": None,  # 결측값 다이어그램 비활성화
            "interactions": None,  # 상호작용 비활성화
            "samples": None  # 샘플 비활성화
        })
    
    try:
        # 프로파일 생성
        profile = ProfileReport(
            df,
            title=title,
            explorative=explorative,
            minimal=minimal,
            **config
        )
        
        # HTML 파일로 저장
        profile.to_file(output_file)
        
        print(f"✓ 프로파일 생성 완료: {output_file}")
        
        # 주요 통계 출력
        print("\n[주요 통계]")
        print(f"  변수 개수: {len(df.columns)}")
        print(f"  관측치 개수: {len(df):,}")
        print(f"  결측값 셀: {df.isnull().sum().sum():,}")
        print(f"  중복 행: {df.duplicated().sum():,}")
        
        return profile
    
    except Exception as e:
        print(f"✗ 프로파일 생성 실패: {e}")
        raise


# 사용 예시 1: 기본 프로파일링
profile = generate_ydata_profile(
    df,
    title="Customer Data Profile",
    output_file="reports/customer_profile.html"
)

# 사용 예시 2: 빠른 프로파일링 (대용량 데이터)
profile_minimal = generate_ydata_profile(
    df,
    title="Quick Profile",
    output_file="reports/quick_profile.html",
    minimal=True
)

# 사용 예시 3: 탐색적 프로파일링 (상세 분석)
profile_explorative = generate_ydata_profile(
    df,
    title="Detailed Exploration",
    output_file="reports/detailed_profile.html",
    explorative=True
)
```

#### 3.1.2 커스텀 설정 프로파일링

```python
def generate_custom_profile(
    df: pd.DataFrame,
    title: str = "Custom Profile",
    output_file: str = "custom_profile.html",
    dark_mode: bool = True,
    correlation_threshold: float = 0.9,
    sample_size: int = 10
) -> ProfileReport:
    """
    커스터마이즈된 프로파일링 설정
    
    Parameters:
    -----------
    df : pd.DataFrame
        프로파일링할 데이터프레임
    title : str
        리포트 제목
    output_file : str
        출력 HTML 파일 경로
    dark_mode : bool, default True
        다크 모드 활성화
    correlation_threshold : float, default 0.9
        상관관계 경고 임계값
    sample_size : int, default 10
        샘플 데이터 행 개수
    
    Returns:
    --------
    ProfileReport
        프로파일 리포트 객체
    
    Examples:
    ---------
    >>> profile = generate_custom_profile(
    ...     df,
    ...     title="My Custom Report",
    ...     dark_mode=True,
    ...     correlation_threshold=0.85
    ... )
    """
    # 커스텀 설정
    config = {
        # 외관 설정
        "html": {
            "style": {
                "theme": "united" if not dark_mode else "flatly"
            }
        },
        
        # 상관관계 설정
        "correlations": {
            "auto": {
                "calculate": True,
                "warn_high_correlations": True,
                "threshold": correlation_threshold
            },
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "kendall": {"calculate": False},  # 느려서 비활성화
            "phi_k": {"calculate": True},
            "cramers": {"calculate": True}
        },
        
        # 결측값 설정
        "missing_diagrams": {
            "bar": True,
            "matrix": True,
            "heatmap": True
        },
        
        # 상호작용 설정
        "interactions": {
            "continuous": True,
            "targets": []
        },
        
        # 샘플 설정
        "samples": {
            "head": sample_size,
            "tail": sample_size,
            "random": sample_size
        },
        
        # 변수별 설정
        "variables": {
            "descriptions": {},
            "num": {
                "low_categorical_threshold": 5,  # 5개 이하면 범주형으로 간주
                "chi_squared_threshold": 0.999
            }
        },
        
        # 정렬 설정
        "sort": "ascending",
        
        # 알림 설정
        "alerts": {
            "high_cardinality": True,
            "high_correlation": True,
            "skewed": True,
            "imbalance": True,
            "constant": True,
            "zeros": True,
            "uniform": True,
            "unique": True
        }
    }
    
    print(f"커스텀 프로파일 생성 중...")
    print(f"  - 다크 모드: {dark_mode}")
    print(f"  - 상관관계 임계값: {correlation_threshold}")
    print(f"  - 샘플 크기: {sample_size}")
    
    # 프로파일 생성
    profile = ProfileReport(
        df,
        title=title,
        config_file=None,
        **config
    )
    
    # 저장
    profile.to_file(output_file)
    print(f"✓ 커스텀 프로파일 저장: {output_file}")
    
    return profile


# 사용 예시
profile = generate_custom_profile(
    df,
    title="Sales Analysis 2024",
    output_file="reports/sales_custom.html",
    dark_mode=True,
    correlation_threshold=0.85,
    sample_size=15
)
```

#### 3.1.3 타겟 변수 중심 프로파일링

```python
def generate_target_focused_profile(
    df: pd.DataFrame,
    target_column: str,
    title: str = "Target-Focused Profile",
    output_file: str = "target_profile.html"
) -> ProfileReport:
    """
    타겟 변수를 중심으로 한 프로파일링
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    target_column : str
        타겟 변수 컬럼명
    title : str
        리포트 제목
    output_file : str
        출력 파일 경로
    
    Returns:
    --------
    ProfileReport
        프로파일 리포트
    
    Examples:
    ---------
    >>> profile = generate_target_focused_profile(
    ...     df,
    ...     target_column='churn',
    ...     title="Churn Prediction Analysis"
    ... )
    """
    if target_column not in df.columns:
        raise ValueError(f"타겟 컬럼 '{target_column}'이 존재하지 않습니다.")
    
    print(f"타겟 변수 중심 프로파일링...")
    print(f"  타겟 변수: {target_column}")
    
    # 타겟 변수 분포 확인
    print(f"\n[타겟 변수 분포]")
    target_dist = df[target_column].value_counts()
    for val, count in target_dist.items():
        pct = 100 * count / len(df)
        print(f"  {val}: {count:,} ({pct:.2f}%)")
    
    # 타겟 중심 설정
    config = {
        "interactions": {
            "continuous": True,
            "targets": [target_column]  # 타겟 변수 지정
        },
        "correlations": {
            "auto": {"calculate": True},
            "pearson": {"calculate": True},
            "spearman": {"calculate": True}
        }
    }
    
    # 프로파일 생성
    profile = ProfileReport(
        df,
        title=f"{title} (Target: {target_column})",
        explorative=True,
        **config
    )
    
    profile.to_file(output_file)
    print(f"\n✓ 타겟 중심 프로파일 저장: {output_file}")
    
    return profile


# 사용 예시
profile = generate_target_focused_profile(
    df,
    target_column='purchase',
    title="Purchase Prediction Analysis",
    output_file="reports/purchase_target_profile.html"
)
```

### 3.2 Great Expectations

#### 3.2.1 기본 데이터 품질 검증

```python
import great_expectations as gx
from great_expectations.data_context import EphemeralDataContext

def validate_with_great_expectations(
    df: pd.DataFrame,
    expectation_suite_name: str = "default_suite",
    auto_create: bool = True
) -> dict:
    """
    Great Expectations를 사용한 데이터 품질 검증
    
    Parameters:
    -----------
    df : pd.DataFrame
        검증할 데이터프레임
    expectation_suite_name : str
        Expectation Suite 이름
    auto_create : bool, default True
        Expectation을 자동 생성할지 여부
    
    Returns:
    --------
    dict
        검증 결과
    
    Examples:
    ---------
    >>> results = validate_with_great_expectations(df)
    >>> print(results['success'])
    """
    print("Great Expectations 검증 시작...")
    
    # Context 생성 (임시 메모리 기반)
    context = gx.get_context()
    
    # Data Asset 추가
    data_source = context.sources.add_pandas(name="pandas_source")
    data_asset = data_source.add_dataframe_asset(name="my_dataframe_asset")
    batch_request = data_asset.build_batch_request(dataframe=df)
    
    if auto_create:
        print("  자동으로 Expectation 생성 중...")
        
        # Expectation Suite 생성
        suite = context.add_expectation_suite(expectation_suite_name=expectation_suite_name)
        validator = context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=expectation_suite_name
        )
        
        # 자동 Expectation 생성
        # 1. 컬럼 존재 확인
        for col in df.columns:
            validator.expect_column_to_exist(column=col)
        
        # 2. 데이터 타입 확인
        for col in df.columns:
            dtype = str(df[col].dtype)
            if dtype.startswith('int') or dtype.startswith('float'):
                # 수치형: 범위 확인
                col_min = df[col].min()
                col_max = df[col].max()
                validator.expect_column_values_to_be_between(
                    column=col,
                    min_value=col_min,
                    max_value=col_max,
                    mostly=0.95  # 95% 이상 조건 충족
                )
            elif dtype == 'object':
                # 문자형: 유니크 값 확인
                unique_values = df[col].nunique()
                if unique_values < 50:  # 카테고리로 간주
                    allowed_values = df[col].unique().tolist()
                    validator.expect_column_values_to_be_in_set(
                        column=col,
                        value_set=allowed_values
                    )
        
        # 3. 결측값 확인
        for col in df.columns:
            null_ratio = df[col].isnull().mean()
            if null_ratio < 0.01:  # 결측값이 1% 미만이면
                validator.expect_column_values_to_not_be_null(column=col)
        
        # Suite 저장
        validator.save_expectation_suite(discard_failed_expectations=False)
        print(f"  ✓ {len(validator.get_expectation_suite().expectations)} 개의 Expectation 생성")
    
    # 검증 실행
    print("  검증 실행 중...")
    checkpoint = context.add_checkpoint(
        name="my_checkpoint",
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": expectation_suite_name
            }
        ]
    )
    
    results = checkpoint.run()
    
    # 결과 요약
    validation_results = results.list_validation_results()[0]
    success = validation_results["success"]
    statistics = validation_results["statistics"]
    
    print(f"\n[검증 결과]")
    print(f"  전체 성공 여부: {'✓ 통과' if success else '✗ 실패'}")
    print(f"  평가된 Expectations: {statistics['evaluated_expectations']}")
    print(f"  성공한 Expectations: {statistics['successful_expectations']}")
    print(f"  실패한 Expectations: {statistics['unsuccessful_expectations']}")
    print(f"  성공률: {statistics['success_percent']:.2f}%")
    
    # 실패한 Expectation 상세 출력
    if not success:
        print(f"\n[실패 상세]")
        for result in validation_results["results"]:
            if not result["success"]:
                expectation_type = result["expectation_config"]["expectation_type"]
                kwargs = result["expectation_config"]["kwargs"]
                print(f"  ✗ {expectation_type}")
                print(f"    컬럼: {kwargs.get('column', 'N/A')}")
                if "observed_value" in result["result"]:
                    print(f"    관측값: {result['result']['observed_value']}")
    
    return {
        'success': success,
        'statistics': statistics,
        'results': validation_results
    }


# 사용 예시
validation_results = validate_with_great_expectations(
    df,
    expectation_suite_name="sales_data_quality",
    auto_create=True
)
```

#### 3.2.2 커스텀 Expectation 정의

```python
def create_custom_expectations(
    df: pd.DataFrame,
    rules: dict,
    suite_name: str = "custom_suite"
) -> dict:
    """
    사용자 정의 Expectation으로 데이터 검증
    
    Parameters:
    -----------
    df : pd.DataFrame
        검증할 데이터프레임
    rules : dict
        검증 규칙
        예: {
            'age': {'min': 0, 'max': 120, 'allow_null': False},
            'email': {'regex': r'^[\w\.-]+@[\w\.-]+\.\w+$'},
            'status': {'values': ['active', 'inactive']}
        }
    suite_name : str
        Suite 이름
    
    Returns:
    --------
    dict
        검증 결과
    
    Examples:
    ---------
    >>> rules = {
    ...     'age': {'min': 0, 'max': 120},
    ...     'email': {'regex': r'^[\w\.-]+@[\w\.-]+\.\w+$'}
    ... }
    >>> results = create_custom_expectations(df, rules)
    """
    print(f"커스텀 Expectation 생성 중...")
    
    context = gx.get_context()
    
    # Data Source 생성
    data_source = context.sources.add_pandas(name="pandas_source")
    data_asset = data_source.add_dataframe_asset(name="dataframe")
    batch_request = data_asset.build_batch_request(dataframe=df)
    
    # Suite 생성
    suite = context.add_expectation_suite(expectation_suite_name=suite_name)
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
    )
    
    # 규칙별 Expectation 생성
    for column, column_rules in rules.items():
        if column not in df.columns:
            print(f"  ⚠ 컬럼 '{column}'이 존재하지 않습니다. 건너뜁니다.")
            continue
        
        print(f"\n  [{column}]")
        
        # 최소값
        if 'min' in column_rules:
            validator.expect_column_min_to_be_between(
                column=column,
                min_value=column_rules['min'],
                max_value=df[column].max()
            )
            print(f"    ✓ 최소값 >= {column_rules['min']}")
        
        # 최대값
        if 'max' in column_rules:
            validator.expect_column_max_to_be_between(
                column=column,
                min_value=df[column].min(),
                max_value=column_rules['max']
            )
            print(f"    ✓ 최대값 <= {column_rules['max']}")
        
        # 허용값
        if 'values' in column_rules:
            validator.expect_column_values_to_be_in_set(
                column=column,
                value_set=column_rules['values']
            )
            print(f"    ✓ 허용값: {column_rules['values']}")
        
        # NULL 허용 여부
        if 'allow_null' in column_rules and not column_rules['allow_null']:
            validator.expect_column_values_to_not_be_null(column=column)
            print(f"    ✓ NULL 불허")
        
        # 정규식 패턴
        if 'regex' in column_rules:
            validator.expect_column_values_to_match_regex(
                column=column,
                regex=column_rules['regex']
            )
            print(f"    ✓ 정규식 패턴 적용")
        
        # 유니크 값
        if 'unique' in column_rules and column_rules['unique']:
            validator.expect_column_values_to_be_unique(column=column)
            print(f"    ✓ 고유값 조건")
    
    # Suite 저장
    validator.save_expectation_suite()
    
    # 검증 실행
    checkpoint = context.add_checkpoint(
        name="custom_checkpoint",
        validations=[{
            "batch_request": batch_request,
            "expectation_suite_name": suite_name
        }]
    )
    
    results = checkpoint.run()
    validation_results = results.list_validation_results()[0]
    
    print(f"\n[검증 완료]")
    print(f"  성공: {validation_results['success']}")
    print(f"  성공률: {validation_results['statistics']['success_percent']:.2f}%")
    
    return {
        'success': validation_results['success'],
        'statistics': validation_results['statistics'],
        'results': validation_results
    }


# 사용 예시
rules = {
    'age': {'min': 18, 'max': 100, 'allow_null': False},
    'email': {'regex': r'^[\w\.-]+@[\w\.-]+\.\w+$'},
    'status': {'values': ['active', 'inactive', 'pending']},
    'user_id': {'unique': True, 'allow_null': False},
    'salary': {'min': 0, 'max': 1000000}
}

results = create_custom_expectations(df, rules, suite_name="user_data_rules")
```

### 3.3 Sweetviz (타겟 변수 비교 분석)

```python
import sweetviz as sv

def generate_sweetviz_report(
    df: pd.DataFrame,
    target_column: str = None,
    compare_df: pd.DataFrame = None,
    output_file: str = "sweetviz_report.html"
) -> None:
    """
    Sweetviz를 사용한 타겟 변수 중심 분석 및 데이터셋 비교
    
    Parameters:
    -----------
    df : pd.DataFrame
        분석할 데이터프레임
    target_column : str, optional
        타겟 변수 컬럼명
    compare_df : pd.DataFrame, optional
        비교할 데이터프레임 (예: test set)
    output_file : str
        출력 HTML 파일 경로
    
    Examples:
    ---------
    >>> # 기본 분석
    >>> generate_sweetviz_report(df, output_file="report.html")
    >>> 
    >>> # 타겟 변수 중심 분석
    >>> generate_sweetviz_report(df, target_column='churn')
    >>> 
    >>> # 데이터셋 비교 (Train vs Test)
    >>> generate_sweetviz_report(train_df, compare_df=test_df)
    """
    print("Sweetviz 리포트 생성 중...")
    
    if compare_df is not None:
        # 데이터셋 비교
        print(f"  데이터셋 비교 모드")
        print(f"  - 기준 데이터: {df.shape}")
        print(f"  - 비교 데이터: {compare_df.shape}")
        
        report = sv.compare(
            source=df,
            compare=compare_df,
            target_feat=target_column,
            pairwise_analysis='auto'
        )
    else:
        # 단일 데이터셋 분석
        print(f"  단일 데이터셋 분석")
        if target_column:
            print(f"  - 타겟 변수: {target_column}")
        
        report = sv.analyze(
            source=df,
            target_feat=target_column,
            pairwise_analysis='auto'
        )
    
    # 리포트 저장
    report.show_html(filepath=output_file, open_browser=False)
    print(f"✓ Sweetviz 리포트 저장: {output_file}")


# 사용 예시 1: 기본 분석
generate_sweetviz_report(df, output_file="reports/sweetviz_basic.html")

# 사용 예시 2: 타겟 변수 중심 분석
generate_sweetviz_report(
    df,
    target_column='purchase',
    output_file="reports/sweetviz_target.html"
)

# 사용 예시 3: Train vs Test 비교
generate_sweetviz_report(
    train_df,
    compare_df=test_df,
    target_column='target',
    output_file="reports/sweetviz_train_vs_test.html"
)
```

### 3.4 통합 프로파일링 함수

```python
def comprehensive_profiling(
    df: pd.DataFrame,
    target_column: str = None,
    output_dir: str = "reports",
    tools: list = ['ydata', 'great_expectations', 'sweetviz']
) -> dict:
    """
    여러 프로파일링 도구를 통합 실행
    
    Parameters:
    -----------
    df : pd.DataFrame
        프로파일링할 데이터프레임
    target_column : str, optional
        타겟 변수 (있는 경우)
    output_dir : str, default "reports"
        출력 디렉토리
    tools : list
        사용할 도구 리스트
    
    Returns:
    --------
    dict
        각 도구별 결과 요약
    
    Examples:
    ---------
    >>> results = comprehensive_profiling(
    ...     df,
    ...     target_column='churn',
    ...     output_dir='analysis_reports'
    ... )
    """
    from pathlib import Path
    import time
    
    # 출력 디렉토리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    print("=" * 80)
    print("통합 프로파일링 시작")
    print("=" * 80)
    print(f"데이터 크기: {df.shape}")
    print(f"타겟 변수: {target_column if target_column else '없음'}")
    print(f"사용 도구: {', '.join(tools)}")
    print("=" * 80)
    
    # 1. YData Profiling
    if 'ydata' in tools:
        print("\n[1/3] YData Profiling 실행 중...")
        start_time = time.time()
        try:
            if target_column:
                profile = generate_target_focused_profile(
                    df,
                    target_column=target_column,
                    title="YData Profile",
                    output_file=f"{output_dir}/ydata_profile.html"
                )
            else:
                profile = generate_ydata_profile(
                    df,
                    title="YData Profile",
                    output_file=f"{output_dir}/ydata_profile.html"
                )
            
            elapsed = time.time() - start_time
            results['ydata'] = {
                'success': True,
                'output': f"{output_dir}/ydata_profile.html",
                'time': elapsed
            }
            print(f"✓ YData Profiling 완료 ({elapsed:.2f}초)")
        except Exception as e:
            print(f"✗ YData Profiling 실패: {e}")
            results['ydata'] = {'success': False, 'error': str(e)}
    
    # 2. Great Expectations
    if 'great_expectations' in tools:
        print("\n[2/3] Great Expectations 검증 중...")
        start_time = time.time()
        try:
            validation_results = validate_with_great_expectations(
                df,
                expectation_suite_name="auto_suite",
                auto_create=True
            )
            
            elapsed = time.time() - start_time
            results['great_expectations'] = {
                'success': validation_results['success'],
                'statistics': validation_results['statistics'],
                'time': elapsed
            }
            print(f"✓ Great Expectations 완료 ({elapsed:.2f}초)")
        except Exception as e:
            print(f"✗ Great Expectations 실패: {e}")
            results['great_expectations'] = {'success': False, 'error': str(e)}
    
    # 3. Sweetviz
    if 'sweetviz' in tools:
        print("\n[3/3] Sweetviz 리포트 생성 중...")
        start_time = time.time()
        try:
            generate_sweetviz_report(
                df,
                target_column=target_column,
                output_file=f"{output_dir}/sweetviz_report.html"
            )
            
            elapsed = time.time() - start_time
            results['sweetviz'] = {
                'success': True,
                'output': f"{output_dir}/sweetviz_report.html",
                'time': elapsed
            }
            print(f"✓ Sweetviz 완료 ({elapsed:.2f}초)")
        except Exception as e:
            print(f"✗ Sweetviz 실패: {e}")
            results['sweetviz'] = {'success': False, 'error': str(e)}
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("통합 프로파일링 완료")
    print("=" * 80)
    
    total_time = sum(r.get('time', 0) for r in results.values())
    print(f"총 소요 시간: {total_time:.2f}초")
    
    for tool, result in results.items():
        if result.get('success'):
            print(f"✓ {tool}: 성공 ({result.get('time', 0):.2f}초)")
            if 'output' in result:
                print(f"  출력: {result['output']}")
        else:
            print(f"✗ {tool}: 실패 - {result.get('error', '알 수 없는 에러')}")
    
    return results


# 사용 예시
results = comprehensive_profiling(
    df,
    target_column='churn',
    output_dir='comprehensive_reports',
    tools=['ydata', 'great_expectations', 'sweetviz']
)
```

---

## 4. 예시 (Examples)

### 4.1 전체 워크플로우 예시

```python
import pandas as pd
import numpy as np
from pathlib import Path

# 샘플 데이터 생성
np.random.seed(42)
n_samples = 10000

df = pd.DataFrame({
    'customer_id': range(1, n_samples + 1),
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.normal(50000, 20000, n_samples).clip(20000, 200000),
    'account_balance': np.random.lognormal(10, 1, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'num_products': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.5, 0.25, 0.15, 0.07, 0.03]),
    'is_active': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
    'tenure_months': np.random.randint(1, 120, n_samples),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
    'churn': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
})

# 결측값 추가 (실제 상황 시뮬레이션)
df.loc[np.random.choice(df.index, 200), 'income'] = np.nan
df.loc[np.random.choice(df.index, 100), 'credit_score'] = np.nan

print("=" * 80)
print("자동화된 프로파일링 데모")
print("=" * 80)
print(f"데이터 크기: {df.shape}")
print(f"타겟 변수: churn")
print("\n")

# 통합 프로파일링 실행
results = comprehensive_profiling(
    df,
    target_column='churn',
    output_dir='demo_reports',
    tools=['ydata', 'great_expectations', 'sweetviz']
)

# 결과 요약 출력
print("\n" + "=" * 80)
print("프로파일링 결과 요약")
print("=" * 80)

if results['ydata']['success']:
    print(f"\n[YData Profiling]")
    print(f"  ✓ 리포트: {results['ydata']['output']}")
    print(f"  ✓ 소요 시간: {results['ydata']['time']:.2f}초")

if results['great_expectations']['success']:
    print(f"\n[Great Expectations]")
    stats = results['great_expectations']['statistics']
    print(f"  ✓ 평가: {stats['evaluated_expectations']}개")
    print(f"  ✓ 성공: {stats['successful_expectations']}개")
    print(f"  ✓ 실패: {stats['unsuccessful_expectations']}개")
    print(f"  ✓ 성공률: {stats['success_percent']:.2f}%")

if results['sweetviz']['success']:
    print(f"\n[Sweetviz]")
    print(f"  ✓ 리포트: {results['sweetviz']['output']}")
    print(f"  ✓ 소요 시간: {results['sweetviz']['time']:.2f}초")

print("\n" + "=" * 80)
print("다음 단계 권장 사항:")
print("=" * 80)
print("1. HTML 리포트를 브라우저에서 열어 시각적 분석")
print("2. Great Expectations 실패 항목 검토 및 데이터 클렌징")
print("3. YData Profiling Alerts 섹션 확인")
print("4. 타겟 변수와 높은 상관관계 변수 확인")
print("5. 결측값 패턴 분석 및 처리 전략 수립")
```

### 4.2 출력 예시

```
================================================================================
통합 프로파일링 시작
================================================================================
데이터 크기: (10000, 10)
타겟 변수: churn
사용 도구: ydata, great_expectations, sweetviz
================================================================================

[1/3] YData Profiling 실행 중...
YData Profiling 시작...
  데이터 크기: (10000, 10)
✓ 프로파일 생성 완료: demo_reports/ydata_profile.html

[주요 통계]
  변수 개수: 10
  관측치 개수: 10,000
  결측값 셀: 300
  중복 행: 0
✓ YData Profiling 완료 (15.34초)

[2/3] Great Expectations 검증 중...
Great Expectations 검증 시작...
  자동으로 Expectation 생성 중...
  ✓ 23 개의 Expectation 생성
  검증 실행 중...

[검증 결과]
  전체 성공 여부: ✓ 통과
  평가된 Expectations: 23
  성공한 Expectations: 21
  실패한 Expectations: 2
  성공률: 91.30%

[실패 상세]
  ✗ expect_column_values_to_not_be_null
    컬럼: income
    관측값: 200 nulls
  ✗ expect_column_values_to_not_be_null
    컬럼: credit_score
    관측값: 100 nulls
✓ Great Expectations 완료 (3.21초)

[3/3] Sweetviz 리포트 생성 중...
Sweetviz 리포트 생성 중...
  단일 데이터셋 분석
  - 타겟 변수: churn
✓ Sweetviz 리포트 저장: demo_reports/sweetviz_report.html
✓ Sweetviz 완료 (8.47초)

================================================================================
통합 프로파일링 완료
================================================================================
총 소요 시간: 27.02초
✓ ydata: 성공 (15.34초)
  출력: demo_reports/ydata_profile.html
✓ great_expectations: 성공 (3.21초)
✓ sweetviz: 성공 (8.47초)
  출력: demo_reports/sweetviz_report.html

================================================================================
프로파일링 결과 요약
================================================================================

[YData Profiling]
  ✓ 리포트: demo_reports/ydata_profile.html
  ✓ 소요 시간: 15.34초

[Great Expectations]
  ✓ 평가: 23개
  ✓ 성공: 21개
  ✓ 실패: 2개
  ✓ 성공률: 91.30%

[Sweetviz]
  ✓ 리포트: demo_reports/sweetviz_report.html
  ✓ 소요 시간: 8.47초

================================================================================
다음 단계 권장 사항:
================================================================================
1. HTML 리포트를 브라우저에서 열어 시각적 분석
2. Great Expectations 실패 항목 검토 및 데이터 클렌징
3. YData Profiling Alerts 섹션 확인
4. 타겟 변수와 높은 상관관계 변수 확인
5. 결측값 패턴 분석 및 처리 전략 수립
```

---

## 5. 에이전트 매핑 (Agent Mapping)

### 5.1 담당 에이전트

**Primary Agent**: `data-scientist`
- 역할: 자동화된 프로파일링 실행 및 결과 해석
- 책임: 도구 선택, 설정 최적화, 결과 분석

**Supporting Agents**: 없음

### 5.2 관련 스킬

| 스킬 | 용도 | 우선순위 |
|------|------|----------|
| ydata-profiling | 포괄적 자동 프로파일링 | 필수 |
| great-expectations | 데이터 품질 검증 | 필수 |
| sweetviz | 타겟 변수 분석 및 비교 | 권장 |
| pandas | 데이터 처리 | 필수 |

### 5.3 자동화 커맨드 예시

```bash
# 기본 프로파일링
/eda:profile --file data.csv --output report.html

# 타겟 변수 중심 프로파일링
/eda:profile --file data.csv --target churn --output churn_analysis.html

# 데이터셋 비교
/eda:compare --train train.csv --test test.csv --output comparison.html

# 데이터 품질 검증
/eda:validate --file data.csv --expectations rules.json
```

---

## 6. 필요 라이브러리 (Required Libraries)

### 6.1 핵심 라이브러리 설치

```bash
# YData Profiling
uv pip install ydata-profiling==4.6.4

# Great Expectations
uv pip install great-expectations==0.18.8

# Sweetviz
uv pip install sweetviz==2.3.1

# 기본 라이브러리
uv pip install pandas==2.2.0 numpy==1.26.3

# 시각화 라이브러리 (의존성)
uv pip install matplotlib==3.8.2 seaborn==0.13.1
```

### 6.2 라이브러리 버전 정보

| 라이브러리 | 버전 | 용도 | 필수 여부 |
|-----------|------|------|-----------|
| ydata-profiling | 4.6.4 | 자동 프로파일링 | 필수 |
| great-expectations | 0.18.8 | 데이터 품질 검증 | 필수 |
| sweetviz | 2.3.1 | 타겟 분석 및 비교 | 권장 |
| pandas | 2.2.0 | 데이터 처리 | 필수 |
| numpy | 1.26.3 | 수치 연산 | 필수 |
| matplotlib | 3.8.2 | 시각화 | 필수 |
| seaborn | 0.13.1 | 통계 시각화 | 필수 |

### 6.3 시스템 요구사항

- **Python**: 3.9 이상
- **RAM**: 최소 8GB (대용량 데이터는 16GB 이상 권장)
- **Storage**: 리포트 저장용 충분한 공간

---

## 7. 체크포인트 (Checkpoints)

### 7.1 검증 항목

- [ ] **프로파일 생성**
  - [ ] HTML 리포트가 정상적으로 생성되었는가?
  - [ ] 브라우저에서 리포트가 올바르게 표시되는가?
  - [ ] 모든 섹션이 포함되어 있는가?

- [ ] **데이터 품질**
  - [ ] Alerts 섹션의 경고를 확인했는가?
  - [ ] 높은 상관관계 변수 쌍을 확인했는가?
  - [ ] 결측값 패턴을 분석했는가?

- [ ] **타겟 변수 (해당시)**
  - [ ] 타겟 분포의 불균형을 확인했는가?
  - [ ] 타겟과 피처 간 관계를 파악했는가?
  - [ ] 유의미한 변수를 식별했는가?

- [ ] **액션 아이템**
  - [ ] 데이터 클렌징 우선순위를 정했는가?
  - [ ] Feature engineering 아이디어를 도출했는가?
  - [ ] 다음 분석 단계를 계획했는가?

### 7.2 품질 기준

**프로파일링 성공 기준**:
- 모든 변수에 대한 통계 및 시각화 생성
- 리포트 생성 시간 < 5분 (중규모 데이터)
- 메모리 오류 없이 완료

**데이터 품질 기준**:
- Alerts 개수 < 10개 (또는 모두 해명 가능)
- 결측값 비율 < 20%
- 높은 상관관계 (|r| > 0.9) 변수 쌍 < 5개

---

## 8. 트러블슈팅 (Troubleshooting)

### 8.1 YData Profiling 오류

#### 오류 1: 메모리 부족

**증상**: `MemoryError` 또는 프로세스 중단

**해결 방법**:
```python
# 최소 모드 사용
profile = ProfileReport(df, minimal=True)

# 또는 일부 분석 비활성화
profile = ProfileReport(
    df,
    correlations=None,
    interactions=None,
    missing_diagrams={'heatmap': False}
)

# 샘플링
df_sample = df.sample(n=10000, random_state=42)
profile = ProfileReport(df_sample)
```

#### 오류 2: 생성 시간 과다

**증상**: 리포트 생성에 30분 이상 소요

**해결 방법**:
```python
# Minimal 모드
profile = ProfileReport(df, minimal=True)

# 상관관계 계산 간소화
config = {
    'correlations': {
        'pearson': {'calculate': True},
        'spearman': {'calculate': False},
        'kendall': {'calculate': False},
        'phi_k': {'calculate': False}
    }
}
profile = ProfileReport(df, **config)
```

### 8.2 Great Expectations 오류

#### 오류 1: Expectation 생성 실패

**증상**: `ValueError` 또는 검증 실패

**해결 방법**:
```python
# 안전 모드로 Expectation 생성
try:
    validator.expect_column_values_to_be_between(
        column=col,
        min_value=col_min,
        max_value=col_max,
        mostly=0.95  # 95%만 충족해도 통과
    )
except Exception as e:
    print(f"Expectation 생성 실패 ({col}): {e}")
```

---

## 9. 참고 자료 (References)

### 9.1 공식 문서

1. **YData Profiling**
   - URL: https://docs.profiling.ydata.ai/
   - GitHub: https://github.com/ydataai/ydata-profiling

2. **Great Expectations**
   - URL: https://docs.greatexpectations.io/
   - GitHub: https://github.com/great-expectations/great_expectations

3. **Sweetviz**
   - GitHub: https://github.com/fbdesignpro/sweetviz

---

**문서 끝**

다음 단계: [04-univariate-analysis.md](./04-univariate-analysis.md) - 단변량 분석
