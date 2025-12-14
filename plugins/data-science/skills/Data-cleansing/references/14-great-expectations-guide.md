# 14. Great Expectations Guide (Great Expectations 가이드)

**생성일**: 2025-01-26  
**버전**: 1.0  
**카테고리**: Automated Data Validation

---

## 1. 개요 (Overview)

### 1.1 목적 (Purpose)

Great Expectations는 데이터 품질을 자동으로 검증하고 문서화하는 오픈소스 Python 라이브러리입니다. 이 가이드는 Great Expectations를 사용하여 데이터 클렌징 워크플로우에 자동화된 검증을 통합하는 방법을 제공합니다.

### 1.2 Great Expectations란?

**핵심 개념**:
- **Expectations**: 데이터에 대한 검증 가능한 주장(assertions)
- **Expectation Suites**: Expectations의 모음
- **Validations**: Expectation Suite를 데이터에 적용
- **Data Docs**: 자동 생성되는 검증 결과 문서
- **Checkpoints**: 검증 실행 및 액션 자동화

### 1.3 적용 시기 (When to Apply)

**필수 적용**:
- ✅ 프로덕션 데이터 파이프라인
- ✅ 자동화된 데이터 품질 모니터링
- ✅ 규제 준수 프로젝트

**권장 적용**:
- 🔹 반복적인 데이터 검증 작업
- 🔹 팀 간 데이터 품질 기준 공유
- 🔹 CI/CD 파이프라인 통합

---

## 2. 이론적 배경 (Theoretical Background)

### 2.1 Great Expectations 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                   Data Context                          │
│  (GX 프로젝트의 중심, 설정 및 메타데이터 관리)            │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼───────┐    ┌───────▼────────┐
│ Data Sources  │    │ Expectation    │
│               │    │ Suites         │
│ - Pandas      │    │                │
│ - SQL         │    │ - Completeness │
│ - Spark       │    │ - Validity     │
│ - Files       │    │ - Consistency  │
└───────┬───────┘    └───────┬────────┘
        │                    │
        └──────────┬─────────┘
                   │
            ┌──────▼──────┐
            │ Validations │
            │             │
            │ - Run       │
            │ - Results   │
            │ - Actions   │
            └──────┬──────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼─────┐      ┌───────▼──────┐
│ Data Docs   │      │ Checkpoints  │
│             │      │              │
│ - HTML      │      │ - Scheduling │
│ - Reports   │      │ - Alerts     │
└─────────────┘      └──────────────┘
```

### 2.2 Expectation 유형

**1. Column Map Expectations** (각 값 검증)
- `expect_column_values_to_not_be_null`
- `expect_column_values_to_be_in_set`
- `expect_column_values_to_match_regex`

**2. Column Aggregate Expectations** (집계 검증)
- `expect_column_mean_to_be_between`
- `expect_column_unique_value_count_to_be_between`
- `expect_column_proportion_of_unique_values_to_be_between`

**3. Table Expectations** (테이블 레벨 검증)
- `expect_table_row_count_to_be_between`
- `expect_table_column_count_to_equal`
- `expect_table_columns_to_match_ordered_list`

**4. Multi-Column Expectations** (다중 컬럼 검증)
- `expect_column_pair_values_A_to_be_greater_than_B`
- `expect_multicolumn_sum_to_equal`
- `expect_compound_columns_to_be_unique`

---

## 3. 구현 (Implementation)

### 3.1 설치 및 초기 설정

```python
# 설치
# pip install great-expectations

import great_expectations as gx
from great_expectations.core.batch import BatchRequest
from great_expectations.checkpoint import Checkpoint
import pandas as pd

def setup_great_expectations(project_dir: str = "./gx"):
    """
    Great Expectations 프로젝트 초기화
    Initialize Great Expectations project
    
    Parameters:
    -----------
    project_dir : str
        프로젝트 디렉토리 경로
        
    Returns:
    --------
    context : DataContext
        Great Expectations 컨텍스트
    """
    # 새 프로젝트 초기화 (처음 실행 시)
    # gx.data_context.DataContext.create(project_dir)
    
    # 기존 프로젝트 로드
    context = gx.get_context(context_root_dir=project_dir)
    
    print(f"✅ Great Expectations initialized at {project_dir}")
    print(f"   Data Docs: {context.get_docs_sites_urls()}")
    
    return context


def add_pandas_datasource(
    context,
    datasource_name: str = "pandas_datasource"
):
    """
    Pandas 데이터 소스 추가
    Add Pandas data source
    """
    # Pandas 데이터 소스 설정
    datasource = context.sources.add_or_update_pandas(datasource_name)
    
    print(f"✅ Added Pandas datasource: {datasource_name}")
    
    return datasource


def add_dataframe_asset(
    datasource,
    asset_name: str,
    df: pd.DataFrame
):
    """
    DataFrame을 Data Asset으로 추가
    Add DataFrame as a data asset
    """
    # DataFrame Asset 추가
    data_asset = datasource.add_dataframe_asset(name=asset_name)
    
    # Batch Request 생성
    batch_request = data_asset.build_batch_request(dataframe=df)
    
    print(f"✅ Added DataFrame asset: {asset_name}")
    
    return data_asset, batch_request
```

### 3.2 Expectation Suite 생성

```python
class ExpectationSuiteBuilder:
    """
    Expectation Suite 빌더 클래스
    Build and manage Expectation Suites
    """
    
    def __init__(self, context, suite_name: str):
        """
        Parameters:
        -----------
        context : DataContext
            Great Expectations 컨텍스트
        suite_name : str
            Expectation Suite 이름
        """
        self.context = context
        self.suite_name = suite_name
        
        # Suite 생성 또는 로드
        self.context.add_or_update_expectation_suite(suite_name)
        
    def create_basic_data_quality_suite(
        self,
        batch_request,
        column_config: dict
    ):
        """
        기본 데이터 품질 Expectation Suite 생성
        Create basic data quality expectation suite
        
        Parameters:
        -----------
        batch_request : BatchRequest
            배치 요청 객체
        column_config : dict
            컬럼별 설정
            {
                'column_name': {
                    'dtype': 'int64',
                    'nullable': False,
                    'unique': False,
                    'min': 0,
                    'max': 100,
                    'values': ['A', 'B', 'C']
                }
            }
        """
        # Validator 생성
        validator = self.context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=self.suite_name
        )
        
        print(f"\n📋 Building Expectation Suite: {self.suite_name}")
        
        # 1. 테이블 레벨 Expectations
        print("\n1. Adding table-level expectations...")
        
        # 컬럼 존재 확인
        validator.expect_table_columns_to_match_set(
            column_set=list(column_config.keys())
        )
        
        # 행 개수 확인 (최소 1개 이상)
        validator.expect_table_row_count_to_be_between(min_value=1)
        
        # 2. 컬럼 레벨 Expectations
        print("2. Adding column-level expectations...")
        
        for col, config in column_config.items():
            print(f"   - {col}")
            
            # 컬럼 존재
            validator.expect_column_to_exist(col)
            
            # Null 값 체크
            if not config.get('nullable', True):
                validator.expect_column_values_to_not_be_null(col)
            
            # 유니크 체크
            if config.get('unique', False):
                validator.expect_column_values_to_be_unique(col)
            
            # 데이터 타입
            if 'dtype' in config:
                validator.expect_column_values_to_be_of_type(
                    col,
                    type_=config['dtype']
                )
            
            # 범위 체크 (수치형)
            if 'min' in config or 'max' in config:
                validator.expect_column_values_to_be_between(
                    col,
                    min_value=config.get('min'),
                    max_value=config.get('max')
                )
            
            # 값 집합 체크 (범주형)
            if 'values' in config:
                validator.expect_column_values_to_be_in_set(
                    col,
                    value_set=config['values']
                )
        
        # Suite 저장
        validator.save_expectation_suite(discard_failed_expectations=False)
        
        print(f"\n✅ Expectation Suite created: {self.suite_name}")
        print(f"   Total expectations: {len(validator.get_expectation_suite().expectations)}")
        
        return validator
    
    def create_ecommerce_suite(self, batch_request):
        """
        E-commerce 데이터용 Expectation Suite
        E-commerce specific expectation suite
        """
        validator = self.context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=self.suite_name
        )
        
        print(f"\n📦 Building E-commerce Expectation Suite")
        
        # Order ID: 유니크, Not Null, 양수
        validator.expect_column_to_exist('order_id')
        validator.expect_column_values_to_not_be_null('order_id')
        validator.expect_column_values_to_be_unique('order_id')
        validator.expect_column_values_to_be_of_type('order_id', 'int64')
        validator.expect_column_values_to_be_between('order_id', min_value=1)
        
        # Customer ID: Not Null, 양수
        validator.expect_column_to_exist('customer_id')
        validator.expect_column_values_to_not_be_null('customer_id')
        validator.expect_column_values_to_be_between('customer_id', min_value=1)
        
        # Quantity: 양수, 합리적 범위
        validator.expect_column_to_exist('quantity')
        validator.expect_column_values_to_not_be_null('quantity')
        validator.expect_column_values_to_be_between(
            'quantity',
            min_value=1,
            max_value=1000
        )
        
        # Unit Price: 양수
        validator.expect_column_to_exist('unit_price')
        validator.expect_column_values_to_not_be_null('unit_price')
        validator.expect_column_values_to_be_between(
            'unit_price',
            min_value=0.01
        )
        
        # Total Amount: 양수, quantity * unit_price와 일치
        validator.expect_column_to_exist('total_amount')
        validator.expect_column_values_to_not_be_null('total_amount')
        validator.expect_column_values_to_be_between(
            'total_amount',
            min_value=0.01
        )
        
        # Status: 특정 값만 허용
        validator.expect_column_to_exist('status')
        validator.expect_column_values_to_be_in_set(
            'status',
            value_set=['pending', 'processing', 'shipped', 'delivered', 'cancelled']
        )
        
        # Order Date: datetime 타입
        validator.expect_column_to_exist('order_date')
        validator.expect_column_values_to_be_of_type(
            'order_date',
            type_='datetime64[ns]'
        )
        
        # Multi-column: total_amount 대략 quantity * unit_price
        # (Great Expectations에서는 직접 지원하지 않으므로 커스텀 필요)
        
        # 저장
        validator.save_expectation_suite(discard_failed_expectations=False)
        
        print(f"✅ E-commerce Suite created")
        
        return validator
    
    def create_healthcare_suite(self, batch_request):
        """
        의료 데이터용 Expectation Suite
        Healthcare specific expectation suite
        """
        validator = self.context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=self.suite_name
        )
        
        print(f"\n🏥 Building Healthcare Expectation Suite")
        
        # Patient ID: 유니크
        validator.expect_column_values_to_be_unique('patient_id')
        
        # Age: 0-120 범위
        validator.expect_column_values_to_be_between(
            'age',
            min_value=0,
            max_value=120
        )
        
        # Blood Pressure: 정상 범위
        validator.expect_column_values_to_be_between(
            'bp_systolic',
            min_value=70,
            max_value=200
        )
        validator.expect_column_values_to_be_between(
            'bp_diastolic',
            min_value=40,
            max_value=130
        )
        
        # Multi-column: bp_systolic > bp_diastolic
        validator.expect_column_pair_values_A_to_be_greater_than_B(
            column_A='bp_systolic',
            column_B='bp_diastolic'
        )
        
        # Temperature: 35-42°C
        validator.expect_column_values_to_be_between(
            'temperature',
            min_value=35.0,
            max_value=42.0
        )
        
        # Dates: admission_date <= discharge_date
        if 'admission_date' in validator.active_batch.data.columns:
            validator.expect_column_pair_values_A_to_be_greater_than_B(
                column_A='discharge_date',
                column_B='admission_date',
                or_equal=True
            )
        
        # 저장
        validator.save_expectation_suite(discard_failed_expectations=False)
        
        print(f"✅ Healthcare Suite created")
        
        return validator


# 사용 예시
def create_expectation_suite_example():
    """
    Expectation Suite 생성 예시
    """
    # 컨텍스트 초기화
    context = setup_great_expectations()
    
    # 샘플 데이터
    df = pd.DataFrame({
        'order_id': range(1, 101),
        'customer_id': [i % 20 + 1 for i in range(100)],
        'quantity': [i % 10 + 1 for i in range(100)],
        'unit_price': [10.0 + i * 5 for i in range(100)],
        'status': ['pending'] * 100
    })
    
    # 데이터 소스 추가
    datasource = add_pandas_datasource(context)
    data_asset, batch_request = add_dataframe_asset(
        datasource,
        asset_name='sample_orders',
        df=df
    )
    
    # Expectation Suite 빌더
    builder = ExpectationSuiteBuilder(context, suite_name='orders_quality_suite')
    
    # 컬럼 설정
    column_config = {
        'order_id': {
            'dtype': 'int64',
            'nullable': False,
            'unique': True,
            'min': 1
        },
        'customer_id': {
            'dtype': 'int64',
            'nullable': False,
            'min': 1
        },
        'quantity': {
            'dtype': 'int64',
            'nullable': False,
            'min': 1,
            'max': 1000
        },
        'unit_price': {
            'dtype': 'float64',
            'nullable': False,
            'min': 0.01
        },
        'status': {
            'dtype': 'object',
            'nullable': False,
            'values': ['pending', 'processing', 'shipped', 'delivered', 'cancelled']
        }
    }
    
    # Suite 생성
    validator = builder.create_basic_data_quality_suite(
        batch_request=batch_request,
        column_config=column_config
    )
    
    return context, validator
```

### 3.3 Validation 실행

```python
class ValidationRunner:
    """
    Validation 실행 및 결과 처리 클래스
    Run and process validations
    """
    
    def __init__(self, context):
        self.context = context
    
    def run_validation(
        self,
        batch_request,
        expectation_suite_name: str,
        run_name: str = None
    ):
        """
        Validation 실행
        Run validation
        
        Parameters:
        -----------
        batch_request : BatchRequest
            배치 요청
        expectation_suite_name : str
            Expectation Suite 이름
        run_name : str, optional
            실행 이름 (타임스탬프 등)
            
        Returns:
        --------
        results : ValidationResults
            검증 결과
        """
        from datetime import datetime
        
        if run_name is None:
            run_name = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n🔍 Running validation: {run_name}")
        
        # Validator 생성
        validator = self.context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=expectation_suite_name
        )
        
        # Validation 실행
        results = validator.validate()
        
        # 결과 요약
        self.print_validation_results(results)
        
        return results
    
    def print_validation_results(self, results):
        """
        Validation 결과 출력
        Print validation results
        """
        print("\n" + "="*80)
        print("VALIDATION RESULTS")
        print("="*80)
        
        statistics = results.statistics
        
        print(f"\n📊 Summary:")
        print(f"   Total Expectations: {statistics['evaluated_expectations']}")
        print(f"   Successful: {statistics['successful_expectations']}")
        print(f"   Failed: {statistics['unsuccessful_expectations']}")
        print(f"   Success Rate: {statistics['success_percent']:.2f}%")
        
        # 실패한 Expectations
        if statistics['unsuccessful_expectations'] > 0:
            print(f"\n❌ Failed Expectations:")
            
            for result in results.results:
                if not result.success:
                    expectation_type = result.expectation_config.expectation_type
                    kwargs = result.expectation_config.kwargs
                    
                    print(f"\n   - {expectation_type}")
                    if 'column' in kwargs:
                        print(f"     Column: {kwargs['column']}")
                    
                    # 실패 상세
                    if hasattr(result, 'result') and result.result:
                        if 'unexpected_count' in result.result:
                            print(f"     Unexpected values: {result.result['unexpected_count']}")
                        if 'unexpected_percent' in result.result:
                            print(f"     Unexpected percent: {result.result['unexpected_percent']:.2f}%")
        
        else:
            print(f"\n✅ All expectations passed!")
    
    def create_checkpoint(
        self,
        checkpoint_name: str,
        datasource_name: str,
        data_asset_name: str,
        expectation_suite_name: str
    ):
        """
        Checkpoint 생성
        Create a checkpoint
        
        Checkpoints는 validation을 자동화하고 결과에 따른 액션을 수행
        """
        checkpoint_config = {
            "name": checkpoint_name,
            "config_version": 1.0,
            "class_name": "Checkpoint",
            "validations": [
                {
                    "batch_request": {
                        "datasource_name": datasource_name,
                        "data_asset_name": data_asset_name
                    },
                    "expectation_suite_name": expectation_suite_name
                }
            ],
            "action_list": [
                {
                    "name": "store_validation_result",
                    "action": {
                        "class_name": "StoreValidationResultAction"
                    }
                },
                {
                    "name": "update_data_docs",
                    "action": {
                        "class_name": "UpdateDataDocsAction"
                    }
                }
            ]
        }
        
        checkpoint = self.context.add_or_update_checkpoint(**checkpoint_config)
        
        print(f"✅ Checkpoint created: {checkpoint_name}")
        
        return checkpoint
    
    def run_checkpoint(self, checkpoint_name: str):
        """
        Checkpoint 실행
        Run a checkpoint
        """
        print(f"\n🚀 Running checkpoint: {checkpoint_name}")
        
        checkpoint = self.context.get_checkpoint(checkpoint_name)
        results = checkpoint.run()
        
        # 결과 처리
        if results.success:
            print("✅ Checkpoint validation passed!")
        else:
            print("❌ Checkpoint validation failed!")
        
        return results


# 사용 예시
def run_validation_example():
    """
    Validation 실행 예시
    """
    # 설정
    context = setup_great_expectations()
    
    # 샘플 데이터 (품질 이슈 포함)
    df = pd.DataFrame({
        'order_id': [1, 2, 3, 4, 5, 5],  # 중복
        'customer_id': [1, 2, None, 4, 5, 6],  # Null
        'quantity': [1, 2, 3, -1, 5, 6],  # 음수
        'unit_price': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        'status': ['pending', 'processing', 'invalid', 'pending', 'shipped', 'delivered']  # 잘못된 값
    })
    
    # 데이터 소스
    datasource = add_pandas_datasource(context)
    data_asset, batch_request = add_dataframe_asset(
        datasource,
        asset_name='orders_with_issues',
        df=df
    )
    
    # Expectation Suite 생성
    builder = ExpectationSuiteBuilder(context, suite_name='test_suite')
    column_config = {
        'order_id': {'nullable': False, 'unique': True},
        'customer_id': {'nullable': False},
        'quantity': {'min': 1},
        'status': {'values': ['pending', 'processing', 'shipped', 'delivered', 'cancelled']}
    }
    validator = builder.create_basic_data_quality_suite(batch_request, column_config)
    
    # Validation 실행
    runner = ValidationRunner(context)
    results = runner.run_validation(
        batch_request=batch_request,
        expectation_suite_name='test_suite'
    )
    
    return results
```

### 3.4 Data Docs 생성

```python
def generate_data_docs(context):
    """
    Data Docs 생성 및 열기
    Generate and open Data Docs
    """
    print("\n📄 Building Data Docs...")
    
    # Data Docs 빌드
    context.build_data_docs()
    
    # Data Docs URL
    docs_sites = context.get_docs_sites_urls()
    
    print("✅ Data Docs built successfully!")
    print("\n📂 Data Docs locations:")
    for site in docs_sites:
        print(f"   - {site['site_url']}")
    
    # 브라우저에서 열기 (선택사항)
    # import webbrowser
    # if docs_sites:
    #     webbrowser.open(docs_sites[0]['site_url'])
    
    return docs_sites
```

---

## 4. 예시 (Examples)

### 4.1 완전한 워크플로우 예시

```python
def complete_great_expectations_workflow():
    """
    Great Expectations 전체 워크플로우
    """
    print("="*80)
    print("GREAT EXPECTATIONS COMPLETE WORKFLOW")
    print("="*80)
    
    # 1. 초기 설정
    print("\n1. Setting up Great Expectations...")
    context = setup_great_expectations(project_dir="./gx_demo")
    
    # 2. 샘플 데이터
    print("\n2. Creating sample data...")
    df = pd.DataFrame({
        'order_id': range(1, 1001),
        'customer_id': [i % 100 + 1 for i in range(1000)],
        'quantity': [i % 10 + 1 for i in range(1000)],
        'unit_price': [10.0 + (i % 50) * 5 for i in range(1000)],
        'total_amount': [0.0] * 1000,
        'status': ['pending'] * 800 + ['processing'] * 150 + ['shipped'] * 50,
        'order_date': pd.date_range('2024-01-01', periods=1000, freq='H')
    })
    df['total_amount'] = (df['quantity'] * df['unit_price']).round(2)
    
    print(f"   Created {len(df)} orders")
    
    # 3. 데이터 소스 설정
    print("\n3. Setting up data source...")
    datasource = add_pandas_datasource(context, datasource_name="orders_datasource")
    data_asset, batch_request = add_dataframe_asset(
        datasource,
        asset_name="orders_data",
        df=df
    )
    
    # 4. Expectation Suite 생성
    print("\n4. Creating Expectation Suite...")
    builder = ExpectationSuiteBuilder(context, suite_name="orders_quality_suite")
    validator = builder.create_ecommerce_suite(batch_request)
    
    # 5. Validation 실행
    print("\n5. Running validation...")
    runner = ValidationRunner(context)
    results = runner.run_validation(
        batch_request=batch_request,
        expectation_suite_name="orders_quality_suite",
        run_name="initial_validation"
    )
    
    # 6. Checkpoint 생성
    print("\n6. Creating checkpoint...")
    checkpoint = runner.create_checkpoint(
        checkpoint_name="orders_checkpoint",
        datasource_name="orders_datasource",
        data_asset_name="orders_data",
        expectation_suite_name="orders_quality_suite"
    )
    
    # 7. Data Docs 생성
    print("\n7. Generating Data Docs...")
    docs_sites = generate_data_docs(context)
    
    print("\n" + "="*80)
    print("WORKFLOW COMPLETED")
    print("="*80)
    print(f"\n✅ All steps completed successfully!")
    print(f"📊 Validation success rate: {results.statistics['success_percent']:.2f}%")
    print(f"📄 Data Docs: {docs_sites[0]['site_url'] if docs_sites else 'N/A'}")
    
    return context, results


# 실행
if __name__ == "__main__":
    context, results = complete_great_expectations_workflow()
```

---

## 5. 에이전트 매핑 (Agent Mapping)

### 5.1 Primary Agent

**`data-cleaning-specialist`**
- Expectation Suite 정의
- Validation 실행
- 결과 분석

### 5.2 Supporting Agents

**`data-scientist`**
- 통계적 Expectations 설계
- 임계값 결정

**`technical-documentation-writer`**
- Data Docs 커스터마이징
- 검증 리포트 작성

---

## 6. 필요 라이브러리 (Required Libraries)

```bash
# 필수
pip install great-expectations>=0.18.0

# 선택 (데이터 소스별)
pip install sqlalchemy>=2.0.0  # SQL 데이터베이스
pip install pyspark>=3.4.0  # Spark
```

---

## 7. 체크포인트 (Checkpoints)

### 7.1 설정 체크리스트

- [ ] Great Expectations 초기화 완료
- [ ] 데이터 소스 설정 완료
- [ ] Expectation Suite 정의 완료
- [ ] Checkpoint 생성 완료

### 7.2 운영 체크리스트

- [ ] Validation 정기 실행
- [ ] 실패 시 알람 설정
- [ ] Data Docs 업데이트
- [ ] 성능 모니터링

---

## 8. 트러블슈팅 (Troubleshooting)

**문제: Context 초기화 실패**
```python
# 해결
import great_expectations as gx
context = gx.get_context(mode="file")
```

**문제: Validation 속도 느림**
```python
# 해결: 샘플링 사용
df_sample = df.sample(n=10000, random_state=42)
```

---

## 9. 참고 자료 (References)

- 공식 문서: https://docs.greatexpectations.io/
- GitHub: https://github.com/great-expectations/great_expectations
- 커뮤니티: https://greatexpectations.io/slack

---

**작성자**: Claude Code  
**최종 수정일**: 2025-01-26  
**버전**: 1.0
