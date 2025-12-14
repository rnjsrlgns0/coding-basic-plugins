# Reference 01: Data Loading and Validation

**Version**: 1.0  
**Last Updated**: 2025-01-25  
**Workflow Phase**: Phase 1 - Data Collection & Initial Validation  
**Estimated Reading Time**: 20-25 minutes

---

## 1. 개요 (Overview)

### 1.1 목적 (Purpose)

데이터 로딩 및 초기 검증(Data Loading and Validation)은 모든 EDA 프로세스의 첫 단계로, 데이터를 올바르게 메모리에 적재하고 기본적인 무결성과 품질을 검증하는 과정입니다. 이 단계에서 발견되지 않은 문제는 후속 분석에서 잘못된 결론을 도출하거나 시스템 오류를 유발할 수 있습니다.

**주요 목적**:
- 다양한 형식의 데이터를 안전하게 로드
- 데이터 구조와 타입의 정확성 검증
- 메모리 효율성 확보
- 데이터 무결성 확인
- 초기 품질 이슈 식별

### 1.2 적용 시기 (When to Apply)

이 프로세스는 다음과 같은 상황에서 적용됩니다:

1. **신규 데이터셋 분석 시작**: 처음 받은 데이터를 분석하기 전
2. **데이터 업데이트 후**: 기존 데이터가 갱신되었을 때
3. **데이터 통합**: 여러 소스의 데이터를 결합하기 전
4. **프로덕션 파이프라인**: 자동화된 데이터 처리 시스템에서
5. **데이터 품질 모니터링**: 정기적인 데이터 검증 필요 시

### 1.3 예상 소요 시간 (Expected Duration)

- **소규모 데이터 (< 1GB)**: 5-10분
- **중규모 데이터 (1-10GB)**: 15-30분
- **대규모 데이터 (> 10GB)**: 30분 이상 (청크 단위 처리 필요)

---

## 2. 이론적 배경 (Theoretical Background)

### 2.1 데이터 로딩의 중요성

데이터 로딩은 단순히 파일을 여는 것 이상의 의미를 가집니다:

1. **데이터 타입 추론**: Pandas는 자동으로 데이터 타입을 추론하지만, 항상 정확하지는 않습니다.
2. **인코딩 처리**: 다양한 문자 인코딩(UTF-8, Latin-1, CP949 등)을 올바르게 처리해야 합니다.
3. **메모리 최적화**: 불필요한 메모리 사용을 줄여 성능을 향상시킵니다.
4. **구조적 검증**: 예상한 스키마와 실제 데이터 구조가 일치하는지 확인합니다.

### 2.2 데이터 검증의 원칙

**GIGO (Garbage In, Garbage Out)**: 잘못된 데이터는 잘못된 결과를 생성합니다. 초기 검증은 이를 방지하는 첫 번째 방어선입니다.

**검증 레벨**:
1. **구조적 검증**: 행/열 개수, 컬럼명, 데이터 타입
2. **내용 검증**: 값의 범위, 패턴, 일관성
3. **논리적 검증**: 비즈니스 규칙 준수 여부
4. **통계적 검증**: 분포, 이상치, 결측값 패턴

### 2.3 일반적인 시나리오

**시나리오 1: CSV 파일 로딩**
- 가장 흔한 데이터 형식
- 구분자, 인코딩, 헤더 처리 필요
- 대용량 파일은 청크 단위 처리

**시나리오 2: 데이터베이스 연결**
- SQL 쿼리를 통한 데이터 추출
- 연결 풀 관리 및 리소스 해제
- 데이터 타입 매핑 주의

**시나리오 3: API 데이터 수집**
- JSON/XML 형식 처리
- 페이지네이션 고려
- Rate limiting 대응

**시나리오 4: 다중 파일 병합**
- 일관된 스키마 확인
- 중복 제거
- 파일 간 정합성 검증

---

## 3. 구현 (Implementation)

### 3.1 기본 데이터 로딩

#### 3.1.1 CSV 파일 로딩

```python
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

def load_csv_safe(
    filepath: str,
    encoding: str = 'utf-8',
    delimiter: str = ',',
    dtype: dict = None,
    parse_dates: list = None,
    low_memory: bool = False
) -> pd.DataFrame:
    """
    안전하게 CSV 파일을 로드하는 함수
    
    Parameters:
    -----------
    filepath : str
        CSV 파일 경로
    encoding : str, default 'utf-8'
        파일 인코딩 (utf-8, latin-1, cp949 등)
    delimiter : str, default ','
        구분자 (쉼표, 탭, 파이프 등)
    dtype : dict, optional
        컬럼별 데이터 타입 지정
    parse_dates : list, optional
        날짜로 파싱할 컬럼 리스트
    low_memory : bool, default False
        메모리 사용량 최적화 (대용량 파일에 사용)
    
    Returns:
    --------
    pd.DataFrame
        로드된 데이터프레임
    
    Examples:
    ---------
    >>> df = load_csv_safe('data.csv', encoding='utf-8')
    >>> df = load_csv_safe('data.csv', parse_dates=['date_column'])
    """
    try:
        # 파일 존재 확인
        if not Path(filepath).exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")
        
        # CSV 로드
        df = pd.read_csv(
            filepath,
            encoding=encoding,
            delimiter=delimiter,
            dtype=dtype,
            parse_dates=parse_dates,
            low_memory=low_memory
        )
        
        print(f"✓ 파일 로드 성공: {filepath}")
        print(f"  - 행 개수: {len(df):,}")
        print(f"  - 열 개수: {len(df.columns):,}")
        
        return df
        
    except UnicodeDecodeError:
        # 인코딩 에러 발생 시 다른 인코딩 시도
        print(f"⚠ {encoding} 인코딩 실패, 다른 인코딩 시도 중...")
        for alt_encoding in ['latin-1', 'cp949', 'euc-kr', 'iso-8859-1']:
            try:
                df = pd.read_csv(filepath, encoding=alt_encoding, delimiter=delimiter)
                print(f"✓ {alt_encoding} 인코딩으로 로드 성공")
                return df
            except:
                continue
        raise ValueError("지원되는 인코딩으로 파일을 열 수 없습니다.")
    
    except pd.errors.ParserError as e:
        print(f"✗ CSV 파싱 에러: {e}")
        # 에러가 발생한 행 건너뛰기
        df = pd.read_csv(filepath, encoding=encoding, on_bad_lines='skip')
        print(f"⚠ 문제가 있는 행을 건너뛰고 로드했습니다.")
        return df
    
    except Exception as e:
        print(f"✗ 예상치 못한 에러: {e}")
        raise


# 사용 예시
df = load_csv_safe('data/sales_data.csv', encoding='utf-8')
```

#### 3.1.2 대용량 CSV 파일 청크 로딩

```python
def load_large_csv_chunks(
    filepath: str,
    chunksize: int = 10000,
    sample_size: int = None,
    encoding: str = 'utf-8'
) -> pd.DataFrame:
    """
    대용량 CSV 파일을 청크 단위로 로드
    
    Parameters:
    -----------
    filepath : str
        CSV 파일 경로
    chunksize : int, default 10000
        한 번에 읽을 행 개수
    sample_size : int, optional
        샘플링할 총 행 개수 (None이면 전체 로드)
    encoding : str, default 'utf-8'
        파일 인코딩
    
    Returns:
    --------
    pd.DataFrame
        로드된 데이터프레임 (또는 샘플)
    
    Examples:
    ---------
    >>> # 전체 파일을 청크로 처리
    >>> df = load_large_csv_chunks('huge_file.csv', chunksize=50000)
    >>> 
    >>> # 100,000개 행만 샘플링
    >>> df_sample = load_large_csv_chunks('huge_file.csv', sample_size=100000)
    """
    chunks = []
    total_rows = 0
    
    print(f"청크 단위로 파일 로딩 중... (청크 크기: {chunksize:,})")
    
    try:
        for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunksize, encoding=encoding)):
            chunks.append(chunk)
            total_rows += len(chunk)
            
            print(f"  청크 {i+1} 로드: {len(chunk):,} 행 (누적: {total_rows:,})", end='\r')
            
            # 샘플 크기에 도달하면 중단
            if sample_size and total_rows >= sample_size:
                print(f"\n✓ 샘플 크기 도달: {total_rows:,} 행")
                break
        
        # 모든 청크 결합
        df = pd.concat(chunks, ignore_index=True)
        
        if sample_size:
            df = df.iloc[:sample_size]
        
        print(f"\n✓ 총 {len(df):,} 행 로드 완료")
        return df
        
    except Exception as e:
        print(f"\n✗ 청크 로딩 에러: {e}")
        raise


# 사용 예시
# 10GB 파일에서 100만 행만 샘플링
df_sample = load_large_csv_chunks('very_large_file.csv', sample_size=1000000)
```

#### 3.1.3 다양한 형식 로딩

```python
def load_data_universal(
    filepath: str,
    file_type: str = None,
    **kwargs
) -> pd.DataFrame:
    """
    다양한 파일 형식을 자동으로 감지하여 로드
    
    Parameters:
    -----------
    filepath : str
        파일 경로
    file_type : str, optional
        파일 형식 ('csv', 'excel', 'json', 'parquet', 'sql')
        None이면 확장자로 자동 감지
    **kwargs
        각 로더에 전달할 추가 인자
    
    Returns:
    --------
    pd.DataFrame
        로드된 데이터프레임
    
    Examples:
    ---------
    >>> df = load_data_universal('data.csv')
    >>> df = load_data_universal('data.xlsx', sheet_name='Sheet1')
    >>> df = load_data_universal('data.json')
    """
    filepath = Path(filepath)
    
    # 파일 형식 자동 감지
    if file_type is None:
        file_type = filepath.suffix.lower().lstrip('.')
    
    print(f"파일 형식: {file_type.upper()}")
    
    # 형식별 로더
    loaders = {
        'csv': lambda: pd.read_csv(filepath, **kwargs),
        'txt': lambda: pd.read_csv(filepath, **kwargs),
        'tsv': lambda: pd.read_csv(filepath, delimiter='\t', **kwargs),
        'xlsx': lambda: pd.read_excel(filepath, **kwargs),
        'xls': lambda: pd.read_excel(filepath, **kwargs),
        'json': lambda: pd.read_json(filepath, **kwargs),
        'parquet': lambda: pd.read_parquet(filepath, **kwargs),
        'feather': lambda: pd.read_feather(filepath, **kwargs),
        'pkl': lambda: pd.read_pickle(filepath, **kwargs),
        'pickle': lambda: pd.read_pickle(filepath, **kwargs),
        'hdf': lambda: pd.read_hdf(filepath, **kwargs),
        'h5': lambda: pd.read_hdf(filepath, **kwargs),
    }
    
    if file_type not in loaders:
        raise ValueError(f"지원하지 않는 파일 형식: {file_type}")
    
    try:
        df = loaders[file_type]()
        print(f"✓ {file_type.upper()} 파일 로드 성공: {df.shape}")
        return df
    except Exception as e:
        print(f"✗ {file_type.upper()} 로드 실패: {e}")
        raise


# 사용 예시
df_csv = load_data_universal('data.csv')
df_excel = load_data_universal('data.xlsx', sheet_name='Sales')
df_json = load_data_universal('data.json', orient='records')
df_parquet = load_data_universal('data.parquet')
```

#### 3.1.4 데이터베이스에서 로딩

```python
import sqlalchemy as sa
from sqlalchemy import create_engine

def load_from_database(
    query: str,
    connection_string: str = None,
    engine: sa.engine.Engine = None,
    chunksize: int = None
) -> pd.DataFrame:
    """
    데이터베이스에서 SQL 쿼리로 데이터 로드
    
    Parameters:
    -----------
    query : str
        SQL 쿼리문
    connection_string : str, optional
        데이터베이스 연결 문자열
        예: 'postgresql://user:password@localhost:5432/database'
    engine : sqlalchemy.engine.Engine, optional
        SQLAlchemy 엔진 객체 (connection_string 대신 사용)
    chunksize : int, optional
        청크 크기 (대용량 쿼리 결과)
    
    Returns:
    --------
    pd.DataFrame
        쿼리 결과 데이터프레임
    
    Examples:
    ---------
    >>> query = "SELECT * FROM sales WHERE date >= '2024-01-01'"
    >>> df = load_from_database(query, connection_string=conn_str)
    """
    try:
        # 엔진 생성
        if engine is None:
            if connection_string is None:
                raise ValueError("connection_string 또는 engine 중 하나는 필수입니다.")
            engine = create_engine(connection_string)
        
        print(f"데이터베이스 쿼리 실행 중...")
        
        # 쿼리 실행
        if chunksize:
            # 청크 단위로 읽기
            chunks = []
            for chunk in pd.read_sql_query(query, engine, chunksize=chunksize):
                chunks.append(chunk)
                print(f"  청크 로드: {len(chunk):,} 행", end='\r')
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_sql_query(query, engine)
        
        print(f"\n✓ 쿼리 완료: {df.shape} 데이터 로드")
        return df
        
    except Exception as e:
        print(f"✗ 데이터베이스 에러: {e}")
        raise
    finally:
        # 연결 종료
        if engine:
            engine.dispose()


# 사용 예시
conn_str = "postgresql://user:pass@localhost:5432/mydb"
query = """
    SELECT 
        customer_id,
        order_date,
        total_amount
    FROM orders
    WHERE order_date >= '2024-01-01'
"""
df = load_from_database(query, connection_string=conn_str)
```

### 3.2 초기 데이터 검증

#### 3.2.1 기본 정보 확인

```python
def inspect_dataframe(df: pd.DataFrame, sample_size: int = 5) -> dict:
    """
    데이터프레임의 기본 정보를 종합적으로 검사
    
    Parameters:
    -----------
    df : pd.DataFrame
        검사할 데이터프레임
    sample_size : int, default 5
        출력할 샘플 행 개수
    
    Returns:
    --------
    dict
        검사 결과 딕셔너리
    
    Examples:
    ---------
    >>> info = inspect_dataframe(df)
    >>> print(info['shape'])
    """
    print("=" * 80)
    print("데이터프레임 기본 정보")
    print("=" * 80)
    
    # 1. 크기 정보
    n_rows, n_cols = df.shape
    print(f"\n[크기]")
    print(f"  행 개수: {n_rows:,}")
    print(f"  열 개수: {n_cols:,}")
    print(f"  총 셀 개수: {n_rows * n_cols:,}")
    
    # 2. 메모리 사용량
    memory_usage = df.memory_usage(deep=True).sum()
    print(f"\n[메모리 사용량]")
    print(f"  총 메모리: {memory_usage / 1024**2:.2f} MB")
    print(f"  행당 평균: {memory_usage / n_rows:.2f} bytes")
    
    # 3. 데이터 타입 분포
    dtype_counts = df.dtypes.value_counts()
    print(f"\n[데이터 타입 분포]")
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} 개 컬럼")
    
    # 4. 컬럼 목록
    print(f"\n[컬럼 목록]")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:3d}. {col} ({df[col].dtype})")
    
    # 5. 결측값 요약
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    print(f"\n[결측값]")
    if len(missing_cols) == 0:
        print("  결측값 없음 ✓")
    else:
        print(f"  결측값이 있는 컬럼: {len(missing_cols)}개")
        for col, count in missing_cols.items():
            pct = 100 * count / n_rows
            print(f"    {col}: {count:,} ({pct:.2f}%)")
    
    # 6. 중복 행
    n_duplicates = df.duplicated().sum()
    print(f"\n[중복 행]")
    print(f"  중복 행 개수: {n_duplicates:,} ({100*n_duplicates/n_rows:.2f}%)")
    
    # 7. 샘플 데이터
    print(f"\n[상위 {sample_size}개 행]")
    print(df.head(sample_size).to_string())
    
    print(f"\n[하위 {sample_size}개 행]")
    print(df.tail(sample_size).to_string())
    
    # 8. 기술 통계 (수치형 컬럼만)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n[수치형 컬럼 기술 통계]")
        print(df[numeric_cols].describe().to_string())
    
    print("\n" + "=" * 80)
    
    # 결과 딕셔너리 반환
    return {
        'shape': (n_rows, n_cols),
        'memory_mb': memory_usage / 1024**2,
        'dtypes': dtype_counts.to_dict(),
        'missing_values': missing_cols.to_dict(),
        'duplicates': n_duplicates,
        'columns': df.columns.tolist()
    }


# 사용 예시
info = inspect_dataframe(df, sample_size=5)
```

#### 3.2.2 데이터 타입 검증

```python
def validate_data_types(
    df: pd.DataFrame,
    expected_schema: dict,
    auto_convert: bool = False
) -> tuple[bool, list]:
    """
    데이터프레임의 데이터 타입을 예상 스키마와 비교 검증
    
    Parameters:
    -----------
    df : pd.DataFrame
        검증할 데이터프레임
    expected_schema : dict
        예상 스키마 {'column_name': 'expected_dtype'}
        예: {'age': 'int64', 'name': 'object', 'date': 'datetime64[ns]'}
    auto_convert : bool, default False
        True면 자동으로 타입 변환 시도
    
    Returns:
    --------
    tuple[bool, list]
        (검증 성공 여부, 불일치 리스트)
    
    Examples:
    ---------
    >>> schema = {'age': 'int64', 'salary': 'float64', 'name': 'object'}
    >>> is_valid, mismatches = validate_data_types(df, schema)
    """
    mismatches = []
    
    print("데이터 타입 검증 중...")
    print("-" * 60)
    
    for col, expected_dtype in expected_schema.items():
        # 컬럼 존재 확인
        if col not in df.columns:
            mismatches.append({
                'column': col,
                'issue': 'missing',
                'expected': expected_dtype,
                'actual': None
            })
            print(f"✗ {col}: 컬럼이 존재하지 않음")
            continue
        
        actual_dtype = str(df[col].dtype)
        
        # 타입 일치 확인
        if actual_dtype != expected_dtype:
            mismatches.append({
                'column': col,
                'issue': 'type_mismatch',
                'expected': expected_dtype,
                'actual': actual_dtype
            })
            print(f"✗ {col}: {actual_dtype} → 예상: {expected_dtype}")
            
            # 자동 변환 시도
            if auto_convert:
                try:
                    if expected_dtype.startswith('int'):
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype(expected_dtype)
                    elif expected_dtype.startswith('float'):
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    elif expected_dtype == 'datetime64[ns]':
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    elif expected_dtype == 'object':
                        df[col] = df[col].astype(str)
                    
                    print(f"  → 자동 변환 성공: {expected_dtype}")
                except Exception as e:
                    print(f"  → 자동 변환 실패: {e}")
        else:
            print(f"✓ {col}: {actual_dtype}")
    
    print("-" * 60)
    
    is_valid = len(mismatches) == 0
    if is_valid:
        print("✓ 모든 데이터 타입이 예상과 일치합니다.")
    else:
        print(f"✗ {len(mismatches)}개 컬럼에서 불일치 발견")
    
    return is_valid, mismatches


# 사용 예시
schema = {
    'customer_id': 'int64',
    'order_date': 'datetime64[ns]',
    'total_amount': 'float64',
    'product_name': 'object',
    'quantity': 'int64'
}

is_valid, issues = validate_data_types(df, schema, auto_convert=True)
```

#### 3.2.3 값 범위 검증

```python
def validate_value_ranges(
    df: pd.DataFrame,
    validation_rules: dict
) -> tuple[bool, list]:
    """
    컬럼별 값의 범위와 제약 조건 검증
    
    Parameters:
    -----------
    df : pd.DataFrame
        검증할 데이터프레임
    validation_rules : dict
        검증 규칙
        예: {
            'age': {'min': 0, 'max': 120},
            'category': {'allowed_values': ['A', 'B', 'C']},
            'email': {'pattern': r'^[\w\.-]+@[\w\.-]+\.\w+$'}
        }
    
    Returns:
    --------
    tuple[bool, list]
        (검증 성공 여부, 위반 사항 리스트)
    
    Examples:
    ---------
    >>> rules = {
    ...     'age': {'min': 0, 'max': 120},
    ...     'price': {'min': 0},
    ...     'status': {'allowed_values': ['active', 'inactive']}
    ... }
    >>> is_valid, violations = validate_value_ranges(df, rules)
    """
    violations = []
    
    print("값 범위 검증 중...")
    print("-" * 60)
    
    for col, rules in validation_rules.items():
        if col not in df.columns:
            print(f"⚠ {col}: 컬럼이 존재하지 않음 (건너뜀)")
            continue
        
        print(f"\n[{col}]")
        
        # 최소값 검증
        if 'min' in rules:
            min_val = rules['min']
            violations_count = (df[col] < min_val).sum()
            if violations_count > 0:
                violations.append({
                    'column': col,
                    'rule': 'min',
                    'threshold': min_val,
                    'violations': violations_count
                })
                print(f"  ✗ 최소값 위반: {violations_count:,}개 ({100*violations_count/len(df):.2f}%)")
                print(f"    예시: {df[df[col] < min_val][col].head(3).tolist()}")
            else:
                print(f"  ✓ 최소값 ({min_val}) 조건 충족")
        
        # 최대값 검증
        if 'max' in rules:
            max_val = rules['max']
            violations_count = (df[col] > max_val).sum()
            if violations_count > 0:
                violations.append({
                    'column': col,
                    'rule': 'max',
                    'threshold': max_val,
                    'violations': violations_count
                })
                print(f"  ✗ 최대값 위반: {violations_count:,}개 ({100*violations_count/len(df):.2f}%)")
                print(f"    예시: {df[df[col] > max_val][col].head(3).tolist()}")
            else:
                print(f"  ✓ 최대값 ({max_val}) 조건 충족")
        
        # 허용값 검증
        if 'allowed_values' in rules:
            allowed = set(rules['allowed_values'])
            invalid_mask = ~df[col].isin(allowed)
            violations_count = invalid_mask.sum()
            if violations_count > 0:
                violations.append({
                    'column': col,
                    'rule': 'allowed_values',
                    'allowed': list(allowed),
                    'violations': violations_count
                })
                invalid_values = df[invalid_mask][col].unique()[:5]
                print(f"  ✗ 허용값 위반: {violations_count:,}개")
                print(f"    허용값: {allowed}")
                print(f"    위반 예시: {list(invalid_values)}")
            else:
                print(f"  ✓ 모든 값이 허용 범위 내")
        
        # 정규식 패턴 검증
        if 'pattern' in rules:
            import re
            pattern = rules['pattern']
            # 문자열로 변환 후 검증
            str_col = df[col].astype(str)
            invalid_mask = ~str_col.str.match(pattern, na=False)
            violations_count = invalid_mask.sum()
            if violations_count > 0:
                violations.append({
                    'column': col,
                    'rule': 'pattern',
                    'pattern': pattern,
                    'violations': violations_count
                })
                print(f"  ✗ 패턴 위반: {violations_count:,}개")
                print(f"    패턴: {pattern}")
                print(f"    위반 예시: {df[invalid_mask][col].head(3).tolist()}")
            else:
                print(f"  ✓ 모든 값이 패턴에 일치")
        
        # NULL 검증
        if 'allow_null' in rules and not rules['allow_null']:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                violations.append({
                    'column': col,
                    'rule': 'allow_null',
                    'violations': null_count
                })
                print(f"  ✗ NULL 값 존재: {null_count:,}개 ({100*null_count/len(df):.2f}%)")
            else:
                print(f"  ✓ NULL 값 없음")
    
    print("\n" + "-" * 60)
    
    is_valid = len(violations) == 0
    if is_valid:
        print("✓ 모든 값 범위 검증 통과")
    else:
        print(f"✗ {len(violations)}개 규칙 위반 발견")
    
    return is_valid, violations


# 사용 예시
rules = {
    'age': {'min': 0, 'max': 120, 'allow_null': False},
    'price': {'min': 0},
    'quantity': {'min': 1, 'max': 10000},
    'status': {'allowed_values': ['active', 'inactive', 'pending']},
    'email': {'pattern': r'^[\w\.-]+@[\w\.-]+\.\w+$'},
    'phone': {'pattern': r'^\d{3}-\d{4}-\d{4}$'}
}

is_valid, violations = validate_value_ranges(df, rules)
```

### 3.3 메모리 최적화

#### 3.3.1 데이터 타입 최적화

```python
def optimize_dtypes(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    데이터프레임의 메모리 사용량을 최적화
    
    Parameters:
    -----------
    df : pd.DataFrame
        최적화할 데이터프레임
    verbose : bool, default True
        상세 출력 여부
    
    Returns:
    --------
    pd.DataFrame
        최적화된 데이터프레임
    
    Examples:
    ---------
    >>> df_optimized = optimize_dtypes(df)
    >>> # 메모리 50-70% 절감 가능
    """
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        print("데이터 타입 최적화 시작...")
        print(f"초기 메모리: {initial_memory:.2f} MB")
        print("-" * 60)
    
    # 1. 정수형 최적화
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        
        # 범위에 따라 최적 타입 선택
        if col_min >= 0:  # unsigned
            if col_max < 255:
                df[col] = df[col].astype('uint8')
                if verbose: print(f"  {col}: int64 → uint8")
            elif col_max < 65535:
                df[col] = df[col].astype('uint16')
                if verbose: print(f"  {col}: int64 → uint16")
            elif col_max < 4294967295:
                df[col] = df[col].astype('uint32')
                if verbose: print(f"  {col}: int64 → uint32")
        else:  # signed
            if col_min > -128 and col_max < 127:
                df[col] = df[col].astype('int8')
                if verbose: print(f"  {col}: int64 → int8")
            elif col_min > -32768 and col_max < 32767:
                df[col] = df[col].astype('int16')
                if verbose: print(f"  {col}: int64 → int16")
            elif col_min > -2147483648 and col_max < 2147483647:
                df[col] = df[col].astype('int32')
                if verbose: print(f"  {col}: int64 → int32")
    
    # 2. 실수형 최적화
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        # float32로 변환 (정밀도 손실 최소화)
        df[col] = df[col].astype('float32')
        if verbose: print(f"  {col}: float64 → float32")
    
    # 3. 범주형 변환 (카디널리티 < 50%)
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        
        # 고유값이 전체의 50% 미만이면 category로 변환
        if num_unique / num_total < 0.5:
            df[col] = df[col].astype('category')
            if verbose: print(f"  {col}: object → category ({num_unique} unique values)")
    
    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    reduction = 100 * (initial_memory - final_memory) / initial_memory
    
    if verbose:
        print("-" * 60)
        print(f"최종 메모리: {final_memory:.2f} MB")
        print(f"메모리 절감: {initial_memory - final_memory:.2f} MB ({reduction:.1f}%)")
    
    return df


# 사용 예시
df_optimized = optimize_dtypes(df, verbose=True)
```

---

## 4. 예시 (Examples)

### 4.1 전체 워크플로우 예시

```python
import pandas as pd
import numpy as np
from pathlib import Path

# ========== 1. 데이터 로딩 ==========
print("=" * 80)
print("STEP 1: 데이터 로딩")
print("=" * 80)

# CSV 파일 로딩
df = load_csv_safe(
    'data/sales_2024.csv',
    encoding='utf-8',
    parse_dates=['order_date', 'ship_date']
)

# ========== 2. 기본 검사 ==========
print("\n" + "=" * 80)
print("STEP 2: 기본 정보 확인")
print("=" * 80)

info = inspect_dataframe(df, sample_size=5)

# ========== 3. 스키마 검증 ==========
print("\n" + "=" * 80)
print("STEP 3: 데이터 타입 검증")
print("=" * 80)

expected_schema = {
    'order_id': 'int64',
    'customer_id': 'int64',
    'order_date': 'datetime64[ns]',
    'ship_date': 'datetime64[ns]',
    'product_name': 'object',
    'quantity': 'int64',
    'price': 'float64',
    'total_amount': 'float64',
    'status': 'object',
    'region': 'object'
}

is_valid, mismatches = validate_data_types(df, expected_schema, auto_convert=True)

# ========== 4. 값 범위 검증 ==========
print("\n" + "=" * 80)
print("STEP 4: 값 범위 검증")
print("=" * 80)

validation_rules = {
    'quantity': {'min': 1, 'max': 10000},
    'price': {'min': 0, 'max': 1000000},
    'total_amount': {'min': 0},
    'status': {'allowed_values': ['pending', 'shipped', 'delivered', 'cancelled']},
    'region': {'allowed_values': ['North', 'South', 'East', 'West']}
}

is_valid, violations = validate_value_ranges(df, validation_rules)

# ========== 5. 메모리 최적화 ==========
print("\n" + "=" * 80)
print("STEP 5: 메모리 최적화")
print("=" * 80)

df_optimized = optimize_dtypes(df, verbose=True)

# ========== 6. 결과 저장 ==========
print("\n" + "=" * 80)
print("STEP 6: 검증 완료 데이터 저장")
print("=" * 80)

# 최적화된 데이터를 Parquet 형식으로 저장 (압축 효율적)
output_path = 'data/sales_2024_validated.parquet'
df_optimized.to_parquet(output_path, compression='snappy', index=False)
print(f"✓ 검증 완료 데이터 저장: {output_path}")

# 검증 리포트 저장
validation_report = {
    'file': 'sales_2024.csv',
    'loaded_at': pd.Timestamp.now().isoformat(),
    'shape': info['shape'],
    'memory_mb': info['memory_mb'],
    'schema_valid': is_valid,
    'violations': violations
}

import json
with open('data/validation_report.json', 'w') as f:
    json.dump(validation_report, f, indent=2, default=str)
print(f"✓ 검증 리포트 저장: data/validation_report.json")
```

### 4.2 출력 예시

```
================================================================================
STEP 1: 데이터 로딩
================================================================================
✓ 파일 로드 성공: data/sales_2024.csv
  - 행 개수: 125,438
  - 열 개수: 10

================================================================================
STEP 2: 기본 정보 확인
================================================================================
데이터프레임 기본 정보
================================================================================

[크기]
  행 개수: 125,438
  열 개수: 10
  총 셀 개수: 1,254,380

[메모리 사용량]
  총 메모리: 18.45 MB
  행당 평균: 154.2 bytes

[데이터 타입 분포]
  int64: 3 개 컬럼
  float64: 2 개 컬럼
  object: 3 개 컬럼
  datetime64[ns]: 2 개 컬럼

[컬럼 목록]
    1. order_id (int64)
    2. customer_id (int64)
    3. order_date (datetime64[ns])
    4. ship_date (datetime64[ns])
    5. product_name (object)
    6. quantity (int64)
    7. price (float64)
    8. total_amount (float64)
    9. status (object)
   10. region (object)

[결측값]
  결측값이 있는 컬럼: 1개
    ship_date: 1,523 (1.21%)

[중복 행]
  중복 행 개수: 0 (0.00%)

================================================================================
STEP 5: 메모리 최적화
================================================================================
데이터 타입 최적화 시작...
초기 메모리: 18.45 MB
------------------------------------------------------------
  order_id: int64 → uint32
  customer_id: int64 → uint32
  quantity: int64 → uint16
  price: float64 → float32
  total_amount: float64 → float32
  status: object → category (4 unique values)
  region: object → category (4 unique values)
------------------------------------------------------------
최종 메모리: 6.82 MB
메모리 절감: 11.63 MB (63.0%)
```

---

## 5. 에이전트 매핑 (Agent Mapping)

### 5.1 담당 에이전트

**Primary Agent**: `data-scientist`
- 역할: 데이터 로딩, 초기 검증, 메모리 최적화 총괄
- 책임: 데이터 품질 보증, 로딩 전략 수립

**Supporting Agents**: 없음 (독립적 수행 가능)

### 5.2 관련 스킬

| 스킬 | 용도 | 우선순위 |
|------|------|----------|
| pandas | 데이터 로딩 및 조작 | 필수 |
| numpy | 수치 연산 및 배열 처리 | 필수 |
| sqlalchemy | 데이터베이스 연결 | 선택 |
| pathlib | 파일 경로 처리 | 권장 |

### 5.3 자동화 커맨드 예시

```bash
# 기본 로딩 및 검증
/eda:load --file data.csv --validate-schema schema.json

# 대용량 파일 샘플링
/eda:load --file huge_file.csv --sample 1000000 --optimize-memory

# 데이터베이스에서 로딩
/eda:load --db postgresql://localhost/mydb --query "SELECT * FROM sales"

# 다중 파일 병합
/eda:load --pattern "data/sales_*.csv" --merge --validate
```

---

## 6. 필요 라이브러리 (Required Libraries)

### 6.1 핵심 라이브러리

```bash
# 필수 라이브러리 설치
uv pip install pandas==2.2.0 numpy==1.26.3

# 데이터베이스 연결 (선택)
uv pip install sqlalchemy==2.0.25 psycopg2-binary==2.9.9

# Excel 파일 지원 (선택)
uv pip install openpyxl==3.1.2 xlrd==2.0.1

# Parquet 파일 지원 (권장)
uv pip install pyarrow==15.0.0

# Feather 파일 지원 (선택)
uv pip install pyarrow==15.0.0
```

### 6.2 라이브러리 버전 정보

| 라이브러리 | 버전 | 용도 | 필수 여부 |
|-----------|------|------|-----------|
| pandas | 2.2.0 | 데이터 조작 | 필수 |
| numpy | 1.26.3 | 수치 연산 | 필수 |
| sqlalchemy | 2.0.25 | DB 연결 | 선택 |
| psycopg2-binary | 2.9.9 | PostgreSQL | 선택 |
| pymysql | 1.1.0 | MySQL | 선택 |
| openpyxl | 3.1.2 | Excel (xlsx) | 선택 |
| xlrd | 2.0.1 | Excel (xls) | 선택 |
| pyarrow | 15.0.0 | Parquet/Feather | 권장 |

### 6.3 최소 시스템 요구사항

- **Python**: 3.9 이상 (권장: 3.11+)
- **RAM**: 데이터 크기의 3-5배 (최소 8GB 권장)
- **Storage**: 데이터 크기의 2배 (임시 파일 포함)

---

## 7. 체크포인트 (Checkpoints)

### 7.1 검증 항목

데이터 로딩 및 검증이 완료되면 다음 항목을 확인하세요:

- [ ] **데이터 로딩**
  - [ ] 파일이 오류 없이 로드되었는가?
  - [ ] 예상한 행/열 개수와 일치하는가?
  - [ ] 특수 문자 및 인코딩 문제가 없는가?

- [ ] **데이터 타입**
  - [ ] 모든 컬럼의 데이터 타입이 올바른가?
  - [ ] 숫자형 컬럼이 문자형으로 로드되지 않았는가?
  - [ ] 날짜형 컬럼이 올바르게 파싱되었는가?

- [ ] **데이터 품질**
  - [ ] 결측값 패턴을 확인했는가?
  - [ ] 중복 행이 있는가? (의도된 것인가?)
  - [ ] 이상치나 잘못된 값이 있는가?

- [ ] **메모리 효율성**
  - [ ] 메모리 사용량이 적절한가?
  - [ ] 필요시 데이터 타입을 최적화했는가?
  - [ ] 대용량 데이터는 청크로 처리했는가?

### 7.2 품질 기준

**통과 기준**:
- 데이터 로딩 성공률 > 99%
- 스키마 일치도 = 100%
- 결측값 비율 < 30% (또는 예상 범위 내)
- 메모리 최적화로 30% 이상 절감

**경고 기준**:
- 결측값 비율 10-30%
- 중복 행 비율 > 1%
- 값 범위 위반 < 5%

**실패 기준**:
- 스키마 불일치 > 20%
- 결측값 비율 > 50%
- 치명적 데이터 품질 이슈 발견

### 7.3 진행 결정

| 상황 | 조치 |
|------|------|
| 모든 검증 통과 | → Phase 2 (자동 프로파일링)으로 진행 |
| 경미한 이슈 발견 | → 이슈 기록 후 진행 (Phase 3에서 처리) |
| 심각한 이슈 발견 | → 데이터 제공자에게 피드백 요청 |
| 로딩 실패 | → 파일 형식/인코딩 재확인 |

---

## 8. 트러블슈팅 (Troubleshooting)

### 8.1 일반적인 오류 및 해결 방법

#### 오류 1: UnicodeDecodeError

**증상**:
```python
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0
```

**원인**: 파일 인코딩이 UTF-8이 아님

**해결 방법**:
```python
# 여러 인코딩 시도
for encoding in ['utf-8', 'latin-1', 'cp949', 'euc-kr', 'iso-8859-1']:
    try:
        df = pd.read_csv('file.csv', encoding=encoding)
        print(f"성공: {encoding}")
        break
    except UnicodeDecodeError:
        continue

# 또는 chardet로 자동 감지
import chardet
with open('file.csv', 'rb') as f:
    result = chardet.detect(f.read(100000))
    encoding = result['encoding']
df = pd.read_csv('file.csv', encoding=encoding)
```

#### 오류 2: ParserError (CSV 파싱 실패)

**증상**:
```python
pandas.errors.ParserError: Error tokenizing data
```

**원인**: 불규칙한 구분자, 줄바꿈 문자 포함, 인용 부호 문제

**해결 방법**:
```python
# 문제 행 건너뛰기
df = pd.read_csv('file.csv', on_bad_lines='skip')

# 또는 error_bad_lines=False (구버전)
df = pd.read_csv('file.csv', error_bad_lines=False, warn_bad_lines=True)

# 구분자 명시
df = pd.read_csv('file.csv', delimiter='|', quotechar='"')

# 엔진 변경
df = pd.read_csv('file.csv', engine='python')
```

#### 오류 3: MemoryError

**증상**:
```python
MemoryError: Unable to allocate array
```

**원인**: 파일이 메모리보다 큼

**해결 방법**:
```python
# 방법 1: 청크로 읽기
chunks = pd.read_csv('large_file.csv', chunksize=100000)
df = pd.concat([chunk for chunk in chunks], ignore_index=True)

# 방법 2: 필요한 컬럼만 로드
df = pd.read_csv('large_file.csv', usecols=['col1', 'col2', 'col3'])

# 방법 3: 데이터 타입 명시로 메모리 절감
dtypes = {
    'id': 'uint32',
    'value': 'float32',
    'category': 'category'
}
df = pd.read_csv('large_file.csv', dtype=dtypes)

# 방법 4: Dask 사용 (Out-of-core 처리)
import dask.dataframe as dd
df = dd.read_csv('large_file.csv').compute()
```

#### 오류 4: 날짜 파싱 실패

**증상**: 날짜 컬럼이 object 타입으로 로드됨

**해결 방법**:
```python
# 로드 시 날짜 파싱
df = pd.read_csv('file.csv', parse_dates=['date_column'])

# 로드 후 변환
df['date_column'] = pd.to_datetime(df['date_column'], errors='coerce')

# 특정 형식 지정
df['date_column'] = pd.to_datetime(df['date_column'], format='%Y-%m-%d')

# 다양한 형식 자동 추론
df['date_column'] = pd.to_datetime(df['date_column'], infer_datetime_format=True)
```

#### 오류 5: Mixed Types 경고

**증상**:
```python
DtypeWarning: Columns have mixed types
```

**원인**: 같은 컬럼에 여러 데이터 타입 혼재

**해결 방법**:
```python
# 타입 명시
df = pd.read_csv('file.csv', dtype={'column': str}, low_memory=False)

# 또는 변환기 사용
converters = {
    'column': lambda x: str(x).strip()
}
df = pd.read_csv('file.csv', converters=converters)
```

### 8.2 성능 최적화 팁

1. **대용량 파일 처리**:
   ```python
   # Parquet 사용 (CSV보다 5-10배 빠름)
   df.to_parquet('data.parquet', compression='snappy')
   df = pd.read_parquet('data.parquet')
   ```

2. **메모리 사용량 모니터링**:
   ```python
   # 메모리 프로파일링
   df.info(memory_usage='deep')
   
   # 컬럼별 메모리
   df.memory_usage(deep=True).sort_values(ascending=False)
   ```

3. **병렬 처리**:
   ```python
   # 여러 파일 병렬 로드
   from joblib import Parallel, delayed
   
   files = ['file1.csv', 'file2.csv', 'file3.csv']
   dfs = Parallel(n_jobs=-1)(delayed(pd.read_csv)(f) for f in files)
   df = pd.concat(dfs, ignore_index=True)
   ```

### 8.3 자주 묻는 질문 (FAQ)

**Q1: CSV와 Parquet 중 어떤 것을 사용해야 하나요?**

A: 
- **CSV**: 사람이 읽기 쉬움, 범용성 높음, 하지만 느리고 메모리 비효율적
- **Parquet**: 압축률 높음, 읽기/쓰기 빠름, 데이터 타입 보존, 분석 작업에 최적
- **권장**: 원본은 CSV로 받아도, 검증 후 Parquet으로 저장하여 사용

**Q2: 데이터가 너무 커서 로드할 수 없어요.**

A: 
1. 샘플링 사용: `nrows` 파라미터로 일부만 로드
2. 청크 처리: `chunksize` 사용
3. 필요 컬럼만 선택: `usecols` 파라미터
4. Dask 사용: Out-of-core 처리로 메모리 한계 극복
5. 데이터베이스 활용: 필터링 후 로드

**Q3: 결측값이 다양한 형태로 표현되어 있어요.**

A:
```python
# 다양한 결측값 표현을 NA로 인식
na_values = ['NA', 'N/A', 'null', 'NULL', '', ' ', '-', '?', 'unknown']
df = pd.read_csv('file.csv', na_values=na_values, keep_default_na=True)
```

---

## 9. 참고 자료 (References)

### 9.1 공식 문서

1. **Pandas Documentation**
   - URL: https://pandas.pydata.org/docs/
   - 주요 페이지:
     - [IO Tools](https://pandas.pydata.org/docs/user_guide/io.html)
     - [Data Types](https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes)
     - [Memory Usage](https://pandas.pydata.org/docs/user_guide/scale.html)

2. **NumPy Documentation**
   - URL: https://numpy.org/doc/stable/
   - 주요 페이지:
     - [Data Types](https://numpy.org/doc/stable/user/basics.types.html)

3. **SQLAlchemy Documentation**
   - URL: https://docs.sqlalchemy.org/
   - 주요 페이지:
     - [Engine Configuration](https://docs.sqlalchemy.org/en/20/core/engines.html)

### 9.2 베스트 프랙티스

1. **데이터 로딩 체크리스트**:
   - 파일 존재 및 권한 확인
   - 샘플 데이터로 먼저 테스트
   - 인코딩 명시적 지정
   - 데이터 타입 사전 지정 (대용량 데이터)
   - 에러 처리 전략 수립

2. **메모리 효율성**:
   - 필요한 컬럼만 로드
   - 적절한 데이터 타입 사용
   - 범주형 변수는 category 타입
   - 대용량 파일은 청크 처리
   - 사용 후 메모리 해제 (`del df`, `gc.collect()`)

3. **데이터 검증 전략**:
   - 구조적 검증 → 내용 검증 → 논리적 검증 순서
   - 자동화된 검증 스크립트 작성
   - 검증 실패 시 명확한 에러 메시지
   - 검증 결과를 로그로 기록
   - 재현 가능한 검증 프로세스 구축

### 9.3 추가 학습 자료

1. **Pandas Performance Optimization**
   - [Enhancing Performance](https://pandas.pydata.org/docs/user_guide/enhancingperf.html)
   - [Working with Large Datasets](https://realpython.com/pandas-read-write-files/)

2. **Data Validation Best Practices**
   - [Great Expectations Documentation](https://docs.greatexpectations.io/)
   - [Pandera: Statistical Data Validation](https://pandera.readthedocs.io/)

3. **File Format Comparison**
   - [Parquet vs CSV Performance](https://towardsdatascience.com/csv-files-for-storage-no-thanks-theres-a-better-option-72c78a414d1d)

---

## 변경 이력 (Change Log)

| 버전 | 날짜 | 변경 내용 | 작성자 |
|------|------|-----------|--------|
| 1.0 | 2025-01-25 | 초안 작성 | Claude Code |

---

**문서 끝**

다음 단계: [02-automated-profiling.md](./02-automated-profiling.md) - 자동화된 프로파일링
