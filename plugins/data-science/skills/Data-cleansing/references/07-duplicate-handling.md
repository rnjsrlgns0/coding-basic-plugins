# Duplicate Handling (중복 데이터 처리)

**생성일**: 2025-01-25  
**버전**: 1.0  
**담당 에이전트**: `data-cleaning-specialist`

---

## 1. 개요

### 1.1 목적

중복 데이터 처리는 데이터 무결성과 분석 정확성을 보장하는 필수 프로세스입니다. 이 레퍼런스는 다음을 제공합니다:

- **중복 유형 식별**: 완전 중복, 키 중복, 유사 중복(fuzzy duplicates)
- **탐지 방법**: 효율적이고 포괄적인 중복 탐지 기법
- **제거 전략**: 상황별 최적의 중복 제거 방법
- **검증**: 제거 후 데이터 품질 확인

### 1.2 중복이 발생하는 이유

**일반적 원인**:
1. **데이터 수집**: 중복 입력, 동일 데이터 여러 번 수집
2. **데이터 병합**: 여러 소스 통합 시 중복
3. **시스템 오류**: 데이터베이스 동기화 오류, 트랜잭션 재시도
4. **사용자 행동**: 폼 재제출, 여러 계정 생성
5. **데이터 마이그레이션**: 시스템 이전 중 중복 발생

**중복의 영향**:
- 통계 왜곡 (평균, 비율 등)
- 저장 공간 낭비
- ML 모델 과적합
- 비즈니스 결정 오류
- 규정 준수 문제

### 1.3 중복 vs 유사

- **완전 중복**: 모든 필드가 정확히 동일
- **키 중복**: ID나 고유 식별자만 동일, 다른 필드는 다름
- **유사 중복**: 오타, 형식 차이로 인한 거의 동일한 레코드

---

## 2. 이론적 배경

### 2.1 중복 유형

#### 2.1.1 완전 중복 (Exact Duplicates)

모든 컬럼의 값이 정확히 동일한 레코드.

```
ID  Name    Age  City
1   John    30   Seoul
2   Jane    25   Busan
1   John    30   Seoul  ← 완전 중복
```

**특징**:
- 가장 쉽게 탐지
- 명확한 제거 대상
- `df.duplicated()` 직접 사용 가능

#### 2.1.2 키 기반 중복 (Key-Based Duplicates)

고유해야 할 키(ID)가 중복되지만 다른 필드는 다름.

```
ID  Name    Age  City
1   John    30   Seoul
2   Jane    25   Busan
1   John    31   Incheon  ← 키 중복 (ID=1)
```

**문제점**:
- 데이터 무결성 위반
- 어느 레코드가 정확한지 판단 필요
- 비즈니스 로직 고려 필요

**처리 전략**:
- 최신 레코드 유지 (timestamp 기반)
- 가장 완전한 레코드 유지 (non-null 많은 것)
- 레코드 병합 (각 필드의 best value 선택)

#### 2.1.3 유사 중복 (Fuzzy Duplicates)

문자열 유사도가 높지만 정확히 같지 않은 레코드.

```
ID  Name          Email
1   John Smith    john@email.com
2   Jon Smith     john@email.com   ← 이름 오타
3   John Smith    john@emil.com    ← 이메일 오타
```

**특징**:
- 사람 이름, 주소, 이메일에서 흔함
- 문자열 유사도 알고리즘 필요
- 수동 검토 필요할 수 있음

**유사도 측정**:
- Levenshtein distance (편집 거리)
- Jaccard similarity (집합 유사도)
- Jaro-Winkler distance

### 2.2 중복 제거 전략

| 전략 | 설명 | 사용 시기 |
|------|------|----------|
| **First** | 첫 번째 발생 유지 | 순서가 중요하지 않을 때 |
| **Last** | 마지막 발생 유지 | 최신 데이터 선호 |
| **Most Recent** | 타임스탬프 기준 최신 | 시간 정보 있을 때 |
| **Most Complete** | 결측값 가장 적은 것 | 데이터 완전성 중요 |
| **Aggregate** | 중복 레코드 집계 | 모든 정보 보존 필요 |
| **Manual Review** | 수동 검토 후 결정 | 비즈니스 임팩트 클 때 |

### 2.3 실제 시나리오

#### 시나리오 1: CRM 고객 데이터베이스

**상황**: 50만 건 고객 데이터, 여러 영업 시스템 병합
**중복 패턴**:
- 완전 중복: 2,300건 (0.5%) - 데이터 이관 오류
- 키 중복: 8,700건 (1.7%) - 동일 고객 ID, 다른 주소/연락처
- 유사 중복: 15,000건 (3%) - 이름 오타, 이메일 변형

**처리 방안**:
1. 완전 중복: 첫 번째만 유지
2. 키 중복: 최신 타임스탬프 레코드 유지
3. 유사 중복: 이메일로 그룹화 후 수동 검토

#### 시나리오 2: 전자상거래 주문 데이터

**상황**: 일일 10만 건 주문, 결제 재시도로 인한 중복
**중복 패턴**:
- 주문 ID 중복: 500건/일 (0.5%)
- 동일 고객, 동일 시간대, 동일 상품

**처리 방안**:
1. 주문 ID + 타임스탬프 기준 중복 탐지
2. 5분 이내 동일 주문은 중복으로 간주
3. 결제 완료 상태 우선 유지

---

## 3. 구현: 상세 Python 코드

### 3.1 완전 중복 탐지 및 제거

```python
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')

def detect_full_duplicates(df: pd.DataFrame,
                          subset: List[str] = None,
                          keep: str = 'first') -> Dict:
    """
    Detect exact duplicates (all columns identical)
    
    Parameters:
    -----------
    subset : list, optional
        Columns to consider for identifying duplicates (default: all columns)
    keep : str
        'first', 'last', or False (mark all duplicates)
        
    Returns:
    --------
    dict
        Analysis report with duplicate statistics
        
    Example:
    --------
    >>> result = detect_full_duplicates(df)
    >>> print(f"Found {result['n_duplicates']} duplicate rows")
    >>> duplicate_rows = df[result['duplicate_mask']]
    """
    
    # Find duplicates
    duplicate_mask = df.duplicated(subset=subset, keep=keep)
    n_duplicates = duplicate_mask.sum()
    
    # Get duplicate groups
    if subset is None:
        subset = df.columns.tolist()
    
    duplicate_groups = df[df.duplicated(subset=subset, keep=False)].groupby(
        subset, dropna=False
    ).size().reset_index(name='count')
    duplicate_groups = duplicate_groups[duplicate_groups['count'] > 1]
    duplicate_groups = duplicate_groups.sort_values('count', ascending=False)
    
    analysis = {
        'n_total': len(df),
        'n_duplicates': n_duplicates,
        'duplicate_pct': round(100 * n_duplicates / len(df), 2),
        'n_unique': len(df) - n_duplicates,
        'n_duplicate_groups': len(duplicate_groups),
        'duplicate_mask': duplicate_mask,
        'duplicate_groups': duplicate_groups.head(20),  # Top 20
        'largest_group': duplicate_groups.iloc[0]['count'] if len(duplicate_groups) > 0 else 0
    }
    
    return analysis


def remove_full_duplicates(df: pd.DataFrame,
                           subset: List[str] = None,
                           keep: str = 'first') -> pd.DataFrame:
    """
    Remove exact duplicates
    
    Parameters:
    -----------
    keep : str
        Which duplicates to keep: 'first', 'last'
        
    Returns:
    --------
    pd.DataFrame
        Deduplicated dataframe
        
    Example:
    --------
    >>> df_clean = remove_full_duplicates(df, keep='first')
    >>> print(f"Removed {len(df) - len(df_clean)} rows")
    """
    
    initial_count = len(df)
    df_clean = df.drop_duplicates(subset=subset, keep=keep)
    removed_count = initial_count - len(df_clean)
    
    print(f"✓ Removed {removed_count} duplicate rows ({100*removed_count/initial_count:.2f}%)")
    print(f"  Before: {initial_count} rows")
    print(f"  After: {len(df_clean)} rows")
    
    return df_clean
```

### 3.2 키 기반 중복 탐지

```python
def detect_key_duplicates(df: pd.DataFrame,
                         key_columns: Union[str, List[str]],
                         exclude_exact: bool = True) -> Dict:
    """
    Detect duplicates based on key columns only
    
    Parameters:
    -----------
    key_columns : str or list
        Column(s) that should be unique
    exclude_exact : bool
        If True, exclude exact duplicates from count
        
    Returns:
    --------
    dict
        Analysis including duplicate groups
        
    Example:
    --------
    >>> result = detect_key_duplicates(df, key_columns='customer_id')
    >>> print(f"Found {result['n_duplicate_keys']} duplicate customer IDs")
    """
    
    if isinstance(key_columns, str):
        key_columns = [key_columns]
    
    # Find key duplicates
    key_duplicates = df[df.duplicated(subset=key_columns, keep=False)]
    
    if exclude_exact:
        # Exclude rows that are exact duplicates (all columns same)
        exact_duplicates = df.duplicated(keep=False)
        key_duplicates = key_duplicates[~exact_duplicates]
    
    # Group by key and count
    duplicate_groups = key_duplicates.groupby(key_columns).size().reset_index(name='count')
    duplicate_groups = duplicate_groups.sort_values('count', ascending=False)
    
    analysis = {
        'key_columns': key_columns,
        'n_total': len(df),
        'n_duplicate_keys': len(key_duplicates),
        'duplicate_key_pct': round(100 * len(key_duplicates) / len(df), 2),
        'n_affected_groups': len(duplicate_groups),
        'duplicate_groups': duplicate_groups.head(20),
        'largest_group': duplicate_groups.iloc[0]['count'] if len(duplicate_groups) > 0 else 0
    }
    
    return analysis


def analyze_key_duplicate_details(df: pd.DataFrame,
                                  key_columns: Union[str, List[str]],
                                  key_value: Any) -> pd.DataFrame:
    """
    Get all records for a specific duplicate key
    
    Useful for manual review of what differs between duplicates
    
    Example:
    --------
    >>> # See all records with customer_id = 12345
    >>> details = analyze_key_duplicate_details(df, 'customer_id', 12345)
    >>> print(details)
    """
    
    if isinstance(key_columns, str):
        key_columns = [key_columns]
    
    if len(key_columns) == 1:
        mask = df[key_columns[0]] == key_value
    else:
        mask = (df[key_columns] == key_value).all(axis=1)
    
    return df[mask].copy()
```

### 3.3 유사 중복 탐지 (Fuzzy Matching)

```python
from difflib import SequenceMatcher

def calculate_string_similarity(str1: str, str2: str) -> float:
    """
    Calculate similarity ratio between two strings (0 to 1)
    
    Uses SequenceMatcher (similar to Levenshtein but faster)
    """
    return SequenceMatcher(None, str(str1).lower(), str(str2).lower()).ratio()


def find_fuzzy_duplicates(df: pd.DataFrame,
                         column: str,
                         threshold: float = 0.85,
                         sample_size: int = None) -> List[Dict]:
    """
    Find fuzzy duplicates based on string similarity
    
    Parameters:
    -----------
    column : str
        Column to check for fuzzy duplicates
    threshold : float
        Similarity threshold (0.0 to 1.0, default 0.85)
    sample_size : int, optional
        Limit comparison to sample (for performance)
        
    Returns:
    --------
    list
        List of similar pairs with similarity scores
        
    Example:
    --------
    >>> fuzzy_dups = find_fuzzy_duplicates(df, 'name', threshold=0.90)
    >>> for pair in fuzzy_dups[:10]:
    ...     print(f"{pair['string1']} ~ {pair['string2']} ({pair['similarity']:.2f})")
    """
    
    # Get non-null unique values
    unique_values = df[column].dropna().unique()
    
    if sample_size and len(unique_values) > sample_size:
        unique_values = np.random.choice(unique_values, sample_size, replace=False)
    
    similar_pairs = []
    checked = set()
    
    for i, val1 in enumerate(unique_values):
        if i % 100 == 0:
            print(f"Processed {i}/{len(unique_values)} values...")
        
        for j, val2 in enumerate(unique_values[i+1:], i+1):
            # Skip if already checked
            pair_key = tuple(sorted([val1, val2]))
            if pair_key in checked:
                continue
            checked.add(pair_key)
            
            similarity = calculate_string_similarity(val1, val2)
            
            if similarity >= threshold:
                similar_pairs.append({
                    'string1': val1,
                    'string2': val2,
                    'similarity': round(similarity, 3),
                    'count1': (df[column] == val1).sum(),
                    'count2': (df[column] == val2).sum()
                })
    
    # Sort by similarity descending
    similar_pairs = sorted(similar_pairs, key=lambda x: x['similarity'], reverse=True)
    
    return similar_pairs


def group_fuzzy_duplicates(df: pd.DataFrame,
                          column: str,
                          threshold: float = 0.85) -> Dict[str, List[str]]:
    """
    Group similar strings together
    
    Returns:
    --------
    dict
        Mapping of representative string to list of similar strings
        
    Example:
    --------
    >>> groups = group_fuzzy_duplicates(df, 'company_name', threshold=0.90)
    >>> for rep, similars in groups.items():
    ...     print(f"{rep}: {similars}")
    """
    
    similar_pairs = find_fuzzy_duplicates(df, column, threshold)
    
    # Build groups using union-find approach
    groups_dict = {}
    string_to_group = {}
    
    for pair in similar_pairs:
        str1, str2 = pair['string1'], pair['string2']
        
        # Find existing groups
        group1 = string_to_group.get(str1)
        group2 = string_to_group.get(str2)
        
        if group1 is None and group2 is None:
            # Create new group
            group_id = str1  # Use first string as group ID
            groups_dict[group_id] = [str1, str2]
            string_to_group[str1] = group_id
            string_to_group[str2] = group_id
        
        elif group1 is None:
            # Add str1 to str2's group
            groups_dict[group2].append(str1)
            string_to_group[str1] = group2
        
        elif group2 is None:
            # Add str2 to str1's group
            groups_dict[group1].append(str2)
            string_to_group[str2] = group1
        
        elif group1 != group2:
            # Merge two groups
            groups_dict[group1].extend(groups_dict[group2])
            # Update mapping
            for s in groups_dict[group2]:
                string_to_group[s] = group1
            del groups_dict[group2]
    
    return groups_dict
```

### 3.4 고급 중복 제거 전략

```python
def remove_duplicates_most_recent(df: pd.DataFrame,
                                  key_columns: Union[str, List[str]],
                                  timestamp_column: str) -> pd.DataFrame:
    """
    Keep most recent record based on timestamp
    
    Example:
    --------
    >>> df_clean = remove_duplicates_most_recent(
    ...     df, key_columns='user_id', timestamp_column='updated_at'
    ... )
    """
    
    if isinstance(key_columns, str):
        key_columns = [key_columns]
    
    # Sort by timestamp descending (most recent first)
    df_sorted = df.sort_values(timestamp_column, ascending=False)
    
    # Keep first (most recent) of each key
    df_clean = df_sorted.drop_duplicates(subset=key_columns, keep='first')
    
    removed = len(df) - len(df_clean)
    print(f"✓ Kept most recent records: removed {removed} duplicates")
    
    return df_clean


def remove_duplicates_most_complete(df: pd.DataFrame,
                                   key_columns: Union[str, List[str]]) -> pd.DataFrame:
    """
    Keep record with fewest missing values
    
    Example:
    --------
    >>> df_clean = remove_duplicates_most_complete(df, key_columns='customer_id')
    """
    
    if isinstance(key_columns, str):
        key_columns = [key_columns]
    
    # Calculate completeness score (number of non-null values)
    df_scored = df.copy()
    df_scored['_completeness'] = df.notnull().sum(axis=1)
    
    # Sort by completeness descending
    df_sorted = df_scored.sort_values('_completeness', ascending=False)
    
    # Keep most complete
    df_clean = df_sorted.drop_duplicates(subset=key_columns, keep='first')
    df_clean = df_clean.drop('_completeness', axis=1)
    
    removed = len(df) - len(df_clean)
    print(f"✓ Kept most complete records: removed {removed} duplicates")
    
    return df_clean


def aggregate_duplicates(df: pd.DataFrame,
                        key_columns: Union[str, List[str]],
                        agg_strategy: Dict[str, str] = None) -> pd.DataFrame:
    """
    Aggregate duplicate records instead of dropping
    
    Parameters:
    -----------
    agg_strategy : dict
        Column-specific aggregation strategies
        e.g., {'amount': 'sum', 'name': 'first', 'count': 'count'}
        
    Example:
    --------
    >>> df_agg = aggregate_duplicates(
    ...     df,
    ...     key_columns='user_id',
    ...     agg_strategy={'purchase_amount': 'sum', 'name': 'first'}
    ... )
    """
    
    if isinstance(key_columns, str):
        key_columns = [key_columns]
    
    if agg_strategy is None:
        # Default strategy
        agg_strategy = {}
        for col in df.columns:
            if col not in key_columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    agg_strategy[col] = 'mean'
                else:
                    agg_strategy[col] = 'first'
    
    df_agg = df.groupby(key_columns, as_index=False).agg(agg_strategy)
    
    print(f"✓ Aggregated {len(df)} rows into {len(df_agg)} rows")
    
    return df_agg
```

### 3.5 중복 제거 검증

```python
def validate_duplicate_removal(df_before: pd.DataFrame,
                              df_after: pd.DataFrame,
                              key_columns: Union[str, List[str]] = None) -> Dict:
    """
    Validate that duplicates were properly removed
    
    Returns:
    --------
    dict
        Validation report
    """
    
    if isinstance(key_columns, str):
        key_columns = [key_columns]
    
    validation = {
        'rows_before': len(df_before),
        'rows_after': len(df_after),
        'rows_removed': len(df_before) - len(df_after),
        'removal_pct': round(100 * (len(df_before) - len(df_after)) / len(df_before), 2)
    }
    
    # Check for remaining full duplicates
    remaining_full = df_after.duplicated().sum()
    validation['remaining_full_duplicates'] = remaining_full
    
    # Check for remaining key duplicates
    if key_columns:
        remaining_key = df_after.duplicated(subset=key_columns).sum()
        validation['remaining_key_duplicates'] = remaining_key
    
    # Check data integrity
    validation['columns_before'] = len(df_before.columns)
    validation['columns_after'] = len(df_after.columns)
    validation['columns_match'] = (validation['columns_before'] == validation['columns_after'])
    
    # Overall status
    if remaining_full == 0 and (key_columns is None or remaining_key == 0):
        validation['status'] = 'PASS'
    else:
        validation['status'] = 'FAIL'
    
    return validation


def print_duplicate_summary(df: pd.DataFrame,
                           key_columns: Union[str, List[str]] = None) -> None:
    """
    Print comprehensive duplicate summary
    """
    
    print("=" * 80)
    print("DUPLICATE ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Full duplicates
    full_dup = detect_full_duplicates(df)
    print(f"\n1. Full Duplicates (all columns identical):")
    print(f"   Total rows: {full_dup['n_total']}")
    print(f"   Duplicate rows: {full_dup['n_duplicates']} ({full_dup['duplicate_pct']}%)")
    print(f"   Unique rows: {full_dup['n_unique']}")
    
    # Key duplicates
    if key_columns:
        key_dup = detect_key_duplicates(df, key_columns)
        print(f"\n2. Key Duplicates (based on {key_columns}):")
        print(f"   Duplicate keys: {key_dup['n_duplicate_keys']} ({key_dup['duplicate_key_pct']}%)")
        print(f"   Affected groups: {key_dup['n_affected_groups']}")
        if len(key_dup['duplicate_groups']) > 0:
            print(f"\n   Top duplicate groups:")
            print(key_dup['duplicate_groups'].head())
    
    print("\n" + "=" * 80)
```

---

## 4. 예시: 입출력 샘플

```python
# Create sample data with various duplicates
np.random.seed(42)
n = 1000

# Base data
df_sample = pd.DataFrame({
    'user_id': np.random.randint(1, 500, n),
    'name': np.random.choice(['John Smith', 'Jane Doe', 'Bob Johnson', 'Alice Brown'], n),
    'email': [f'user{i}@example.com' for i in np.random.randint(1, 500, n)],
    'purchase_amount': np.random.uniform(10, 1000, n),
    'timestamp': pd.date_range('2024-01-01', periods=n, freq='H')
})

# Add exact duplicates
df_sample = pd.concat([df_sample, df_sample.iloc[:50]], ignore_index=True)

# Add key duplicates (same user_id, different data)
key_dups = df_sample.iloc[:30].copy()
key_dups['purchase_amount'] = np.random.uniform(10, 1000, 30)
df_sample = pd.concat([df_sample, key_dups], ignore_index=True)

print("Sample data created with duplicates")
print(f"Total rows: {len(df_sample)}")

# Detect duplicates
print("\n" + "="*80)
full_dup_result = detect_full_duplicates(df_sample)
print(f"Full duplicates: {full_dup_result['n_duplicates']}")

key_dup_result = detect_key_duplicates(df_sample, 'user_id')
print(f"Key duplicates: {key_dup_result['n_duplicate_keys']}")

# Remove duplicates with different strategies
print("\n" + "="*80)
print("Strategy 1: Remove exact duplicates (keep first)")
df_clean1 = remove_full_duplicates(df_sample, keep='first')

print("\nStrategy 2: Remove by key, keep most recent")
df_clean2 = remove_duplicates_most_recent(df_sample, 'user_id', 'timestamp')

print("\nStrategy 3: Aggregate by key")
df_clean3 = aggregate_duplicates(
    df_sample,
    key_columns='user_id',
    agg_strategy={'purchase_amount': 'sum', 'name': 'first'}
)

# Validate
print("\n" + "="*80)
print("Validation:")
validation = validate_duplicate_removal(df_sample, df_clean1, 'user_id')
for key, value in validation.items():
    print(f"{key}: {value}")
```

---

## 5. 에이전트 매핑

- **Primary**: `data-cleaning-specialist` - 모든 중복 처리
- **Supporting**: `data-scientist` - 유사 중복 알고리즘

---

## 6. 필요 라이브러리

```bash
pip install pandas>=2.0.0
pip install numpy>=1.24.0
```

---

## 7. 체크포인트

- [ ] 완전 중복을 확인했는가?
- [ ] 키 중복을 확인했는가?
- [ ] 유사 중복을 확인했는가?
- [ ] 적절한 제거 전략을 선택했는가?
- [ ] 제거 후 검증을 수행했는가?

---

## 8. 트러블슈팅

### 문제 1: Fuzzy matching이 너무 느림
**해결**: sample_size 파라미터로 샘플링

### 문제 2: 어떤 레코드를 유지할지 모름
**해결**: 비즈니스 팀과 협의, 도메인 지식 활용

### 문제 3: 중복 제거 후 필요한 데이터 손실
**해결**: aggregate_duplicates()로 정보 보존

---

## 9. 참고 자료

- Pandas Duplicates: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.duplicated.html
- String Matching Algorithms

---

## 10. 요약

**핵심 원칙**:
1. 중복 유형 명확히 구분
2. 비즈니스 로직 고려
3. 정보 손실 최소화
4. 항상 검증 수행

**다음 단계**: Reference 08 (Data Standardization)

---

**작성자**: Claude Code  
**마지막 업데이트**: 2025-01-25
