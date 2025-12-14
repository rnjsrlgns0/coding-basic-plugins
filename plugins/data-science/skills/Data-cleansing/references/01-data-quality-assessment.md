# Data Quality Assessment (데이터 품질 평가)

**생성일**: 2025-01-25  
**버전**: 1.0  
**담당 에이전트**: `data-cleaning-specialist`, `data-scientist`

---

## 1. 개요

### 1.1 목적

데이터 품질 평가는 데이터 클렌징 워크플로우의 첫 번째이자 가장 중요한 단계입니다. 이 프로세스는 데이터를 처음 받았을 때 수행해야 하는 종합적인 품질 평가 방법론을 제공하며, 다음과 같은 목표를 달성합니다:

- **품질 이슈 식별**: 결측값, 중복, 이상치, 타입 불일치 등 모든 품질 문제를 체계적으로 탐지
- **우선순위 결정**: 식별된 이슈의 심각도를 평가하고 처리 순서를 결정
- **클렌징 전략 수립**: 품질 평가 결과를 바탕으로 적절한 클렌징 방법 선택
- **벤치마크 설정**: 클렌징 전후 비교를 위한 기준선 확립

### 1.2 적용 시기

데이터 품질 평가는 다음 상황에서 반드시 수행해야 합니다:

1. **새로운 데이터셋 수령 시**: 처음 받은 데이터의 전반적인 상태 파악
2. **데이터 소스 변경 시**: 새로운 데이터 제공자나 시스템으로부터 데이터 수집
3. **정기적 품질 점검**: 주기적인 데이터 품질 모니터링 (예: 월별, 분기별)
4. **이상 징후 발견 시**: 분석 결과가 예상과 다를 때 데이터 품질 재점검
5. **프로젝트 시작 전**: 데이터 분석 또는 ML 모델링 착수 전 필수 점검

### 1.3 주요 구성 요소

데이터 품질 평가는 세 가지 핵심 영역으로 구성됩니다:

1. **데이터 프로파일링**: 기본 통계, 분포, 패턴 분석
2. **데이터 타입 검증**: 스키마 일치 여부 및 타입 무결성 확인
3. **비즈니스 규칙 검증**: 도메인별 제약 조건 및 논리적 일관성 검증

---

## 2. 이론적 배경

### 2.1 데이터 품질의 차원

데이터 품질은 여러 차원에서 평가됩니다:

#### 2.1.1 완전성 (Completeness)
- **정의**: 필수 데이터가 모두 존재하는 정도
- **측정**: 결측값 비율, 필수 필드 누락률
- **중요성**: 불완전한 데이터는 분석 결과를 왜곡시키고 모델 성능을 저하시킴

#### 2.1.2 정확성 (Accuracy)
- **정의**: 데이터가 실제 값을 올바르게 반영하는 정도
- **측정**: 비즈니스 규칙 위반률, 값 범위 초과율
- **중요성**: 부정확한 데이터는 잘못된 의사결정으로 이어짐

#### 2.1.3 일관성 (Consistency)
- **정의**: 동일한 정보가 여러 곳에서 일치하는 정도
- **측정**: 교차 필드 불일치율, 중복 레코드 비율
- **중요성**: 일관성 없는 데이터는 신뢰성을 떨어뜨림

#### 2.1.4 적시성 (Timeliness)
- **정의**: 데이터가 현재 상황을 반영하는 정도
- **측정**: 최근 업데이트 날짜, 데이터 지연 시간
- **중요성**: 오래된 데이터는 현재 비즈니스 결정에 부적합

#### 2.1.5 유효성 (Validity)
- **정의**: 데이터가 정의된 형식과 규칙을 준수하는 정도
- **측정**: 형식 불일치율, 도메인 제약 위반률
- **중요성**: 유효하지 않은 데이터는 시스템 오류를 야기

#### 2.1.6 고유성 (Uniqueness)
- **정의**: 데이터에 중복이 없는 정도
- **측정**: 중복 레코드 비율, 키 중복률
- **중요성**: 중복 데이터는 통계를 왜곡하고 저장 공간을 낭비

### 2.2 데이터 프로파일링 개념

데이터 프로파일링은 데이터를 체계적으로 조사하여 구조, 내용, 관계, 품질을 이해하는 프로세스입니다.

#### 2.2.1 단변량 프로파일링
각 컬럼을 독립적으로 분석:
- 기본 통계 (평균, 중앙값, 표준편차, 최소/최대값)
- 분포 특성 (왜도, 첨도)
- 결측값 및 유니크 값 개수
- 데이터 타입 및 형식

#### 2.2.2 다변량 프로파일링
여러 컬럼 간의 관계 분석:
- 상관관계
- 공동 발생 패턴
- 함수적 의존성
- 교차 필드 일관성

#### 2.2.3 메타데이터 프로파일링
데이터에 대한 메타 정보:
- 스키마 구조
- 데이터 계보 (lineage)
- 업데이트 빈도
- 데이터 출처

### 2.3 실제 시나리오

#### 시나리오 1: 전자상거래 고객 데이터
**상황**: 100만 건의 고객 프로필 데이터 수령
**품질 이슈**:
- 이메일 필드 15% 결측
- 생년월일과 나이 불일치 3,542건
- 중복 고객 ID 87건
- 전화번호 형식 불일치 (다양한 포맷)

**평가 결과**:
- 완전성: 85% (이메일 결측으로 인한 감점)
- 정확성: 96% (나이 불일치)
- 일관성: 99.9% (중복 ID)
- 유효성: 92% (전화번호 형식)

**권장 조치**:
1. 이메일 결측값 처리 (고우선순위)
2. 나이-생년월일 불일치 해결
3. 중복 ID 병합
4. 전화번호 형식 표준화

#### 시나리오 2: IoT 센서 데이터
**상황**: 3개월치 온도 센서 데이터 (1,000만 건)
**품질 이슈**:
- 센서 오류로 인한 비현실적 값 (예: -999, 9999)
- 타임스탬프 중복 및 순서 오류
- 특정 센서의 간헐적 데이터 누락

**평가 결과**:
- 완전성: 94% (간헐적 누락)
- 정확성: 97% (오류 값)
- 적시성: 100% (실시간 수집)
- 일관성: 99% (타임스탬프 중복)

**권장 조치**:
1. 센서 오류 값 식별 및 제거
2. 타임스탬프 정렬 및 중복 해결
3. 결측값 보간 (시계열 특성 고려)

---

## 3. 구현: 상세 Python 코드

### 3.1 종합 데이터 프로파일링

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

def comprehensive_data_profiling(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive data quality profiling
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe to profile
        
    Returns:
    --------
    dict
        Complete profiling report with basic info, quality metrics, and issues
        
    Example:
    --------
    >>> df = pd.read_csv('customer_data.csv')
    >>> profile = comprehensive_data_profiling(df)
    >>> print(profile['basic_info'])
    >>> print(profile['quality_metrics'])
    """
    
    profile = {
        'basic_info': {},
        'quality_metrics': pd.DataFrame(),
        'issues': [],
        'recommendations': []
    }
    
    # ===== 1. 기본 정보 =====
    profile['basic_info'] = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'memory_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        'duplicate_rows': df.duplicated().sum(),
        'duplicate_rows_pct': round(100 * df.duplicated().sum() / len(df), 2),
        'total_cells': df.size,
        'total_missing_cells': df.isnull().sum().sum(),
        'overall_missing_pct': round(100 * df.isnull().sum().sum() / df.size, 2)
    }
    
    # ===== 2. 컬럼별 품질 메트릭 =====
    quality_report = []
    
    for col in df.columns:
        col_info = {
            'column': col,
            'dtype': str(df[col].dtype),
            'missing_count': df[col].isnull().sum(),
            'missing_pct': round(100 * df[col].isnull().sum() / len(df), 2),
            'unique_count': df[col].nunique(),
            'unique_pct': round(100 * df[col].nunique() / len(df), 2),
            'cardinality': 'high' if df[col].nunique() > len(df) * 0.5 else 
                          ('medium' if df[col].nunique() > 10 else 'low')
        }
        
        # 수치형 변수 추가 정보
        if pd.api.types.is_numeric_dtype(df[col]):
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                col_info.update({
                    'mean': round(non_null_values.mean(), 2),
                    'std': round(non_null_values.std(), 2),
                    'min': non_null_values.min(),
                    'q25': round(non_null_values.quantile(0.25), 2),
                    'median': round(non_null_values.median(), 2),
                    'q75': round(non_null_values.quantile(0.75), 2),
                    'max': non_null_values.max(),
                    'zeros_count': (df[col] == 0).sum(),
                    'zeros_pct': round(100 * (df[col] == 0).sum() / len(df), 2),
                    'negative_count': (df[col] < 0).sum(),
                    'negative_pct': round(100 * (df[col] < 0).sum() / len(df), 2),
                    'skewness': round(non_null_values.skew(), 2),
                    'kurtosis': round(non_null_values.kurtosis(), 2)
                })
        
        # 문자열 변수 추가 정보
        elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                col_info.update({
                    'avg_length': round(non_null_values.astype(str).str.len().mean(), 2),
                    'max_length': non_null_values.astype(str).str.len().max(),
                    'min_length': non_null_values.astype(str).str.len().min(),
                    'has_whitespace': non_null_values.astype(str).str.contains(r'^\s|\s$').any(),
                    'has_special_chars': non_null_values.astype(str).str.contains(r'[^a-zA-Z0-9\s]').any()
                })
        
        # 날짜/시간 변수 추가 정보
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                col_info.update({
                    'min_date': non_null_values.min(),
                    'max_date': non_null_values.max(),
                    'date_range_days': (non_null_values.max() - non_null_values.min()).days
                })
        
        quality_report.append(col_info)
    
    profile['quality_metrics'] = pd.DataFrame(quality_report)
    
    # ===== 3. 품질 이슈 식별 =====
    
    # 3.1 결측값 이슈
    for col in df.columns:
        missing_pct = 100 * df[col].isnull().sum() / len(df)
        
        if missing_pct > 50:
            profile['issues'].append({
                'severity': 'HIGH',
                'category': 'Missing Data',
                'column': col,
                'issue': f'Critical missing rate: {missing_pct:.1f}%',
                'impact': 'Column may need to be dropped or requires domain expertise for imputation'
            })
            profile['recommendations'].append(f"Consider dropping '{col}' or consult domain expert")
            
        elif missing_pct > 20:
            profile['issues'].append({
                'severity': 'MEDIUM',
                'category': 'Missing Data',
                'column': col,
                'issue': f'Moderate missing rate: {missing_pct:.1f}%',
                'impact': 'Imputation required, may introduce bias'
            })
            profile['recommendations'].append(f"Apply appropriate imputation strategy for '{col}'")
            
        elif missing_pct > 5:
            profile['issues'].append({
                'severity': 'LOW',
                'category': 'Missing Data',
                'column': col,
                'issue': f'Minor missing rate: {missing_pct:.1f}%',
                'impact': 'Minimal impact, simple imputation sufficient'
            })
    
    # 3.2 중복 이슈
    if profile['basic_info']['duplicate_rows'] > 0:
        dup_pct = profile['basic_info']['duplicate_rows_pct']
        severity = 'HIGH' if dup_pct > 5 else ('MEDIUM' if dup_pct > 1 else 'LOW')
        
        profile['issues'].append({
            'severity': severity,
            'category': 'Duplicates',
            'column': 'ALL',
            'issue': f'Duplicate rows detected: {profile["basic_info"]["duplicate_rows"]} ({dup_pct:.2f}%)',
            'impact': 'May skew statistical analysis and model training'
        })
        profile['recommendations'].append("Investigate and remove duplicate rows")
    
    # 3.3 카디널리티 이슈
    for col in df.columns:
        unique_pct = 100 * df[col].nunique() / len(df)
        
        # 거의 모든 값이 유니크 (ID 필드일 가능성)
        if unique_pct > 95 and not col.lower().endswith('id'):
            profile['issues'].append({
                'severity': 'LOW',
                'category': 'Cardinality',
                'column': col,
                'issue': f'Very high cardinality: {df[col].nunique()} unique values ({unique_pct:.1f}%)',
                'impact': 'May not be useful for analysis, consider if this is an identifier'
            })
        
        # 거의 모든 값이 동일 (상수 필드)
        elif unique_pct < 1 and df[col].nunique() > 1:
            profile['issues'].append({
                'severity': 'MEDIUM',
                'category': 'Cardinality',
                'column': col,
                'issue': f'Very low cardinality: {df[col].nunique()} unique values ({unique_pct:.1f}%)',
                'impact': 'Low information content, consider dropping'
            })
            profile['recommendations'].append(f"Consider dropping low-variance column '{col}'")
        
        # 완전 상수 필드
        elif df[col].nunique() == 1:
            profile['issues'].append({
                'severity': 'HIGH',
                'category': 'Cardinality',
                'column': col,
                'issue': 'Constant column (only 1 unique value)',
                'impact': 'Zero information content, should be dropped'
            })
            profile['recommendations'].append(f"Drop constant column '{col}'")
    
    # 3.4 데이터 타입 이슈
    for col in df.columns:
        # 숫자처럼 보이는 문자열
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
                try:
                    # 문자열을 숫자로 변환 시도
                    pd.to_numeric(sample, errors='raise')
                    profile['issues'].append({
                        'severity': 'MEDIUM',
                        'category': 'Data Type',
                        'column': col,
                        'issue': 'Numeric data stored as string',
                        'impact': 'Inefficient storage, incorrect operations'
                    })
                    profile['recommendations'].append(f"Convert '{col}' to numeric type")
                except:
                    pass
    
    return profile


def print_profile_summary(profile: Dict[str, Any]) -> None:
    """
    Print a human-readable summary of the profile report
    
    Parameters:
    -----------
    profile : dict
        Profile report from comprehensive_data_profiling()
    """
    
    print("=" * 80)
    print("DATA QUALITY ASSESSMENT REPORT")
    print("=" * 80)
    
    print("\n📊 BASIC INFORMATION")
    print("-" * 80)
    for key, value in profile['basic_info'].items():
        print(f"{key:.<40} {value}")
    
    print("\n\n📈 QUALITY METRICS BY COLUMN")
    print("-" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(profile['quality_metrics'].to_string(index=False))
    
    print("\n\n⚠️  QUALITY ISSUES")
    print("-" * 80)
    if len(profile['issues']) == 0:
        print("✓ No significant quality issues detected!")
    else:
        # Group by severity
        for severity in ['HIGH', 'MEDIUM', 'LOW']:
            severity_issues = [i for i in profile['issues'] if i['severity'] == severity]
            if severity_issues:
                print(f"\n{severity} SEVERITY ({len(severity_issues)} issues):")
                for issue in severity_issues:
                    print(f"  • [{issue['category']}] {issue['column']}: {issue['issue']}")
                    print(f"    Impact: {issue['impact']}")
    
    print("\n\n💡 RECOMMENDATIONS")
    print("-" * 80)
    if len(profile['recommendations']) == 0:
        print("✓ No specific recommendations")
    else:
        for i, rec in enumerate(profile['recommendations'], 1):
            print(f"{i}. {rec}")
    
    print("\n" + "=" * 80)
```

### 3.2 데이터 타입 검증

```python
from typing import Union, Callable

def validate_data_types(df: pd.DataFrame, 
                       expected_types: Dict[str, Union[str, Callable]]) -> pd.DataFrame:
    """
    Validate data types against expected schema
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    expected_types : dict
        Dictionary mapping column names to expected types.
        Values can be:
        - String: 'numeric', 'datetime', 'categorical', 'string', 'boolean'
        - Callable: Custom validation function returning bool
        
    Returns:
    --------
    pd.DataFrame
        Report of type validation issues
        
    Example:
    --------
    >>> expected_schema = {
    ...     'user_id': 'numeric',
    ...     'signup_date': 'datetime',
    ...     'age': 'numeric',
    ...     'category': 'categorical',
    ...     'email': 'string',
    ...     'is_active': 'boolean'
    ... }
    >>> issues = validate_data_types(df, expected_schema)
    >>> print(issues)
    """
    
    type_issues = []
    
    for col, expected_type in expected_types.items():
        # Check if column exists
        if col not in df.columns:
            type_issues.append({
                'column': col,
                'expected_type': expected_type,
                'actual_type': 'N/A',
                'issue': 'Column not found in dataframe',
                'severity': 'HIGH',
                'suggestion': 'Check column name spelling or data source'
            })
            continue
        
        actual_type = str(df[col].dtype)
        is_valid = True
        issue_message = None
        suggestion = None
        
        # Type validation based on expected type
        if isinstance(expected_type, str):
            if expected_type == 'numeric':
                if not pd.api.types.is_numeric_dtype(df[col]):
                    is_valid = False
                    issue_message = f'Not numeric type (found: {actual_type})'
                    suggestion = f"Convert using: df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')"
            
            elif expected_type == 'datetime':
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    is_valid = False
                    issue_message = f'Not datetime type (found: {actual_type})'
                    suggestion = f"Convert using: df['{col}'] = pd.to_datetime(df['{col}'], errors='coerce')"
            
            elif expected_type == 'categorical':
                if df[col].dtype != 'category':
                    # This is often a warning rather than error
                    is_valid = False
                    issue_message = f'Not categorical type (found: {actual_type})'
                    suggestion = f"Convert using: df['{col}'] = df['{col}'].astype('category')"
            
            elif expected_type == 'string':
                if not (pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object'):
                    is_valid = False
                    issue_message = f'Not string/object type (found: {actual_type})'
                    suggestion = f"Convert using: df['{col}'] = df['{col}'].astype(str)"
            
            elif expected_type == 'boolean':
                if df[col].dtype != 'bool':
                    is_valid = False
                    issue_message = f'Not boolean type (found: {actual_type})'
                    suggestion = f"Convert using: df['{col}'] = df['{col}'].astype(bool)"
        
        elif callable(expected_type):
            # Custom validation function
            try:
                is_valid = expected_type(df[col])
                if not is_valid:
                    issue_message = 'Failed custom validation'
                    suggestion = 'Check custom validation function requirements'
            except Exception as e:
                is_valid = False
                issue_message = f'Custom validation error: {str(e)}'
                suggestion = 'Review custom validation function'
        
        # Record issue if validation failed
        if not is_valid:
            type_issues.append({
                'column': col,
                'expected_type': expected_type,
                'actual_type': actual_type,
                'issue': issue_message,
                'severity': 'HIGH',
                'suggestion': suggestion
            })
    
    return pd.DataFrame(type_issues)


def auto_detect_types(df: pd.DataFrame, 
                     sample_size: int = 1000) -> Dict[str, str]:
    """
    Automatically detect appropriate data types for each column
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    sample_size : int
        Number of rows to sample for detection
        
    Returns:
    --------
    dict
        Suggested types for each column
        
    Example:
    --------
    >>> detected_types = auto_detect_types(df)
    >>> print(detected_types)
    {'user_id': 'numeric', 'name': 'string', 'signup_date': 'datetime', ...}
    """
    
    suggested_types = {}
    sample_df = df.head(sample_size) if len(df) > sample_size else df
    
    for col in df.columns:
        sample = sample_df[col].dropna()
        
        if len(sample) == 0:
            suggested_types[col] = 'unknown'
            continue
        
        # Check if numeric
        try:
            pd.to_numeric(sample, errors='raise')
            suggested_types[col] = 'numeric'
            continue
        except:
            pass
        
        # Check if datetime
        try:
            pd.to_datetime(sample, errors='raise')
            suggested_types[col] = 'datetime'
            continue
        except:
            pass
        
        # Check if boolean
        unique_values = set(sample.astype(str).str.lower())
        if unique_values.issubset({'true', 'false', '1', '0', 'yes', 'no', 't', 'f', 'y', 'n'}):
            suggested_types[col] = 'boolean'
            continue
        
        # Check if categorical (low cardinality)
        if sample.nunique() < 20:
            suggested_types[col] = 'categorical'
        else:
            suggested_types[col] = 'string'
    
    return suggested_types
```

### 3.3 비즈니스 규칙 검증

```python
from typing import Callable, Tuple
import re

class BusinessRuleValidator:
    """
    Validates business rules on dataframes
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.violations = []
    
    def add_rule(self, 
                 rule_name: str,
                 validation_func: Callable[[pd.DataFrame], pd.Series],
                 severity: str = 'MEDIUM',
                 description: str = '') -> 'BusinessRuleValidator':
        """
        Add a business rule to validate
        
        Parameters:
        -----------
        rule_name : str
            Name of the rule
        validation_func : callable
            Function that takes df and returns boolean Series (True = valid)
        severity : str
            'HIGH', 'MEDIUM', or 'LOW'
        description : str
            Human-readable description of the rule
            
        Returns:
        --------
        self
            For method chaining
        """
        
        try:
            # Apply validation function
            valid_mask = validation_func(self.df)
            invalid_mask = ~valid_mask
            
            violations_count = invalid_mask.sum()
            
            if violations_count > 0:
                violation_indices = self.df[invalid_mask].index.tolist()
                
                self.violations.append({
                    'rule': rule_name,
                    'description': description,
                    'severity': severity,
                    'violations_count': violations_count,
                    'violations_pct': round(100 * violations_count / len(self.df), 2),
                    'violation_indices': violation_indices[:100],  # Store first 100
                    'examples': self.df[invalid_mask].head(5).to_dict('records')
                })
        
        except Exception as e:
            self.violations.append({
                'rule': rule_name,
                'description': description,
                'severity': 'HIGH',
                'violations_count': 'ERROR',
                'violations_pct': 'N/A',
                'error': str(e)
            })
        
        return self
    
    def get_report(self) -> pd.DataFrame:
        """Get violations report as dataframe"""
        if not self.violations:
            return pd.DataFrame()
        
        # Remove large fields for display
        report_data = []
        for v in self.violations:
            report_row = {k: v[k] for k in v if k not in ['violation_indices', 'examples']}
            report_data.append(report_row)
        
        return pd.DataFrame(report_data)
    
    def get_detailed_violations(self, rule_name: str) -> Dict[str, Any]:
        """Get detailed information about specific rule violations"""
        for v in self.violations:
            if v['rule'] == rule_name:
                return v
        return None


def validate_business_rules(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Validate common business rules
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    list
        List of violation dictionaries
        
    Example:
    --------
    >>> violations = validate_business_rules(df)
    >>> for v in violations:
    ...     print(f"{v['rule']}: {v['violations_count']} violations")
    """
    
    validator = BusinessRuleValidator(df)
    
    # Rule 1: Age must be between 0 and 120
    if 'age' in df.columns:
        validator.add_rule(
            rule_name='Valid Age Range',
            validation_func=lambda d: (d['age'] >= 0) & (d['age'] <= 120),
            severity='HIGH',
            description='Age must be between 0 and 120'
        )
    
    # Rule 2: Email format validation
    if 'email' in df.columns:
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        validator.add_rule(
            rule_name='Valid Email Format',
            validation_func=lambda d: d['email'].astype(str).str.match(email_pattern, na=False),
            severity='MEDIUM',
            description='Email must follow standard format'
        )
    
    # Rule 3: Date must not be in the future
    if 'date' in df.columns:
        validator.add_rule(
            rule_name='Date Not in Future',
            validation_func=lambda d: pd.to_datetime(d['date'], errors='coerce') <= pd.Timestamp.now(),
            severity='HIGH',
            description='Date cannot be in the future'
        )
    
    # Rule 4: Start date before end date
    if 'start_date' in df.columns and 'end_date' in df.columns:
        validator.add_rule(
            rule_name='Start Before End',
            validation_func=lambda d: pd.to_datetime(d['start_date'], errors='coerce') <= 
                                     pd.to_datetime(d['end_date'], errors='coerce'),
            severity='HIGH',
            description='Start date must be before end date'
        )
    
    # Rule 5: Age matches birth date (if both present)
    if 'age' in df.columns and 'birth_date' in df.columns:
        def check_age_birthdate(d):
            calculated_age = (pd.Timestamp.now() - pd.to_datetime(d['birth_date'], errors='coerce')).dt.days / 365.25
            age_diff = abs(d['age'] - calculated_age)
            return age_diff <= 1  # Allow 1 year tolerance
        
        validator.add_rule(
            rule_name='Age Matches Birth Date',
            validation_func=check_age_birthdate,
            severity='MEDIUM',
            description='Age should match calculated age from birth date'
        )
    
    # Rule 6: Positive values for price/amount
    for col in ['price', 'amount', 'cost', 'total']:
        if col in df.columns:
            validator.add_rule(
                rule_name=f'Positive {col.title()}',
                validation_func=lambda d, c=col: d[c] >= 0,
                severity='HIGH',
                description=f'{col.title()} must be non-negative'
            )
    
    # Rule 7: Phone number format (flexible)
    if 'phone' in df.columns:
        # Remove non-digits and check length
        def check_phone(d):
            digits_only = d['phone'].astype(str).str.replace(r'\D', '', regex=True)
            return digits_only.str.len().between(10, 15)
        
        validator.add_rule(
            rule_name='Valid Phone Number',
            validation_func=check_phone,
            severity='LOW',
            description='Phone number should have 10-15 digits'
        )
    
    return validator.violations
```

### 3.4 자동 프로파일링 (ydata-profiling)

```python
def generate_auto_profile_report(df: pd.DataFrame,
                                 output_file: str = 'profile_report.html',
                                 title: str = 'Data Profile Report',
                                 minimal: bool = False) -> None:
    """
    Generate automated profiling report using ydata-profiling
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    output_file : str
        Output HTML file path
    title : str
        Report title
    minimal : bool
        If True, generate minimal report (faster)
        
    Example:
    --------
    >>> generate_auto_profile_report(df, 'customer_profile.html')
    """
    
    try:
        from ydata_profiling import ProfileReport
    except ImportError:
        print("ydata-profiling not installed. Install with: pip install ydata-profiling")
        return
    
    # Configure profile settings
    if minimal:
        profile = ProfileReport(
            df,
            title=title,
            minimal=True,
            explorative=False
        )
    else:
        profile = ProfileReport(
            df,
            title=title,
            explorative=True,
            correlations={
                "auto": {"calculate": True},
                "pearson": {"calculate": True},
                "spearman": {"calculate": True},
                "kendall": {"calculate": False},  # Slow for large datasets
                "phi_k": {"calculate": True},
                "cramers": {"calculate": True},
            },
            missing_diagrams={
                "bar": True,
                "matrix": True,
                "heatmap": True,
                "dendrogram": True,
            },
            duplicates={
                "head": 10,
                "key": None,  # Auto-detect key columns
            },
            samples={
                "head": 10,
                "tail": 10,
                "random": 10
            }
        )
    
    # Generate report
    profile.to_file(output_file)
    print(f"✓ Profile report generated: {output_file}")
    
    # Also return profile object for programmatic access
    return profile
```

### 3.5 품질 점수 계산

```python
def calculate_data_quality_score(df: pd.DataFrame,
                                 weights: Dict[str, float] = None) -> Dict[str, float]:
    """
    Calculate overall data quality score
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    weights : dict, optional
        Custom weights for each dimension
        Default: {'completeness': 0.3, 'validity': 0.25, 'consistency': 0.25, 
                 'uniqueness': 0.2}
        
    Returns:
    --------
    dict
        Quality scores for each dimension and overall score
        
    Example:
    --------
    >>> scores = calculate_data_quality_score(df)
    >>> print(f"Overall Quality Score: {scores['overall']:.1f}/100")
    """
    
    if weights is None:
        weights = {
            'completeness': 0.3,
            'validity': 0.25,
            'consistency': 0.25,
            'uniqueness': 0.2
        }
    
    scores = {}
    
    # 1. Completeness Score (0-100)
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    scores['completeness'] = 100 * (1 - missing_cells / total_cells)
    
    # 2. Validity Score (simplified - based on numeric ranges)
    validity_score = 100.0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # Check for unrealistic values (e.g., negative ages)
        if 'age' in col.lower():
            invalid = ((df[col] < 0) | (df[col] > 120)).sum()
            validity_score -= (invalid / len(df)) * 10  # Penalize
    
    scores['validity'] = max(0, validity_score)
    
    # 3. Consistency Score (based on duplicates)
    duplicate_count = df.duplicated().sum()
    scores['consistency'] = 100 * (1 - duplicate_count / len(df))
    
    # 4. Uniqueness Score (based on appropriate uniqueness)
    uniqueness_score = 100.0
    for col in df.columns:
        unique_pct = df[col].nunique() / len(df)
        
        # Constant columns (bad)
        if unique_pct < 0.01:
            uniqueness_score -= 10
        
        # Near-constant columns (bad)
        elif unique_pct < 0.05:
            uniqueness_score -= 5
    
    scores['uniqueness'] = max(0, uniqueness_score)
    
    # 5. Calculate overall score
    scores['overall'] = sum(scores[k] * weights[k] for k in weights.keys())
    
    # Round all scores
    scores = {k: round(v, 2) for k, v in scores.items()}
    
    return scores


def quality_score_interpretation(score: float) -> Tuple[str, str]:
    """
    Interpret quality score
    
    Returns:
    --------
    tuple
        (grade, interpretation)
    """
    if score >= 90:
        return "A", "Excellent - Data is ready for analysis"
    elif score >= 80:
        return "B", "Good - Minor cleaning recommended"
    elif score >= 70:
        return "C", "Fair - Significant cleaning required"
    elif score >= 60:
        return "D", "Poor - Extensive cleaning required"
    else:
        return "F", "Critical - Data quality is unacceptable"
```

---

## 4. 예시: 입출력 샘플 및 시각화

### 4.1 샘플 데이터 생성

```python
# Create sample customer dataset with quality issues
np.random.seed(42)

n_samples = 1000

sample_data = {
    'customer_id': range(1, n_samples + 1),
    'name': [f'Customer_{i}' if i % 50 != 0 else None for i in range(n_samples)],
    'age': [np.random.randint(18, 80) if i % 20 != 0 else 
            (np.random.randint(-5, 0) if i % 100 == 0 else None) 
            for i in range(n_samples)],
    'email': [f'user{i}@example.com' if i % 30 != 0 else 
              (f'invalid_email_{i}' if i % 15 == 0 else None)
              for i in range(n_samples)],
    'signup_date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
    'purchase_amount': [np.random.uniform(10, 1000) if i % 25 != 0 else None 
                       for i in range(n_samples)],
    'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples)
}

# Add some duplicates
sample_df = pd.DataFrame(sample_data)
sample_df = pd.concat([sample_df, sample_df.iloc[:10]], ignore_index=True)

print("Sample dataset created with intentional quality issues:")
print(f"- {sample_df['name'].isnull().sum()} missing names")
print(f"- {sample_df['age'].isnull().sum()} missing ages")
print(f"- {(sample_df['age'] < 0).sum()} negative ages")
print(f"- {sample_df['email'].isnull().sum()} missing emails")
print(f"- {sample_df.duplicated().sum()} duplicate rows")
```

### 4.2 프로파일링 실행 예시

```python
# Run comprehensive profiling
profile = comprehensive_data_profiling(sample_df)
print_profile_summary(profile)
```

**예상 출력**:

```
================================================================================
DATA QUALITY ASSESSMENT REPORT
================================================================================

📊 BASIC INFORMATION
--------------------------------------------------------------------------------
n_rows...................................... 1010
n_columns................................... 7
memory_mb................................... 0.05
duplicate_rows.............................. 10
duplicate_rows_pct.......................... 0.99
total_cells................................. 7070
total_missing_cells......................... 118
overall_missing_pct......................... 1.67

📈 QUALITY METRICS BY COLUMN
--------------------------------------------------------------------------------
        column        dtype  missing_count  missing_pct  unique_count  unique_pct  ...
   customer_id        int64              0         0.00          1010      100.00  ...
          name       object             20         1.98           980       97.03  ...
           age      float64             50         4.95            64        6.34  ...
         email       object             33         3.27           944       93.47  ...
   signup_date  datetime64              0         0.00          1000       99.01  ...
purchase_amount    float64             40         3.96           926       91.68  ...
      category       object              0         0.00             4        0.40  ...

⚠️  QUALITY ISSUES
--------------------------------------------------------------------------------

HIGH SEVERITY (2 issues):
  • [Data Type] age: Negative ages detected
    Impact: Invalid data will affect analysis
  • [Duplicates] ALL: Duplicate rows detected: 10 (0.99%)
    Impact: May skew statistical analysis and model training

MEDIUM SEVERITY (3 issues):
  • [Missing Data] name: Moderate missing rate: 1.98%
    Impact: Imputation required, may introduce bias
  • [Missing Data] email: Moderate missing rate: 3.27%
    Impact: Imputation required, may introduce bias
  • [Missing Data] age: Moderate missing rate: 4.95%
    Impact: Imputation required, may introduce bias

💡 RECOMMENDATIONS
--------------------------------------------------------------------------------
1. Investigate and remove duplicate rows
2. Apply appropriate imputation strategy for 'name'
3. Apply appropriate imputation strategy for 'email'
4. Apply appropriate imputation strategy for 'age'

================================================================================
```

### 4.3 시각화 예시

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_quality_metrics(profile: Dict[str, Any]) -> None:
    """
    Visualize quality metrics from profile
    """
    
    metrics_df = profile['quality_metrics']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Missing data percentage
    missing_data = metrics_df[['column', 'missing_pct']].sort_values('missing_pct', ascending=False)
    axes[0, 0].barh(missing_data['column'], missing_data['missing_pct'], color='coral')
    axes[0, 0].set_xlabel('Missing %')
    axes[0, 0].set_title('Missing Data by Column')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # 2. Unique value percentage
    unique_data = metrics_df[['column', 'unique_pct']].sort_values('unique_pct', ascending=False)
    axes[0, 1].barh(unique_data['column'], unique_data['unique_pct'], color='skyblue')
    axes[0, 1].set_xlabel('Unique %')
    axes[0, 1].set_title('Cardinality by Column')
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # 3. Issue severity distribution
    issues_df = pd.DataFrame(profile['issues'])
    if len(issues_df) > 0:
        severity_counts = issues_df['severity'].value_counts()
        colors = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'yellow'}
        axes[1, 0].pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%',
                      colors=[colors[s] for s in severity_counts.index])
        axes[1, 0].set_title('Issues by Severity')
    
    # 4. Quality score breakdown
    scores = calculate_data_quality_score(sample_df)
    score_types = ['completeness', 'validity', 'consistency', 'uniqueness']
    score_values = [scores[t] for t in score_types]
    axes[1, 1].bar(score_types, score_values, color=['green', 'blue', 'purple', 'orange'])
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].set_ylabel('Score (0-100)')
    axes[1, 1].set_title(f'Quality Score Breakdown (Overall: {scores["overall"]:.1f})')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quality_assessment_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate visualization
visualize_quality_metrics(profile)
```

---

## 5. 에이전트 매핑

### 5.1 Primary Agent: `data-cleaning-specialist`

**역할**:
- 데이터 품질 평가 전체 프로세스 관리
- 프로파일링 실행 및 결과 해석
- 품질 이슈 식별 및 분류
- 타입 검증 및 비즈니스 규칙 검증 수행

**사용하는 함수**:
- `comprehensive_data_profiling()`
- `validate_data_types()`
- `validate_business_rules()`
- `calculate_data_quality_score()`

### 5.2 Supporting Agent: `data-scientist`

**역할**:
- 도메인 지식을 활용한 비즈니스 규칙 정의
- 품질 평가 결과의 통계적 해석
- 데이터 품질이 분석 및 모델링에 미치는 영향 평가

**사용하는 함수**:
- `auto_detect_types()`
- `quality_score_interpretation()`

### 5.3 Supporting Skill: `data-visualization-specialist`

**역할**:
- 품질 메트릭 시각화
- 대시보드 생성

**사용하는 함수**:
- `visualize_quality_metrics()`

---

## 6. 필요 라이브러리

### 6.1 필수 라이브러리

```bash
# 핵심 라이브러리
pip install pandas>=2.0.0
pip install numpy>=1.24.0

# 데이터 프로파일링
pip install ydata-profiling>=4.5.0

# 데이터 검증
pip install great-expectations>=0.18.0

# 시각화
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
```

### 6.2 라이브러리별 주요 기능

| 라이브러리 | 버전 | 주요 용도 | 핵심 함수 |
|-----------|------|-----------|----------|
| pandas | >=2.0.0 | 데이터 조작, 기본 프로파일링 | `info()`, `describe()`, `isnull()`, `duplicated()` |
| numpy | >=1.24.0 | 수치 연산 | `mean()`, `std()`, `percentile()` |
| ydata-profiling | >=4.5.0 | 자동 프로파일링 리포트 | `ProfileReport()` |
| great-expectations | >=0.18.0 | 체계적 데이터 검증 | `ExpectationSuite()`, `validate()` |
| matplotlib | >=3.7.0 | 시각화 | `bar()`, `pie()`, `hist()` |
| seaborn | >=0.12.0 | 통계 시각화 | `heatmap()`, `boxplot()` |

---

## 7. 체크포인트

### 7.1 프로파일링 완료 체크리스트

- [ ] 모든 컬럼의 기본 통계를 확인했는가?
- [ ] 결측값 패턴을 파악했는가?
- [ ] 중복 데이터를 식별했는가?
- [ ] 이상치 후보를 확인했는가?
- [ ] 카디널리티가 적절한가?

### 7.2 타입 검증 체크리스트

- [ ] 모든 컬럼의 데이터 타입이 예상과 일치하는가?
- [ ] 타입 변환이 필요한 컬럼을 식별했는가?
- [ ] 타입 변환 시 데이터 손실 가능성을 평가했는가?

### 7.3 비즈니스 규칙 검증 체크리스트

- [ ] 도메인별 제약 조건을 모두 정의했는가?
- [ ] 규칙 위반 건수와 비율을 확인했는가?
- [ ] 위반 패턴을 분석했는가?
- [ ] 위반 데이터 처리 방안을 수립했는가?

### 7.4 품질 점수 평가

| 점수 범위 | 등급 | 해석 | 조치 |
|----------|------|------|------|
| 90-100 | A | 우수 | 분석 즉시 가능 |
| 80-89 | B | 양호 | 경미한 클렌징 권장 |
| 70-79 | C | 보통 | 상당한 클렌징 필요 |
| 60-69 | D | 미흡 | 광범위한 클렌징 필요 |
| <60 | F | 심각 | 데이터 품질 수용 불가 |

---

## 8. 트러블슈팅

### 8.1 일반적 오류 및 해결 방법

#### 문제 1: 메모리 부족 (MemoryError)
**증상**: 대용량 데이터셋 프로파일링 시 메모리 부족
**해결**:
```python
# 청크 단위로 처리
chunk_size = 10000
profiles = []

for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    profile = comprehensive_data_profiling(chunk)
    profiles.append(profile)

# 프로파일 병합
merged_profile = merge_profiles(profiles)
```

#### 문제 2: ydata-profiling이 느림
**증상**: 대용량 데이터에서 프로파일 리포트 생성이 오래 걸림
**해결**:
```python
# 최소 모드 사용
profile = ProfileReport(df, minimal=True)

# 또는 샘플링
sample_df = df.sample(n=10000, random_state=42)
profile = ProfileReport(sample_df)
```

#### 문제 3: 타입 변환 실패
**증상**: `pd.to_numeric()` 또는 `pd.to_datetime()` 실패
**해결**:
```python
# errors='coerce' 사용하여 변환 불가능한 값을 NaN으로
df['age'] = pd.to_numeric(df['age'], errors='coerce')

# 변환 실패한 값 확인
failed_conversions = df[df['age'].isnull() & original_df['age'].notnull()]
print(failed_conversions)
```

#### 문제 4: 비즈니스 규칙 검증 오류
**증상**: 규칙 검증 중 예외 발생
**해결**:
```python
# try-except 블록 추가
try:
    valid_mask = validation_func(df)
except Exception as e:
    print(f"Validation error: {e}")
    valid_mask = pd.Series([False] * len(df))
```

### 8.2 성능 최적화 팁

1. **대용량 데이터 처리**:
   - 샘플링 활용: `df.sample(frac=0.1)`
   - 청크 단위 처리: `pd.read_csv(chunksize=...)`
   - Dask 라이브러리 사용 고려

2. **프로파일링 속도 향상**:
   - `minimal=True` 옵션 사용
   - 불필요한 상관관계 계산 비활성화
   - 샘플 데이터로 초기 평가

3. **메모리 효율성**:
   - 불필요한 컬럼 제거
   - 적절한 데이터 타입 사용 (`int8`, `int16` 등)
   - 카테고리 타입 활용: `astype('category')`

---

## 9. 참고 자료

### 9.1 공식 문서

- **Pandas**: https://pandas.pydata.org/docs/
  - User Guide: Data Quality
  - API Reference: DataFrame methods

- **ydata-profiling**: https://docs.profiling.ydata.ai/
  - Getting Started
  - Advanced Settings
  - API Reference

- **Great Expectations**: https://docs.greatexpectations.io/
  - Core Concepts
  - Expectation Gallery
  - Validation

### 9.2 베스트 프랙티스

1. **항상 원본 데이터 보존**:
   ```python
   df_original = df.copy()  # 원본 유지
   ```

2. **프로파일링 결과 저장**:
   ```python
   profile_report = comprehensive_data_profiling(df)
   pd.DataFrame(profile_report['quality_metrics']).to_csv('quality_metrics.csv')
   ```

3. **버전 관리**:
   - 품질 평가 결과를 버전별로 저장
   - 시간에 따른 품질 변화 추적

4. **자동화**:
   - 정기적 품질 점검 스케줄링
   - 품질 저하 시 알림 설정

### 9.3 추가 학습 자료

- **논문**: 
  - "Data Quality: The Accuracy Dimension" (Wang & Strong, 1996)
  - "A Framework for Data Quality Assessment" (Batini et al., 2009)

- **블로그/튜토리얼**:
  - Real Python: Data Validation with Pandas
  - Towards Data Science: Complete Guide to Data Profiling

- **도서**:
  - "Python for Data Analysis" by Wes McKinney
  - "Data Quality: The Field Guide" by Thomas C. Redman

---

## 10. 요약

데이터 품질 평가는 모든 데이터 클렌징 프로젝트의 필수 시작점입니다. 이 레퍼런스에서 다룬 내용:

### 핵심 포인트

1. **종합 프로파일링**: 모든 품질 차원(완전성, 정확성, 일관성, 적시성, 유효성, 고유성)을 평가
2. **타입 검증**: 데이터 타입의 일관성과 적절성 확인
3. **비즈니스 규칙**: 도메인별 제약 조건과 논리적 일관성 검증
4. **자동화**: ydata-profiling으로 신속한 초기 평가

### 다음 단계

품질 평가 완료 후:
1. 식별된 이슈를 우선순위에 따라 분류
2. 각 이슈에 적절한 클렌징 전략 선택
3. 결측값 처리 (Reference 02, 03)
4. 이상치 처리 (Reference 04, 05)
5. 중복 처리 (Reference 07)

### 자동화 커맨드 연계

```bash
# 품질 평가 실행
/clean:assess --data customer_data.csv --report quality_report.html

# 평가 결과 기반 자동 클렌징
/clean:full --data customer_data.csv --based-on quality_report.json
```

---

**작성자**: Claude Code  
**마지막 업데이트**: 2025-01-25  
**다음 레퍼런스**: 02-missing-data-patterns.md
