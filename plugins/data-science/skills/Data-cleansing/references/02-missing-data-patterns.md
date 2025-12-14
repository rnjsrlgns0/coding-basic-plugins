# Missing Data Patterns (결측값 패턴 분석)

**생성일**: 2025-01-25  
**버전**: 1.0  
**담당 에이전트**: `data-cleaning-specialist`, `data-visualization-specialist`

---

## 1. 개요

### 1.1 목적

결측값 패턴 분석은 데이터 클렌징에서 가장 중요한 단계 중 하나입니다. 단순히 결측값이 "얼마나 있는가"를 넘어서 "왜 결측되었는가"와 "어떤 패턴으로 결측되었는가"를 이해하는 것이 핵심입니다. 이 레퍼런스는 다음 목표를 달성합니다:

- **결측 메커니즘 식별**: MCAR, MAR, MNAR 구분을 통한 적절한 처리 방법 선택
- **패턴 시각화**: 결측값의 분포와 관계를 직관적으로 파악
- **공동 결측 분석**: 여러 변수가 함께 결측되는 패턴 탐지
- **대체 전략 수립**: 결측 메커니즘에 따른 최적 imputation 방법 결정

### 1.2 적용 시기

결측값 패턴 분석은 다음 상황에서 필수적으로 수행해야 합니다:

1. **데이터 품질 평가 직후**: 결측값이 5% 이상 발견된 경우
2. **imputation 전**: 어떤 대체 방법을 사용할지 결정하기 전
3. **모델 성능 저하 시**: 결측값 처리가 부적절했을 가능성 조사
4. **새로운 데이터 소스**: 결측값 생성 메커니즘이 불명확한 경우
5. **설문조사 데이터**: 응답 누락 패턴이 중요한 의미를 가지는 경우

### 1.3 왜 중요한가?

결측값 패턴을 제대로 이해하지 못하면:
- **부적절한 imputation**: 잘못된 방법으로 결측값을 채워 편향(bias) 발생
- **정보 손실**: 결측값에 담긴 중요한 정보(예: 응답 거부) 무시
- **모델 성능 저하**: ML 모델이 잘못된 가정에 기반하여 학습
- **잘못된 결론**: 통계적 추론이 왜곡됨

---

## 2. 이론적 배경

### 2.1 결측 메커니즘 (Missing Data Mechanisms)

결측값은 왜 발생했는가에 따라 세 가지 메커니즘으로 분류됩니다 (Rubin, 1976).

#### 2.1.1 MCAR (Missing Completely At Random)

**정의**: 결측값이 완전히 무작위로 발생. 결측 여부가 관측된 데이터나 결측된 데이터 자체와 무관함.

**수학적 표현**:
```
P(Missing | X_observed, X_missing) = P(Missing)
```

**예시**:
- 연구자가 실수로 데이터 일부를 입력하지 않음
- 센서가 무작위로 고장남
- 설문지 일부가 우연히 분실됨

**특징**:
- 가장 이상적인 결측 메커니즘
- 결측된 데이터를 제거해도 편향이 발생하지 않음
- 단순 imputation(평균, 중앙값)도 비교적 안전

**식별 방법**:
- Little's MCAR Test
- 결측 여부와 다른 변수들 간 상관관계 검정

#### 2.1.2 MAR (Missing At Random)

**정의**: 결측 여부가 관측된 데이터와는 관련이 있지만, 결측된 값 자체와는 무관함.

**수학적 표현**:
```
P(Missing | X_observed, X_missing) = P(Missing | X_observed)
```

**예시**:
- 남성이 여성보다 소득을 덜 보고하는 경향 (성별=관측됨, 소득=결측)
- 나이가 많은 사람이 건강 설문에 덜 응답 (나이=관측됨, 건강=결측)
- 고가 제품의 가격 정보가 더 자주 누락 (카테고리=관측됨, 가격=결측)

**특징**:
- 실무에서 가장 흔한 메커니즘
- 관측된 변수를 활용하여 결측값 예측 가능
- Multiple Imputation, ML 기반 imputation 효과적

**식별 방법**:
- 결측 여부를 종속변수로 한 로지스틱 회귀
- 결측 그룹과 비결측 그룹의 관측된 변수 비교

#### 2.1.3 MNAR (Missing Not At Random)

**정의**: 결측 여부가 결측된 값 자체와 관련이 있음.

**수학적 표현**:
```
P(Missing | X_observed, X_missing) ≠ P(Missing | X_observed)
```

**예시**:
- 소득이 매우 높거나 낮은 사람이 소득을 보고하지 않음
- 우울증이 심한 사람이 우울증 설문에 응답하지 않음
- 성적이 나쁜 학생이 시험을 결시함

**특징**:
- 가장 다루기 어려운 메커니즘
- 단순 imputation은 심각한 편향 야기
- 도메인 지식이 필수적
- 민감도 분석(sensitivity analysis) 필요

**식별 방법**:
- 명확한 통계적 검정 없음
- 도메인 지식과 논리적 추론에 의존
- 패턴 분석과 전문가 판단

### 2.2 결측값 패턴 유형

#### 2.2.1 단변량 패턴 (Univariate Pattern)
하나의 변수만 결측값을 가짐.

```
X1  X2  X3  X4
10  5   3   ?
20  7   ?   8
30  ?   5   9
40  9   7   10
```

#### 2.2.2 단조 패턴 (Monotone Pattern)
변수들을 정렬했을 때 결측 패턴이 계단식으로 나타남.

```
X1  X2  X3  X4
10  5   3   2
20  7   5   ?
30  9   ?   ?
40  ?   ?   ?
```

- 종단 연구(longitudinal study)에서 흔함
- 참가자가 중간에 탈락하는 경우

#### 2.2.3 임의 패턴 (Arbitrary Pattern)
불규칙적인 결측 패턴.

```
X1  X2  X3  X4
10  ?   3   2
?   7   5   9
30  9   ?   ?
40  1   7   ?
```

- 가장 복잡하고 흔한 패턴
- 다양한 결측 메커니즘이 혼재

### 2.3 실제 시나리오

#### 시나리오 1: 의료 설문조사
**상황**: 5,000명의 건강 설문 데이터
**결측 패턴**:
- 소득 질문: 15% 결측 (고소득/저소득자가 더 많이 누락 - MNAR)
- 체중: 8% 결측 (여성이 남성보다 더 많이 누락 - MAR)
- 혈압: 3% 결측 (측정 장비 오류 - MCAR)

**발견된 패턴**:
- 소득과 체중이 함께 결측되는 비율이 높음 (공동 결측)
- 나이가 많을수록 전반적인 결측률 증가

**대응 전략**:
- MCAR (혈압): 평균 대체 가능
- MAR (체중): 성별을 고려한 KNN imputation
- MNAR (소득): 별도 '응답거부' 범주 생성 또는 삭제

#### 시나리오 2: 전자상거래 로그
**상황**: 100만 건의 거래 데이터
**결측 패턴**:
- 배송 주소: 0.1% 결측 (시스템 오류 - MCAR)
- 제품 리뷰: 70% 결측 (고객이 작성 안 함 - MNAR)
- 할인 코드: 85% 결측 (대부분 사용 안 함 - MNAR, but 의도적)

**발견된 패턴**:
- 리뷰 결측 여부가 제품 만족도와 관련 (MNAR)
- 할인 코드 결측은 정상적인 상황

**대응 전략**:
- 배송 주소: 다른 주소 정보로 imputation
- 제품 리뷰: 결측을 "리뷰 없음"으로 명시적 처리
- 할인 코드: 0 또는 "미사용"으로 채움

---

## 3. 구현: 상세 Python 코드

### 3.1 결측값 종합 분석

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def analyze_missing_patterns(df: pd.DataFrame, 
                            threshold: float = 0.0) -> Dict:
    """
    Comprehensive missing data pattern analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    threshold : float
        Only analyze columns with missing % >= threshold (default: 0)
        
    Returns:
    --------
    dict
        Complete missing data analysis including statistics, patterns, and mechanisms
        
    Example:
    --------
    >>> df = pd.read_csv('survey_data.csv')
    >>> missing_analysis = analyze_missing_patterns(df)
    >>> print(missing_analysis['summary'])
    """
    
    analysis = {
        'summary': {},
        'column_stats': pd.DataFrame(),
        'patterns': {},
        'correlations': pd.DataFrame(),
        'mechanisms': {}
    }
    
    # ===== 1. 기본 통계 =====
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    
    analysis['summary'] = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'total_cells': total_cells,
        'missing_cells': missing_cells,
        'missing_pct': round(100 * missing_cells / total_cells, 2),
        'columns_with_missing': (df.isnull().sum() > 0).sum(),
        'rows_with_missing': (df.isnull().any(axis=1)).sum(),
        'rows_with_missing_pct': round(100 * (df.isnull().any(axis=1)).sum() / len(df), 2),
        'complete_rows': (~df.isnull().any(axis=1)).sum(),
        'complete_rows_pct': round(100 * (~df.isnull().any(axis=1)).sum() / len(df), 2)
    }
    
    # ===== 2. 컬럼별 결측 통계 =====
    column_stats = []
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = 100 * missing_count / len(df)
        
        if missing_pct >= threshold:
            stats = {
                'column': col,
                'dtype': str(df[col].dtype),
                'missing_count': missing_count,
                'missing_pct': round(missing_pct, 2),
                'present_count': len(df) - missing_count,
                'present_pct': round(100 - missing_pct, 2),
                'first_missing_index': df[df[col].isnull()].index[0] if missing_count > 0 else None,
                'last_missing_index': df[df[col].isnull()].index[-1] if missing_count > 0 else None
            }
            
            column_stats.append(stats)
    
    analysis['column_stats'] = pd.DataFrame(column_stats).sort_values('missing_pct', ascending=False)
    
    # ===== 3. 결측 패턴 분석 =====
    
    # 3.1 패턴 유형 식별
    missing_mask = df.isnull()
    
    # 패턴별 행 개수
    pattern_counts = missing_mask.groupby(list(missing_mask.columns)).size().sort_values(ascending=False)
    
    analysis['patterns']['unique_patterns'] = len(pattern_counts)
    analysis['patterns']['most_common_patterns'] = pattern_counts.head(10).to_dict()
    
    # 3.2 단조 패턴 검사
    analysis['patterns']['is_monotone'] = check_monotone_pattern(df)
    
    # 3.3 공동 결측 분석
    cooccurrence = analyze_cooccurrence(df)
    analysis['patterns']['cooccurrence_pairs'] = cooccurrence
    
    # ===== 4. 결측값 상관관계 =====
    # 결측 여부를 이진 변수로 변환하여 상관관계 계산
    missing_binary = df.isnull().astype(int)
    
    # 상관관계가 있는 컬럼만 (적어도 하나에 결측값이 있는 경우)
    cols_with_missing = missing_binary.columns[missing_binary.sum() > 0].tolist()
    
    if len(cols_with_missing) > 1:
        missing_corr = missing_binary[cols_with_missing].corr()
        
        # 강한 상관관계만 추출 (|r| > 0.3)
        strong_corr = []
        for i in range(len(missing_corr.columns)):
            for j in range(i+1, len(missing_corr.columns)):
                corr_value = missing_corr.iloc[i, j]
                if abs(corr_value) > 0.3:
                    strong_corr.append({
                        'column1': missing_corr.columns[i],
                        'column2': missing_corr.columns[j],
                        'correlation': round(corr_value, 3)
                    })
        
        analysis['correlations'] = pd.DataFrame(strong_corr)
    
    # ===== 5. 결측 메커니즘 추정 =====
    for col in cols_with_missing:
        mechanism = estimate_missing_mechanism(df, col)
        analysis['mechanisms'][col] = mechanism
    
    return analysis


def check_monotone_pattern(df: pd.DataFrame) -> bool:
    """
    Check if missing data follows monotone pattern
    
    Returns:
    --------
    bool
        True if pattern is monotone
    """
    
    missing_mask = df.isnull()
    
    # 결측률로 컬럼 정렬
    sorted_cols = missing_mask.sum().sort_values().index
    sorted_missing = missing_mask[sorted_cols]
    
    # 단조성 검사: 이전 컬럼이 결측이면 이후 컬럼도 결측이어야 함
    for i in range(len(sorted_cols) - 1):
        col1 = sorted_cols[i]
        col2 = sorted_cols[i + 1]
        
        # col1이 결측이지만 col2가 결측이 아닌 경우가 있으면 단조 패턴 아님
        if (sorted_missing[col1] & ~sorted_missing[col2]).any():
            return False
    
    return True


def analyze_cooccurrence(df: pd.DataFrame, 
                         min_count: int = 5) -> List[Dict]:
    """
    Analyze co-occurrence of missing values
    
    Parameters:
    -----------
    min_count : int
        Minimum co-occurrence count to report
        
    Returns:
    --------
    list
        List of column pairs with significant co-occurrence
    """
    
    missing_mask = df.isnull()
    cols_with_missing = missing_mask.columns[missing_mask.sum() > 0].tolist()
    
    cooccurrence_pairs = []
    
    for i in range(len(cols_with_missing)):
        for j in range(i+1, len(cols_with_missing)):
            col1 = cols_with_missing[i]
            col2 = cols_with_missing[j]
            
            # 두 컬럼이 동시에 결측인 행 개수
            both_missing = (missing_mask[col1] & missing_mask[col2]).sum()
            
            if both_missing >= min_count:
                # 기대 빈도 (독립 가정)
                expected = (missing_mask[col1].sum() * missing_mask[col2].sum()) / len(df)
                
                # 관측 빈도 / 기대 빈도
                ratio = both_missing / expected if expected > 0 else 0
                
                cooccurrence_pairs.append({
                    'column1': col1,
                    'column2': col2,
                    'both_missing_count': both_missing,
                    'both_missing_pct': round(100 * both_missing / len(df), 2),
                    'expected_count': round(expected, 1),
                    'obs_exp_ratio': round(ratio, 2),
                    'association': 'Strong' if ratio > 2 else ('Moderate' if ratio > 1.5 else 'Weak')
                })
    
    return sorted(cooccurrence_pairs, key=lambda x: x['obs_exp_ratio'], reverse=True)


def estimate_missing_mechanism(df: pd.DataFrame, 
                               target_col: str,
                               alpha: float = 0.05) -> Dict:
    """
    Estimate missing data mechanism for a column
    
    Parameters:
    -----------
    target_col : str
        Column to analyze
    alpha : float
        Significance level for tests
        
    Returns:
    --------
    dict
        Estimated mechanism and supporting evidence
    """
    
    from scipy import stats
    
    result = {
        'column': target_col,
        'mechanism': 'Unknown',
        'confidence': 'Low',
        'evidence': []
    }
    
    missing_mask = df[target_col].isnull()
    
    if missing_mask.sum() == 0:
        result['mechanism'] = 'No Missing Data'
        return result
    
    # Test 1: Little's MCAR Test (simplified version)
    # 실제로는 완전한 Little's MCAR test가 복잡하므로 간소화된 버전 사용
    
    # Test 2: Compare observed variables between missing and non-missing groups
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != target_col]
    
    significant_differences = 0
    
    for col in numeric_cols[:10]:  # 처음 10개 컬럼만 테스트 (속도)
        group_missing = df[missing_mask][col].dropna()
        group_present = df[~missing_mask][col].dropna()
        
        if len(group_missing) > 0 and len(group_present) > 0:
            # T-test
            stat, p_value = stats.ttest_ind(group_missing, group_present, equal_var=False)
            
            if p_value < alpha:
                significant_differences += 1
                result['evidence'].append(f"{col}: significant difference (p={p_value:.4f})")
    
    # Mechanism estimation based on tests
    if significant_differences == 0:
        result['mechanism'] = 'MCAR'
        result['confidence'] = 'Medium'
        result['evidence'].append(f"No significant differences found in {len(numeric_cols)} variables")
    
    elif significant_differences < len(numeric_cols) * 0.3:
        result['mechanism'] = 'MAR'
        result['confidence'] = 'Medium'
        result['evidence'].append(f"{significant_differences}/{len(numeric_cols)} variables show differences")
    
    else:
        result['mechanism'] = 'MAR or MNAR'
        result['confidence'] = 'Low'
        result['evidence'].append(f"Many variables ({significant_differences}) show differences")
        result['evidence'].append("Domain knowledge required to distinguish MAR from MNAR")
    
    return result
```

### 3.2 결측값 시각화 (missingno 활용)

```python
import missingno as msno

def visualize_missing_bar(df: pd.DataFrame, 
                         figsize: Tuple[int, int] = (12, 6),
                         save_path: str = None) -> None:
    """
    Bar chart showing missing data counts
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Example:
    --------
    >>> visualize_missing_bar(df, save_path='missing_bar.png')
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    msno.bar(df, ax=ax, color='steelblue', fontsize=10)
    
    plt.title('Missing Data Count by Column', fontsize=14, fontweight='bold')
    plt.ylabel('Non-Missing Count', fontsize=12)
    plt.xlabel('Columns', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Bar chart saved to {save_path}")
    
    plt.show()


def visualize_missing_matrix(df: pd.DataFrame,
                             figsize: Tuple[int, int] = (12, 8),
                             sample: int = None,
                             save_path: str = None) -> None:
    """
    Matrix visualization of missing data patterns
    
    Shows data completeness pattern - useful for identifying:
    - Sequential patterns
    - Clusters of missing data
    - Relationships between columns
    
    Parameters:
    -----------
    sample : int, optional
        Number of rows to sample (for large datasets)
    """
    
    df_plot = df.sample(n=sample) if sample and len(df) > sample else df
    
    fig, ax = plt.subplots(figsize=figsize)
    msno.matrix(df_plot, ax=ax, sparkline=True, fontsize=10)
    
    plt.title('Missing Data Matrix (White = Missing)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Matrix chart saved to {save_path}")
    
    plt.show()


def visualize_missing_heatmap(df: pd.DataFrame,
                              figsize: Tuple[int, int] = (10, 8),
                              save_path: str = None) -> None:
    """
    Heatmap showing correlation of missing values between columns
    
    Useful for identifying:
    - Columns that tend to be missing together
    - Strong co-occurrence patterns
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    msno.heatmap(df, ax=ax, fontsize=10, cmap='RdYlGn_r')
    
    plt.title('Missing Data Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Heatmap saved to {save_path}")
    
    plt.show()


def visualize_missing_dendrogram(df: pd.DataFrame,
                                 figsize: Tuple[int, int] = (12, 6),
                                 save_path: str = None) -> None:
    """
    Dendrogram showing hierarchical clustering of missing data patterns
    
    Useful for identifying:
    - Groups of columns with similar missingness patterns
    - Which columns' missingness can be predicted from others
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    msno.dendrogram(df, ax=ax, fontsize=10)
    
    plt.title('Missing Data Dendrogram', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Dendrogram saved to {save_path}")
    
    plt.show()


def visualize_all_missing(df: pd.DataFrame,
                         sample: int = None,
                         save_dir: str = None) -> None:
    """
    Generate all four missing data visualizations
    
    Parameters:
    -----------
    save_dir : str, optional
        Directory to save all figures
    """
    
    print("Generating missing data visualizations...")
    print("=" * 60)
    
    # 1. Bar chart
    print("\n1. Bar Chart (Missing counts)")
    save_path = f"{save_dir}/missing_bar.png" if save_dir else None
    visualize_missing_bar(df, save_path=save_path)
    
    # 2. Matrix
    print("\n2. Matrix (Missingness pattern)")
    save_path = f"{save_dir}/missing_matrix.png" if save_dir else None
    visualize_missing_matrix(df, sample=sample, save_path=save_path)
    
    # 3. Heatmap
    print("\n3. Heatmap (Missing correlations)")
    save_path = f"{save_dir}/missing_heatmap.png" if save_dir else None
    visualize_missing_heatmap(df, save_path=save_path)
    
    # 4. Dendrogram
    print("\n4. Dendrogram (Hierarchical clustering)")
    save_path = f"{save_dir}/missing_dendrogram.png" if save_dir else None
    visualize_missing_dendrogram(df, save_path=save_path)
    
    print("\n" + "=" * 60)
    print("✓ All visualizations generated!")
```

### 3.3 Little's MCAR Test

```python
def littles_mcar_test(df: pd.DataFrame, 
                     alpha: float = 0.05) -> Dict:
    """
    Little's MCAR (Missing Completely At Random) Test
    
    Null Hypothesis: Data is MCAR
    If p-value < alpha: Reject null (data is NOT MCAR, likely MAR or MNAR)
    If p-value >= alpha: Fail to reject (data may be MCAR)
    
    Note: This is a simplified implementation. For production use,
    consider using specialized packages like 'statsmodels' or 'fancyimpute'.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (numeric columns only)
    alpha : float
        Significance level
        
    Returns:
    --------
    dict
        Test results including chi-square statistic and p-value
        
    Example:
    --------
    >>> result = littles_mcar_test(df)
    >>> print(f"MCAR Test p-value: {result['p_value']:.4f}")
    >>> print(f"Conclusion: {result['conclusion']}")
    """
    
    from scipy import stats
    
    # Select only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    
    if df_numeric.empty:
        return {'error': 'No numeric columns found'}
    
    # Remove columns with no missing data
    cols_with_missing = df_numeric.columns[df_numeric.isnull().any()].tolist()
    
    if not cols_with_missing:
        return {
            'test': "Little's MCAR Test",
            'conclusion': 'No missing data to test',
            'p_value': 1.0
        }
    
    df_test = df_numeric[cols_with_missing]
    
    # Create missing indicator matrix
    missing_patterns = df_test.isnull().astype(int)
    unique_patterns = missing_patterns.drop_duplicates()
    
    # Calculate chi-square statistic (simplified version)
    chi_square = 0
    df_freedom = 0
    
    for _, pattern in unique_patterns.iterrows():
        # Get rows matching this pattern
        mask = (missing_patterns == pattern).all(axis=1)
        pattern_data = df_test[mask]
        
        # Compare means of observed values
        for col in cols_with_missing:
            if not pattern[col]:  # If this column is observed in this pattern
                observed_mean = pattern_data[col].mean()
                overall_mean = df_test[col].mean()
                observed_var = pattern_data[col].var()
                
                if observed_var > 0:
                    chi_square += ((observed_mean - overall_mean) ** 2) / observed_var
                    df_freedom += 1
    
    # Calculate p-value
    p_value = 1 - stats.chi2.cdf(chi_square, df_freedom) if df_freedom > 0 else 1.0
    
    # Interpretation
    if p_value < alpha:
        conclusion = "Reject MCAR: Data is likely MAR or MNAR"
        recommendation = "Use advanced imputation methods (KNN, MICE)"
    else:
        conclusion = "Fail to reject MCAR: Data may be MCAR"
        recommendation = "Simple imputation (mean, median) may be acceptable"
    
    return {
        'test': "Little's MCAR Test (Simplified)",
        'chi_square': round(chi_square, 4),
        'degrees_of_freedom': df_freedom,
        'p_value': round(p_value, 4),
        'alpha': alpha,
        'conclusion': conclusion,
        'recommendation': recommendation
    }
```

### 3.4 결측값 간 상관관계 분석

```python
def find_missing_correlations(df: pd.DataFrame,
                              threshold: float = 0.3,
                              method: str = 'pearson') -> pd.DataFrame:
    """
    Find correlations between missing indicators of different columns
    
    Parameters:
    -----------
    threshold : float
        Minimum absolute correlation to report
    method : str
        'pearson', 'spearman', or 'kendall'
        
    Returns:
    --------
    pd.DataFrame
        Pairs of columns with correlated missingness
        
    Example:
    --------
    >>> corr_pairs = find_missing_correlations(df, threshold=0.5)
    >>> print(corr_pairs)
    """
    
    # Convert missing to binary indicators
    missing_indicators = df.isnull().astype(int)
    
    # Calculate correlation matrix
    corr_matrix = missing_indicators.corr(method=method)
    
    # Extract pairs with correlation above threshold
    corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            
            if abs(corr_value) >= threshold:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                
                # Additional statistics
                both_missing = (missing_indicators[col1] & missing_indicators[col2]).sum()
                either_missing = (missing_indicators[col1] | missing_indicators[col2]).sum()
                
                corr_pairs.append({
                    'column1': col1,
                    'column2': col2,
                    'correlation': round(corr_value, 3),
                    'both_missing': both_missing,
                    'either_missing': either_missing,
                    'jaccard_index': round(both_missing / either_missing if either_missing > 0 else 0, 3)
                })
    
    result_df = pd.DataFrame(corr_pairs).sort_values('correlation', 
                                                     key=abs, 
                                                     ascending=False)
    
    return result_df


def visualize_missing_correlation_network(df: pd.DataFrame,
                                         threshold: float = 0.5,
                                         figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Visualize missing data correlations as a network graph
    
    Requires: networkx, matplotlib
    """
    
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX not installed. Install with: pip install networkx")
        return
    
    # Get correlated pairs
    corr_pairs = find_missing_correlations(df, threshold=threshold)
    
    if len(corr_pairs) == 0:
        print(f"No correlations found above threshold {threshold}")
        return
    
    # Create graph
    G = nx.Graph()
    
    for _, row in corr_pairs.iterrows():
        G.add_edge(row['column1'], row['column2'], 
                  weight=abs(row['correlation']))
    
    # Draw graph
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=3000, alpha=0.9)
    
    # Draw edges (thickness based on correlation strength)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights], 
                          alpha=0.6, edge_color='gray')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Draw edge labels (correlations)
    edge_labels = {(row['column1'], row['column2']): f"{row['correlation']:.2f}" 
                   for _, row in corr_pairs.iterrows()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    plt.title(f'Missing Data Correlation Network (threshold={threshold})', 
             fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
```

---

## 4. 예시: 입출력 샘플 및 시각화

### 4.1 샘플 데이터 생성

```python
# Create sample dataset with intentional missing patterns
np.random.seed(42)

n_samples = 1000

# Create base data
data = {
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.uniform(20000, 150000, n_samples),
    'education_years': np.random.randint(8, 20, n_samples),
    'health_score': np.random.uniform(0, 100, n_samples),
    'exercise_hours': np.random.uniform(0, 20, n_samples),
    'bmi': np.random.uniform(18, 35, n_samples)
}

sample_df = pd.DataFrame(data)

# Introduce missing patterns

# 1. MCAR: Random 5% missing in age
mcar_mask = np.random.random(n_samples) < 0.05
sample_df.loc[mcar_mask, 'age'] = np.nan

# 2. MAR: Income missing more for older people
mar_mask = (sample_df['age'] > 60) & (np.random.random(n_samples) < 0.3)
sample_df.loc[mar_mask, 'income'] = np.nan

# 3. MNAR: Low health scores missing more often
mnar_mask = (sample_df['health_score'] < 30) & (np.random.random(n_samples) < 0.4)
sample_df.loc[mnar_mask, 'health_score'] = np.nan

# 4. Co-occurrence: Exercise and BMI missing together
cooccur_mask = np.random.random(n_samples) < 0.1
sample_df.loc[cooccur_mask, 'exercise_hours'] = np.nan
sample_df.loc[cooccur_mask, 'bmi'] = np.nan

print("Sample dataset created with missing data:")
print(sample_df.isnull().sum())
print(f"\nTotal missing cells: {sample_df.isnull().sum().sum()}")
```

### 4.2 종합 분석 실행

```python
# Run comprehensive analysis
analysis = analyze_missing_patterns(sample_df)

# Print summary
print("=" * 80)
print("MISSING DATA PATTERN ANALYSIS")
print("=" * 80)

print("\n📊 SUMMARY STATISTICS")
print("-" * 80)
for key, value in analysis['summary'].items():
    print(f"{key:.<40} {value}")

print("\n\n📈 COLUMN-LEVEL STATISTICS")
print("-" * 80)
print(analysis['column_stats'].to_string(index=False))

print("\n\n🔗 CO-OCCURRENCE PATTERNS")
print("-" * 80)
if analysis['patterns']['cooccurrence_pairs']:
    for pair in analysis['patterns']['cooccurrence_pairs'][:5]:
        print(f"{pair['column1']} ↔ {pair['column2']}: "
              f"{pair['both_missing_count']} rows ({pair['both_missing_pct']}%), "
              f"Association: {pair['association']}")
else:
    print("No significant co-occurrence found")

print("\n\n🔍 MISSING MECHANISMS (Estimated)")
print("-" * 80)
for col, mech in analysis['mechanisms'].items():
    print(f"\n{col}:")
    print(f"  Mechanism: {mech['mechanism']}")
    print(f"  Confidence: {mech['confidence']}")
    if mech['evidence']:
        print(f"  Evidence:")
        for ev in mech['evidence'][:3]:
            print(f"    - {ev}")

print("\n\n📊 MISSING CORRELATIONS")
print("-" * 80)
if not analysis['correlations'].empty:
    print(analysis['correlations'].to_string(index=False))
else:
    print("No strong correlations found")
```

### 4.3 시각화 생성

```python
# Generate all visualizations
print("\nGenerating visualizations...")
visualize_all_missing(sample_df, sample=500, save_dir='./missing_analysis')

# Network visualization
visualize_missing_correlation_network(sample_df, threshold=0.3)

# Little's MCAR Test
mcar_result = littles_mcar_test(sample_df)
print("\n" + "=" * 80)
print("LITTLE'S MCAR TEST")
print("=" * 80)
for key, value in mcar_result.items():
    print(f"{key}: {value}")
```

---

## 5. 에이전트 매핑

### 5.1 Primary Agent: `data-cleaning-specialist`

**역할**:
- 결측값 패턴 분석 전체 프로세스 주도
- 결측 메커니즘 식별 및 해석
- imputation 전략 수립

**사용 함수**:
- `analyze_missing_patterns()`
- `estimate_missing_mechanism()`
- `littles_mcar_test()`
- `find_missing_correlations()`

### 5.2 Supporting Agent: `data-visualization-specialist`

**역할**:
- 결측값 시각화 생성
- 패턴 발견을 위한 그래프 작성

**사용 함수**:
- `visualize_missing_bar()`
- `visualize_missing_matrix()`
- `visualize_missing_heatmap()`
- `visualize_missing_dendrogram()`
- `visualize_missing_correlation_network()`

### 5.3 Supporting Agent: `data-scientist`

**역할**:
- 통계적 검정 해석
- 도메인 지식을 활용한 메커니즘 판단

**사용 함수**:
- `littles_mcar_test()`
- `estimate_missing_mechanism()`

---

## 6. 필요 라이브러리

### 6.1 설치 명령

```bash
# 필수 라이브러리
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install scipy>=1.10.0

# 시각화
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install missingno>=0.5.2

# 네트워크 시각화 (선택)
pip install networkx>=3.0
```

### 6.2 라이브러리 상세

| 라이브러리 | 버전 | 용도 | 핵심 기능 |
|-----------|------|------|----------|
| pandas | >=2.0.0 | 데이터 조작 | `isnull()`, `dropna()`, `groupby()` |
| numpy | >=1.24.0 | 수치 연산 | 배열 연산, 통계 |
| scipy | >=1.10.0 | 통계 검정 | `ttest_ind()`, `chi2.cdf()` |
| missingno | >=0.5.2 | 결측값 시각화 | `bar()`, `matrix()`, `heatmap()`, `dendrogram()` |
| matplotlib | >=3.7.0 | 기본 시각화 | 플롯 생성 |
| seaborn | >=0.12.0 | 통계 시각화 | 향상된 스타일 |
| networkx | >=3.0 | 네트워크 그래프 | 상관관계 네트워크 |

---

## 7. 체크포인트

### 7.1 분석 완료 체크리스트

- [ ] 모든 컬럼의 결측률을 확인했는가?
- [ ] 결측 패턴을 시각화했는가? (bar, matrix, heatmap, dendrogram)
- [ ] 공동 결측 패턴을 식별했는가?
- [ ] 결측값 간 상관관계를 분석했는가?
- [ ] 각 컬럼의 결측 메커니즘을 추정했는가?
- [ ] Little's MCAR test를 수행했는가?

### 7.2 메커니즘 판단 가이드

| 메커니즘 | 특징 | 검증 방법 | 권장 대응 |
|---------|------|-----------|----------|
| MCAR | 완전 무작위 | Little's test (p>0.05), 그룹 간 차이 없음 | 단순 imputation 가능 |
| MAR | 관측 변수와 관련 | 그룹 간 유의한 차이, 예측 가능 | KNN, MICE 등 고급 imputation |
| MNAR | 결측값 자체와 관련 | 논리적 추론, 도메인 지식 | 별도 범주, 민감도 분석 |

### 7.3 품질 기준

**우수 (Excellent)**:
- 결측률 < 5%
- MCAR 메커니즘
- 공동 결측 패턴 없음

**양호 (Good)**:
- 결측률 5-15%
- MAR 메커니즘
- 약한 공동 결측 패턴

**미흡 (Poor)**:
- 결측률 > 15%
- MNAR 메커니즘
- 강한 공동 결측 패턴

---

## 8. 트러블슈팅

### 8.1 일반적 문제

#### 문제 1: missingno 시각화가 느림
**증상**: 대용량 데이터에서 시각화 생성이 오래 걸림
**해결**:
```python
# 샘플링 사용
sample_df = df.sample(n=10000, random_state=42)
visualize_missing_matrix(sample_df)
```

#### 문제 2: Little's MCAR test 오류
**증상**: Chi-square 계산 중 오류
**해결**:
```python
# 수치형 컬럼만 선택
df_numeric = df.select_dtypes(include=[np.number])
result = littles_mcar_test(df_numeric)
```

#### 문제 3: 결측 상관관계가 모두 NaN
**증상**: 결측값이 너무 적거나 많음
**해결**:
```python
# 결측률이 5-95% 사이인 컬럼만 분석
cols_to_analyze = [col for col in df.columns 
                   if 5 < df[col].isnull().mean()*100 < 95]
result = find_missing_correlations(df[cols_to_analyze])
```

### 8.2 메커니즘 판단 시 주의사항

1. **MCAR과 MAR 구분의 어려움**:
   - 통계적 검정만으로는 불충분
   - 도메인 지식 필수
   - 보수적으로 접근 (의심스러우면 MAR로 가정)

2. **MNAR 식별의 한계**:
   - 결측된 값 자체를 관측할 수 없으므로 명확한 검증 불가
   - 논리적 추론과 전문가 판단 필요
   - 민감도 분석으로 영향 평가

3. **혼합 메커니즘**:
   - 실제 데이터는 종종 여러 메커니즘이 혼재
   - 컬럼별로 다른 메커니즘 가능
   - 가장 보수적인 가정 채택

---

## 9. 참고 자료

### 9.1 공식 문서

- **missingno**: https://github.com/ResidentMario/missingno
  - Visualization gallery
  - API reference

- **Pandas Missing Data**: https://pandas.pydata.org/docs/user_guide/missing_data.html
  - Working with missing data
  - Best practices

- **SciPy Stats**: https://docs.scipy.org/doc/scipy/reference/stats.html
  - Statistical tests
  - Distributions

### 9.2 학술 자료

- **Rubin, D. B. (1976)**. "Inference and missing data." Biometrika, 63(3), 581-592.
  - 결측 메커니즘의 고전적 정의

- **Little, R. J. (1988)**. "A test of missing completely at random for multivariate data with missing values." Journal of the American Statistical Association, 83(404), 1198-1202.
  - Little's MCAR test 원논문

- **Schafer, J. L., & Graham, J. W. (2002)**. "Missing data: our view of the state of the art." Psychological methods, 7(2), 147.
  - 결측값 처리 종합 리뷰

### 9.3 추천 도서

- **"Flexible Imputation of Missing Data"** by Stef van Buuren
  - MICE 알고리즘 상세 설명
  - R 중심이지만 개념은 Python에도 적용 가능

---

## 10. 요약

### 10.1 핵심 포인트

1. **메커니즘 이해**: MCAR, MAR, MNAR의 차이를 이해하고 올바르게 식별
2. **패턴 시각화**: 4가지 시각화(bar, matrix, heatmap, dendrogram)로 종합 파악
3. **공동 결측**: 여러 변수가 함께 결측되는 패턴 탐지
4. **통계적 검정**: Little's MCAR test로 객관적 평가

### 10.2 의사결정 플로우

```
1. 결측값 발견
   ↓
2. 패턴 분석 (시각화 + 통계)
   ↓
3. 메커니즘 추정
   ├─ MCAR → 단순 imputation 가능
   ├─ MAR → 고급 imputation 필요
   └─ MNAR → 도메인 지식 + 민감도 분석
   ↓
4. imputation 전략 선택 (Reference 03)
```

### 10.3 다음 단계

패턴 분석 완료 후:
- **Reference 03**: Imputation Strategies - 적절한 대체 방법 선택 및 적용
- **Reference 02 결과 활용**: 메커니즘에 따른 최적 전략 결정

---

**작성자**: Claude Code  
**마지막 업데이트**: 2025-01-25  
**이전 레퍼런스**: 01-data-quality-assessment.md  
**다음 레퍼런스**: 03-imputation-strategies.md
