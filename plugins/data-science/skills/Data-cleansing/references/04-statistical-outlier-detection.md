# Statistical Outlier Detection (통계적 이상치 탐지)

**생성일**: 2025-01-25  
**버전**: 1.0  
**담당 에이전트**: `data-cleaning-specialist`, `data-scientist`

---

## 1. 개요

### 1.1 목적

통계적 이상치 탐지는 데이터의 정상 범위를 벗어난 값을 식별하는 핵심 프로세스입니다. 이 레퍼런스는 단변량 통계적 방법에 집중하며 다음을 제공합니다:

- **다양한 탐지 방법**: Z-Score, IQR, Modified Z-Score 등 검증된 통계 기법
- **방법 비교**: 각 방법의 장단점과 적용 시나리오
- **시각화**: 이상치를 직관적으로 파악하는 그래프
- **자동화**: 여러 컬럼에 대한 일괄 이상치 탐지

### 1.2 이상치란?

이상치(Outlier)는 데이터의 일반적인 패턴에서 현저하게 벗어난 관측값입니다.

**이상치의 원인**:
1. **데이터 오류**: 입력 실수, 센서 오작동
2. **정상적 극단값**: 실제로 발생한 극단적 사건
3. **실험적 오류**: 측정 오류, 표본 추출 오류
4. **의도적 이상**: 사기, 이상 행동

**이상치의 영향**:
- 평균과 표준편차를 왜곡
- 회귀 분석 결과 왜곡
- ML 모델 성능 저하
- 통계적 검정력 감소

### 1.3 탐지 vs 처리

이 레퍼런스는 **탐지**에 집중합니다. 처리는 Reference 06에서 다룹니다.

---

## 2. 이론적 배경

### 2.1 통계적 방법 비교

| 방법 | 가정 | 강점 | 약점 | 사용 시기 |
|------|------|------|------|----------|
| **Z-Score** | 정규분포 | 간단, 해석 용이 | 정규성 필요, 이상치에 민감 | 대용량, 정규분포 |
| **IQR** | 없음 | Robust, 분포 무관 | 고정 임계값 | 왜곡 분포, 소규모 |
| **Modified Z-Score** | 없음 | Robust, IQR보다 정교 | 계산 복잡 | 이상치 많을 때 |
| **Percentile** | 없음 | 단순, 직관적 | 비율 기반, 상대적 | 고정 비율 제거 |

### 2.2 Z-Score 방법

**공식**:
```
Z = (X - μ) / σ
```
- X: 관측값
- μ: 평균
- σ: 표준편차

**임계값**:
- |Z| > 3: 일반적 기준 (99.7% 신뢰구간 밖)
- |Z| > 2.5: 보수적 기준
- |Z| > 4: 매우 엄격한 기준

**장점**:
- 표준화된 척도
- 해석이 명확 (표준편차 단위)
- 빠른 계산

**단점**:
- 정규분포 가정 필수
- 평균과 표준편차가 이상치에 영향받음

### 2.3 IQR (Interquartile Range) 방법

**공식**:
```
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

**Tukey's Fences**:
- 내부 울타리 (Inner fence): 1.5 × IQR
- 외부 울타리 (Outer fence): 3.0 × IQR (극단 이상치)

**장점**:
- 분포 가정 불필요
- 중앙값 기반이라 robust
- 박스플롯과 자연스럽게 연계

**단점**:
- 고정 비율 (1.5)이 모든 상황에 최적은 아님
- 정규분포에서는 Z-Score보다 덜 정교

### 2.4 Modified Z-Score

**공식**:
```
Modified Z = 0.6745 × (X - Median) / MAD
```
- MAD: Median Absolute Deviation
- MAD = median(|X - median(X)|)
- 0.6745: 정규분포에서 MAD를 표준편차로 변환하는 상수

**임계값**: |Modified Z| > 3.5

**장점**:
- 중앙값과 MAD 사용으로 매우 robust
- 이상치가 많을 때도 안정적
- Z-Score와 유사한 해석

**단점**:
- 계산이 상대적으로 복잡
- MAD가 0일 때 문제 (모든 값이 동일)

### 2.5 실제 시나리오

#### 시나리오 1: 온라인 쇼핑몰 거래액
**데이터**: 일일 거래액 (1,000건)
**분포**: 오른쪽 왜곡 (대부분 소액, 일부 고액)

**탐지 결과**:
- Z-Score: 15개 이상치 (정규분포 가정 위반으로 부정확)
- IQR: 48개 이상치 (고액 거래 포함)
- Modified Z-Score: 23개 이상치 (가장 적절)

**결론**: Modified Z-Score 또는 IQR 사용 권장

#### 시나리오 2: 센서 온도 데이터
**데이터**: 시간당 온도 측정 (10,000건)
**분포**: 정규분포에 가까움

**탐지 결과**:
- Z-Score: 32개 이상치 (센서 오류 잘 탐지)
- IQR: 54개 이상치 (과도 탐지)
- Modified Z-Score: 35개 이상치

**결론**: Z-Score 사용 권장 (정규분포 만족)

---

## 3. 구현: 상세 Python 코드

### 3.1 Z-Score 이상치 탐지

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union
from scipy import stats

def detect_outliers_zscore(df: pd.DataFrame,
                          columns: List[str] = None,
                          threshold: float = 3.0) -> Dict[str, pd.Series]:
    """
    Detect outliers using Z-Score method
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        Columns to analyze (default: all numeric)
    threshold : float
        Z-score threshold (default: 3.0)
        
    Returns:
    --------
    dict
        Dictionary mapping column names to boolean Series (True = outlier)
        
    Example:
    --------
    >>> outliers = detect_outliers_zscore(df, threshold=3.0)
    >>> print(f"Age outliers: {outliers['age'].sum()}")
    >>> df_clean = df[~outliers['age']]
    """
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers = {}
    
    for col in columns:
        if col not in df.columns:
            continue
        
        # Calculate Z-scores
        mean = df[col].mean()
        std = df[col].std()
        
        if std == 0:
            # All values are the same, no outliers
            outliers[col] = pd.Series([False] * len(df), index=df.index)
            continue
        
        z_scores = np.abs((df[col] - mean) / std)
        outliers[col] = z_scores > threshold
    
    return outliers


def zscore_analysis(df: pd.DataFrame, 
                   column: str,
                   threshold: float = 3.0) -> Dict:
    """
    Detailed Z-score analysis for a single column
    
    Returns statistics and visualization data
    """
    
    data = df[column].dropna()
    
    mean = data.mean()
    std = data.std()
    z_scores = np.abs((data - mean) / std)
    
    outlier_mask = z_scores > threshold
    outliers = data[outlier_mask]
    
    analysis = {
        'column': column,
        'method': 'Z-Score',
        'threshold': threshold,
        'n_total': len(data),
        'n_outliers': outlier_mask.sum(),
        'outlier_pct': round(100 * outlier_mask.sum() / len(data), 2),
        'mean': round(mean, 2),
        'std': round(std, 2),
        'min': data.min(),
        'max': data.max(),
        'outlier_indices': data[outlier_mask].index.tolist(),
        'outlier_values': outliers.values.tolist(),
        'max_zscore': round(z_scores.max(), 2),
        'normality_test': stats.shapiro(data.sample(min(5000, len(data))))[1]  # p-value
    }
    
    return analysis
```

### 3.2 IQR 이상치 탐지

```python
def detect_outliers_iqr(df: pd.DataFrame,
                       columns: List[str] = None,
                       multiplier: float = 1.5) -> Dict[str, pd.Series]:
    """
    Detect outliers using IQR (Interquartile Range) method
    
    Parameters:
    -----------
    multiplier : float
        IQR multiplier (default: 1.5 for standard, 3.0 for extreme outliers)
        
    Returns:
    --------
    dict
        Dictionary mapping column names to boolean Series (True = outlier)
        
    Example:
    --------
    >>> outliers = detect_outliers_iqr(df, multiplier=1.5)
    >>> df[outliers['price']]['price']  # View price outliers
    """
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers = {}
    
    for col in columns:
        if col not in df.columns:
            continue
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
    
    return outliers


def iqr_analysis(df: pd.DataFrame,
                column: str,
                multiplier: float = 1.5) -> Dict:
    """
    Detailed IQR analysis for a single column
    """
    
    data = df[column].dropna()
    
    Q1 = data.quantile(0.25)
    Q2 = data.quantile(0.50)  # Median
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    outliers = data[outlier_mask]
    
    # Separate lower and upper outliers
    lower_outliers = data[data < lower_bound]
    upper_outliers = data[data > upper_bound]
    
    analysis = {
        'column': column,
        'method': 'IQR',
        'multiplier': multiplier,
        'n_total': len(data),
        'n_outliers': outlier_mask.sum(),
        'outlier_pct': round(100 * outlier_mask.sum() / len(data), 2),
        'Q1': round(Q1, 2),
        'Q2': round(Q2, 2),
        'Q3': round(Q3, 2),
        'IQR': round(IQR, 2),
        'lower_bound': round(lower_bound, 2),
        'upper_bound': round(upper_bound, 2),
        'n_lower_outliers': len(lower_outliers),
        'n_upper_outliers': len(upper_outliers),
        'outlier_indices': data[outlier_mask].index.tolist(),
        'min': data.min(),
        'max': data.max()
    }
    
    return analysis
```

### 3.3 Modified Z-Score 이상치 탐지

```python
def detect_outliers_modified_zscore(df: pd.DataFrame,
                                   columns: List[str] = None,
                                   threshold: float = 3.5) -> Dict[str, pd.Series]:
    """
    Detect outliers using Modified Z-Score (MAD-based)
    
    More robust than standard Z-score, especially with existing outliers
    
    Parameters:
    -----------
    threshold : float
        Modified Z-score threshold (default: 3.5)
        
    Returns:
    --------
    dict
        Dictionary mapping column names to boolean Series (True = outlier)
        
    Example:
    --------
    >>> outliers = detect_outliers_modified_zscore(df, threshold=3.5)
    """
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers = {}
    
    for col in columns:
        if col not in df.columns:
            continue
        
        # Calculate median and MAD
        median = df[col].median()
        mad = np.median(np.abs(df[col] - median))
        
        if mad == 0:
            # All values are the same, no outliers
            outliers[col] = pd.Series([False] * len(df), index=df.index)
            continue
        
        # Modified Z-score
        # 0.6745 is the 0.75th quartile of the standard normal distribution
        modified_z_scores = 0.6745 * (df[col] - median) / mad
        outliers[col] = np.abs(modified_z_scores) > threshold
    
    return outliers


def modified_zscore_analysis(df: pd.DataFrame,
                            column: str,
                            threshold: float = 3.5) -> Dict:
    """
    Detailed Modified Z-score analysis
    """
    
    data = df[column].dropna()
    
    median = data.median()
    mad = np.median(np.abs(data - median))
    
    if mad == 0:
        return {
            'column': column,
            'method': 'Modified Z-Score',
            'error': 'MAD is zero (all values are identical)'
        }
    
    modified_z_scores = 0.6745 * (data - median) / mad
    outlier_mask = np.abs(modified_z_scores) > threshold
    outliers = data[outlier_mask]
    
    analysis = {
        'column': column,
        'method': 'Modified Z-Score',
        'threshold': threshold,
        'n_total': len(data),
        'n_outliers': outlier_mask.sum(),
        'outlier_pct': round(100 * outlier_mask.sum() / len(data), 2),
        'median': round(median, 2),
        'mad': round(mad, 2),
        'min': data.min(),
        'max': data.max(),
        'outlier_indices': data[outlier_mask].index.tolist(),
        'outlier_values': outliers.values.tolist(),
        'max_modified_zscore': round(np.abs(modified_z_scores).max(), 2)
    }
    
    return analysis
```

### 3.4 통합 이상치 탐지

```python
def detect_outliers_all_methods(df: pd.DataFrame,
                               column: str,
                               z_threshold: float = 3.0,
                               iqr_multiplier: float = 1.5,
                               mz_threshold: float = 3.5) -> pd.DataFrame:
    """
    Apply all three methods and compare results
    
    Returns:
    --------
    pd.DataFrame
        Comparison of all methods with consensus
        
    Example:
    --------
    >>> comparison = detect_outliers_all_methods(df, 'price')
    >>> print(comparison)
    >>> # Get consensus outliers (detected by 2+ methods)
    >>> consensus = comparison[comparison['outlier_count'] >= 2]
    """
    
    # Apply all methods
    zscore_outliers = detect_outliers_zscore(df, [column], z_threshold)[column]
    iqr_outliers = detect_outliers_iqr(df, [column], iqr_multiplier)[column]
    mz_outliers = detect_outliers_modified_zscore(df, [column], mz_threshold)[column]
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'value': df[column],
        'zscore': zscore_outliers,
        'iqr': iqr_outliers,
        'modified_zscore': mz_outliers
    })
    
    # Count methods that flagged as outlier
    comparison['outlier_count'] = (comparison[['zscore', 'iqr', 'modified_zscore']]
                                   .sum(axis=1))
    
    # Consensus: at least 2 methods agree
    comparison['consensus_outlier'] = comparison['outlier_count'] >= 2
    
    # Only return outliers (at least one method)
    result = comparison[comparison['outlier_count'] > 0].copy()
    result = result.sort_values('outlier_count', ascending=False)
    
    return result


def print_outlier_summary(df: pd.DataFrame,
                         columns: List[str] = None) -> pd.DataFrame:
    """
    Generate summary table comparing all methods for multiple columns
    
    Returns:
    --------
    pd.DataFrame
        Summary table with outlier counts for each method
    """
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    summaries = []
    
    for col in columns:
        zscore_out = detect_outliers_zscore(df, [col])[col].sum()
        iqr_out = detect_outliers_iqr(df, [col])[col].sum()
        mz_out = detect_outliers_modified_zscore(df, [col])[col].sum()
        
        summaries.append({
            'column': col,
            'n_total': len(df[col].dropna()),
            'zscore_outliers': zscore_out,
            'zscore_pct': round(100 * zscore_out / len(df[col].dropna()), 2),
            'iqr_outliers': iqr_out,
            'iqr_pct': round(100 * iqr_out / len(df[col].dropna()), 2),
            'mz_outliers': mz_out,
            'mz_pct': round(100 * mz_out / len(df[col].dropna()), 2)
        })
    
    return pd.DataFrame(summaries)
```

### 3.5 시각화

```python
def visualize_outliers(df: pd.DataFrame,
                      column: str,
                      method: str = 'all',
                      figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Visualize outliers using multiple plots
    
    Parameters:
    -----------
    method : str
        'zscore', 'iqr', 'modified_zscore', or 'all'
        
    Creates:
    --------
    - Boxplot
    - Histogram with outlier threshold
    - Scatter plot with outliers highlighted
    """
    
    data = df[column].dropna()
    
    if method == 'all':
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Determine outliers based on method
    if method == 'zscore' or method == 'all':
        outliers_z = detect_outliers_zscore(df, [column])[column]
        
        ax_idx = 0 if method == 'all' else 0
        
        # Boxplot
        axes[ax_idx].boxplot(data, vert=True)
        axes[ax_idx].set_title(f'{column} - Boxplot')
        axes[ax_idx].set_ylabel('Value')
        
        # Histogram
        axes[ax_idx + 1].hist(data, bins=50, alpha=0.7, edgecolor='black')
        mean = data.mean()
        std = data.std()
        axes[ax_idx + 1].axvline(mean, color='red', linestyle='--', label='Mean')
        axes[ax_idx + 1].axvline(mean + 3*std, color='orange', linestyle='--', label='±3σ')
        axes[ax_idx + 1].axvline(mean - 3*std, color='orange', linestyle='--')
        axes[ax_idx + 1].set_title(f'{column} - Z-Score Method')
        axes[ax_idx + 1].set_xlabel('Value')
        axes[ax_idx + 1].set_ylabel('Frequency')
        axes[ax_idx + 1].legend()
        
        # Scatter plot
        axes[ax_idx + 2].scatter(range(len(data)), data, 
                               c=outliers_z[data.index], 
                               cmap='coolwarm', alpha=0.6)
        axes[ax_idx + 2].set_title(f'{column} - Outliers Highlighted (Z-Score)')
        axes[ax_idx + 2].set_xlabel('Index')
        axes[ax_idx + 2].set_ylabel('Value')
    
    if method == 'iqr' or method == 'all':
        outliers_iqr = detect_outliers_iqr(df, [column])[column]
        
        ax_idx = 3 if method == 'all' else 0
        
        # Boxplot
        axes[ax_idx].boxplot(data, vert=True)
        axes[ax_idx].set_title(f'{column} - Boxplot (IQR)')
        
        # Histogram with IQR bounds
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        axes[ax_idx + 1].hist(data, bins=50, alpha=0.7, edgecolor='black')
        axes[ax_idx + 1].axvline(Q2, color='red', linestyle='--', label='Median')
        axes[ax_idx + 1].axvline(lower, color='orange', linestyle='--', label='IQR Bounds')
        axes[ax_idx + 1].axvline(upper, color='orange', linestyle='--')
        axes[ax_idx + 1].set_title(f'{column} - IQR Method')
        axes[ax_idx + 1].set_xlabel('Value')
        axes[ax_idx + 1].legend()
        
        # Scatter plot
        axes[ax_idx + 2].scatter(range(len(data)), data,
                               c=outliers_iqr[data.index],
                               cmap='coolwarm', alpha=0.6)
        axes[ax_idx + 2].set_title(f'{column} - Outliers Highlighted (IQR)')
        axes[ax_idx + 2].set_xlabel('Index')
    
    plt.tight_layout()
    plt.show()


def plot_outlier_comparison(df: pd.DataFrame, column: str) -> None:
    """
    Compare all three methods side by side
    """
    
    comparison = detect_outliers_all_methods(df, column)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Venn diagram style bar chart
    method_counts = {
        'Z-Score Only': (comparison['zscore'] & ~comparison['iqr'] & ~comparison['modified_zscore']).sum(),
        'IQR Only': (~comparison['zscore'] & comparison['iqr'] & ~comparison['modified_zscore']).sum(),
        'Modified Z Only': (~comparison['zscore'] & ~comparison['iqr'] & comparison['modified_zscore']).sum(),
        '2 Methods': (comparison['outlier_count'] == 2).sum(),
        'All 3 Methods': (comparison['outlier_count'] == 3).sum()
    }
    
    axes[0].bar(method_counts.keys(), method_counts.values(), color='skyblue', edgecolor='black')
    axes[0].set_title('Outlier Detection Agreement')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Outlier values by method
    methods = ['zscore', 'iqr', 'modified_zscore']
    outlier_counts = [comparison[m].sum() for m in methods]
    
    axes[1].bar(['Z-Score', 'IQR', 'Modified Z'], outlier_counts, 
               color=['red', 'blue', 'green'], alpha=0.7, edgecolor='black')
    axes[1].set_title('Total Outliers by Method')
    axes[1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()
```

---

## 4. 예시: 입출력 샘플

```python
# Create sample data with outliers
np.random.seed(42)
n = 1000

# Normal data with intentional outliers
normal_data = np.random.normal(100, 15, n)
outliers = np.array([200, 210, -50, -40, 250])
indices = np.random.choice(n, len(outliers), replace=False)
normal_data[indices] = outliers

df_sample = pd.DataFrame({
    'value': normal_data,
    'category': np.random.choice(['A', 'B', 'C'], n)
})

# Apply all methods
print("=" * 80)
print("OUTLIER DETECTION RESULTS")
print("=" * 80)

# Z-Score
z_analysis = zscore_analysis(df_sample, 'value', threshold=3.0)
print(f"\n1. Z-Score Method (threshold=3.0):")
print(f"   Outliers: {z_analysis['n_outliers']} ({z_analysis['outlier_pct']}%)")

# IQR
iqr_analysis_result = iqr_analysis(df_sample, 'value', multiplier=1.5)
print(f"\n2. IQR Method (multiplier=1.5):")
print(f"   Outliers: {iqr_analysis_result['n_outliers']} ({iqr_analysis_result['outlier_pct']}%)")
print(f"   Lower outliers: {iqr_analysis_result['n_lower_outliers']}")
print(f"   Upper outliers: {iqr_analysis_result['n_upper_outliers']}")

# Modified Z-Score
mz_analysis = modified_zscore_analysis(df_sample, 'value', threshold=3.5)
print(f"\n3. Modified Z-Score Method (threshold=3.5):")
print(f"   Outliers: {mz_analysis['n_outliers']} ({mz_analysis['outlier_pct']}%)")

# Comparison
print("\n4. Method Comparison:")
comparison = detect_outliers_all_methods(df_sample, 'value')
print(comparison[['value', 'outlier_count', 'consensus_outlier']].head(10))

# Visualize
visualize_outliers(df_sample, 'value', method='all')
plot_outlier_comparison(df_sample, 'value')
```

---

## 5. 에이전트 매핑

- **Primary**: `data-cleaning-specialist` - 모든 탐지 방법 실행
- **Supporting**: `data-scientist` - 통계적 검정 및 해석
- **Supporting**: `data-visualization-specialist` - 시각화

---

## 6. 필요 라이브러리

```bash
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install scipy>=1.10.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
```

---

## 7. 체크포인트

- [ ] 여러 방법으로 이상치를 탐지했는가?
- [ ] 분포 특성을 고려했는가?
- [ ] 이상치가 오류인지 정당한 극단값인지 조사했는가?
- [ ] 시각화로 확인했는가?
- [ ] 도메인 전문가와 검토했는가?

---

## 8. 트러블슈팅

### 문제 1: Z-Score가 모든 값을 이상치로 탐지
**원인**: 분포가 정규분포가 아님
**해결**: IQR 또는 Modified Z-Score 사용

### 문제 2: IQR이 너무 많은 이상치 탐지
**원인**: multiplier가 너무 작음
**해결**: multiplier를 2.0 또는 3.0으로 증가

### 문제 3: MAD가 0
**원인**: 모든 값이 동일
**해결**: 해당 컬럼 분석 제외

---

## 9. 참고 자료

- **Tukey, J. W. (1977)**. Exploratory Data Analysis. Addison-Wesley.
- **Iglewicz, B., & Hoaglin, D. (1993)**. Volume 16: How to Detect and Handle Outliers.
- **SciPy Stats Documentation**: https://docs.scipy.org/doc/scipy/reference/stats.html

---

## 10. 요약

**핵심 원칙**:
1. 여러 방법 비교
2. 분포 특성 고려
3. 시각화 필수
4. 도메인 지식 활용

**다음 단계**: Reference 06 (Outlier Treatment) - 탐지된 이상치 처리

---

**작성자**: Claude Code  
**마지막 업데이트**: 2025-01-25
