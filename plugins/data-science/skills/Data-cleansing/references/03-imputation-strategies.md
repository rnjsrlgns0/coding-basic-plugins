# Imputation Strategies (결측값 대체 전략)

**생성일**: 2025-01-25  
**버전**: 1.0  
**담당 에이전트**: `data-cleaning-specialist`, `feature-engineering-specialist`

---

## 1. 개요

### 1.1 목적

결측값 대체(imputation)는 데이터 클렌징에서 가장 중요하고 섬세한 작업입니다. 이 레퍼런스는 단순 대체부터 고급 ML 기반 대체까지 모든 방법을 다루며, 다음 목표를 달성합니다:

- **최적 전략 선택**: 데이터 타입, 결측 메커니즘, 결측 비율에 따른 적절한 방법 선택
- **편향 최소화**: 대체로 인한 통계적 편향을 최소화하는 기법 적용
- **검증 체계**: 대체 품질을 평가하고 검증하는 방법론 제공
- **자동화**: 다양한 전략을 자동으로 비교하고 최적을 선택하는 파이프라인 구축

### 1.2 적용 시기

결측값 대체는 다음 상황에서 수행합니다:

1. **패턴 분석 완료 후**: Reference 02에서 결측 메커니즘 식별 후
2. **분석/모델링 전**: 대부분의 알고리즘은 결측값을 처리할 수 없음
3. **결측률이 중간 수준**: 5-40% (너무 낮으면 삭제, 너무 높으면 컬럼 제거 고려)
4. **도메인 지식 확보 후**: 비즈니스 맥락을 이해한 상태에서 수행

### 1.3 Imputation의 딜레마

모든 imputation은 trade-off를 포함합니다:

- **정보 추가 vs 편향 도입**: 새로운 값을 채우면서 원본 분포를 왜곡할 위험
- **단순성 vs 정확성**: 간단한 방법은 빠르지만 부정확, 복잡한 방법은 느리지만 정확
- **완전성 vs 불확실성**: 모든 값을 채우지만 그 값에 대한 확신은 낮음

---

## 2. 이론적 배경

### 2.1 Imputation 방법 분류

```
Imputation Methods
│
├── 1. Deletion (삭제)
│   ├── Listwise deletion (완전 케이스 분석)
│   └── Pairwise deletion (가용 케이스 분석)
│
├── 2. Simple Imputation (단순 대체)
│   ├── Mean/Median/Mode
│   ├── Constant value
│   └── Random sampling
│
├── 3. Time-Series Imputation (시계열 대체)
│   ├── Forward fill (LOCF)
│   ├── Backward fill (NOCB)
│   ├── Linear interpolation
│   └── Spline interpolation
│
├── 4. Model-Based Imputation (모델 기반)
│   ├── Regression imputation
│   ├── K-Nearest Neighbors (KNN)
│   ├── Iterative Imputer (MICE)
│   └── Matrix Factorization
│
└── 5. Advanced Methods (고급 방법)
    ├── Multiple Imputation
    ├── Deep Learning (AutoEncoder, GAN)
    └── Probabilistic methods
```

### 2.2 결측 메커니즘별 권장 전략

| 메커니즘 | 특징 | 권장 Imputation | 피해야 할 방법 |
|---------|------|----------------|--------------|
| **MCAR** | 완전 무작위 | Mean/Median, 삭제도 가능 | - |
| **MAR** | 관측 변수와 관련 | KNN, MICE, Regression | 단순 평균/중앙값 |
| **MNAR** | 결측값 자체와 관련 | 별도 범주, 민감도 분석 | 모든 자동 imputation |

### 2.3 데이터 타입별 전략

#### 2.3.1 수치형 데이터

**연속형**:
- **정규분포**: Mean imputation
- **왜도 높음**: Median imputation
- **다변량 관계**: KNN, MICE
- **시계열**: Interpolation

**이산형**:
- **카운트 데이터**: Median 또는 Mode
- **순서형**: Mode 또는 Ordinal regression

#### 2.3.2 범주형 데이터

- **명목형**: Mode (최빈값)
- **고카디널리티**: "Missing" 범주 추가
- **순서형**: Mode 또는 인접 값

#### 2.3.3 시계열 데이터

- **트렌드 있음**: Linear interpolation
- **계절성 있음**: Seasonal decomposition + interpolation
- **불규칙**: Forward fill 또는 KNN

### 2.4 실제 시나리오

#### 시나리오 1: 전자상거래 고객 데이터
**데이터**: 100만 건, 고객 프로필
**결측 패턴**:
- 나이: 8% 결측 (MCAR) → Median imputation
- 소득: 15% 결측 (MAR, 교육수준과 관련) → KNN imputation
- 선호 카테고리: 20% 결측 (MNAR, 신규 고객) → "Unknown" 범주

**적용 전략**:
```python
imputation_strategy = {
    'age': SimpleImputer(strategy='median'),
    'income': KNNImputer(n_neighbors=5),
    'preferred_category': ConstantImputer(fill_value='Unknown')
}
```

#### 시나리오 2: IoT 센서 데이터
**데이터**: 시간당 측정, 온도/습도/압력
**결측 패턴**:
- 센서 오류로 인한 간헐적 누락 (5-10%)
- 시간적 연속성이 중요

**적용 전략**:
```python
# 시계열 특성 활용
df['temperature'] = df['temperature'].interpolate(method='linear')
df['humidity'] = df['humidity'].interpolate(method='spline', order=2)

# 짧은 갭만 채우고 긴 갭은 유지
df['pressure'] = df['pressure'].interpolate(limit=5)  # 최대 5개까지만
```

---

## 3. 구현: 상세 Python 코드

### 3.1 통합 Imputation 프레임워크

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from typing import Dict, Union, Any, Callable
import warnings
warnings.filterwarnings('ignore')

class UnifiedImputer:
    """
    Unified imputation framework supporting multiple strategies
    """
    
    def __init__(self):
        self.imputers = {}
        self.strategies = {}
        self.fitted = False
        
    def fit(self, df: pd.DataFrame, 
            strategy_map: Dict[str, Union[str, Dict]]) -> 'UnifiedImputer':
        """
        Fit imputers for each column
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data
        strategy_map : dict
            Mapping of column names to strategies
            
        Example:
        --------
        >>> strategy_map = {
        ...     'age': 'median',
        ...     'income': {'method': 'knn', 'n_neighbors': 5},
        ...     'category': 'mode',
        ...     'timestamp': 'ffill'
        ... }
        >>> imputer = UnifiedImputer().fit(df, strategy_map)
        """
        
        self.strategies = strategy_map
        
        for col, strategy in strategy_map.items():
            if col not in df.columns:
                continue
                
            # Parse strategy
            if isinstance(strategy, str):
                method = strategy
                params = {}
            else:
                method = strategy.get('method', 'mean')
                params = {k: v for k, v in strategy.items() if k != 'method'}
            
            # Create and fit imputer
            if method in ['mean', 'median', 'most_frequent', 'constant']:
                imputer = SimpleImputer(strategy=method, **params)
                self.imputers[col] = imputer.fit(df[[col]])
            
            elif method == 'mode':
                # Mode using pandas
                self.imputers[col] = {'method': 'mode', 
                                     'value': df[col].mode()[0] if len(df[col].mode()) > 0 else None}
            
            elif method == 'knn':
                n_neighbors = params.get('n_neighbors', 5)
                imputer = KNNImputer(n_neighbors=n_neighbors)
                # KNN requires numeric data
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if col in numeric_cols:
                    self.imputers[col] = {'method': 'knn', 
                                         'imputer': imputer.fit(df[numeric_cols]),
                                         'numeric_cols': numeric_cols}
            
            elif method == 'iterative':
                max_iter = params.get('max_iter', 10)
                imputer = IterativeImputer(max_iter=max_iter, random_state=42)
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if col in numeric_cols:
                    self.imputers[col] = {'method': 'iterative',
                                         'imputer': imputer.fit(df[numeric_cols]),
                                         'numeric_cols': numeric_cols}
            
            elif method in ['ffill', 'bfill', 'interpolate']:
                self.imputers[col] = {'method': method, 'params': params}
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply imputation to dataframe
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data to impute
            
        Returns:
        --------
        pd.DataFrame
            Imputed dataframe
        """
        
        if not self.fitted:
            raise ValueError("Imputer not fitted. Call fit() first.")
        
        df_imputed = df.copy()
        
        for col, imputer_obj in self.imputers.items():
            if col not in df.columns:
                continue
            
            if isinstance(imputer_obj, dict):
                method = imputer_obj['method']
                
                if method == 'mode':
                    df_imputed[col].fillna(imputer_obj['value'], inplace=True)
                
                elif method == 'knn':
                    numeric_cols = imputer_obj['numeric_cols']
                    imputed_values = imputer_obj['imputer'].transform(df[numeric_cols])
                    col_idx = numeric_cols.index(col)
                    df_imputed[col] = imputed_values[:, col_idx]
                
                elif method == 'iterative':
                    numeric_cols = imputer_obj['numeric_cols']
                    imputed_values = imputer_obj['imputer'].transform(df[numeric_cols])
                    col_idx = numeric_cols.index(col)
                    df_imputed[col] = imputed_values[:, col_idx]
                
                elif method == 'ffill':
                    df_imputed[col].fillna(method='ffill', inplace=True)
                
                elif method == 'bfill':
                    df_imputed[col].fillna(method='bfill', inplace=True)
                
                elif method == 'interpolate':
                    interp_method = imputer_obj['params'].get('method', 'linear')
                    df_imputed[col] = df[col].interpolate(method=interp_method)
            
            else:
                # SimpleImputer
                df_imputed[col] = imputer_obj.transform(df[[col]])
        
        return df_imputed
    
    def fit_transform(self, df: pd.DataFrame, 
                     strategy_map: Dict[str, Union[str, Dict]]) -> pd.DataFrame:
        """
        Fit and transform in one step
        """
        return self.fit(df, strategy_map).transform(df)


def simple_imputation(df: pd.DataFrame, 
                     column: str,
                     strategy: str = 'mean') -> pd.Series:
    """
    Simple imputation for a single column
    
    Parameters:
    -----------
    strategy : str
        'mean', 'median', 'mode', 'constant'
        
    Returns:
    --------
    pd.Series
        Imputed column
        
    Example:
    --------
    >>> df['age'] = simple_imputation(df, 'age', strategy='median')
    """
    
    if strategy == 'mean':
        return df[column].fillna(df[column].mean())
    
    elif strategy == 'median':
        return df[column].fillna(df[column].median())
    
    elif strategy == 'mode':
        mode_value = df[column].mode()[0] if len(df[column].mode()) > 0 else None
        return df[column].fillna(mode_value)
    
    elif strategy == 'constant':
        # Default constant values by dtype
        if pd.api.types.is_numeric_dtype(df[column]):
            return df[column].fillna(0)
        else:
            return df[column].fillna('Unknown')
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
```

### 3.2 시계열 Imputation

```python
def time_series_imputation(df: pd.DataFrame,
                          column: str,
                          method: str = 'linear',
                          limit: int = None,
                          limit_direction: str = 'forward') -> pd.Series:
    """
    Imputation methods for time-series data
    
    Parameters:
    -----------
    method : str
        'ffill' (forward fill), 'bfill' (backward fill), 
        'linear', 'spline', 'polynomial'
    limit : int, optional
        Maximum number of consecutive NaNs to fill
    limit_direction : str
        'forward', 'backward', or 'both'
        
    Returns:
    --------
    pd.Series
        Imputed time series
        
    Example:
    --------
    >>> df['temperature'] = time_series_imputation(
    ...     df, 'temperature', method='linear', limit=5
    ... )
    """
    
    if method == 'ffill':
        return df[column].fillna(method='ffill', limit=limit)
    
    elif method == 'bfill':
        return df[column].fillna(method='bfill', limit=limit)
    
    elif method == 'linear':
        return df[column].interpolate(method='linear', 
                                       limit=limit, 
                                       limit_direction=limit_direction)
    
    elif method == 'spline':
        # Spline interpolation (requires more data points)
        return df[column].interpolate(method='spline', order=2, 
                                       limit=limit)
    
    elif method == 'polynomial':
        return df[column].interpolate(method='polynomial', order=2,
                                       limit=limit)
    
    elif method == 'time':
        # Time-based interpolation (requires datetime index)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Time interpolation requires DatetimeIndex")
        return df[column].interpolate(method='time')
    
    else:
        raise ValueError(f"Unknown method: {method}")


def seasonal_imputation(df: pd.DataFrame,
                       column: str,
                       period: int = 7) -> pd.Series:
    """
    Imputation using seasonal patterns
    
    Parameters:
    -----------
    period : int
        Seasonality period (e.g., 7 for weekly, 12 for monthly)
        
    Example:
    --------
    >>> # Fill missing values using same weekday's average
    >>> df['sales'] = seasonal_imputation(df, 'sales', period=7)
    """
    
    result = df[column].copy()
    missing_mask = result.isnull()
    
    if not missing_mask.any():
        return result
    
    # Calculate seasonal means
    seasonal_means = {}
    for i in range(period):
        season_values = result[i::period].dropna()
        if len(season_values) > 0:
            seasonal_means[i] = season_values.mean()
    
    # Fill missing values with seasonal means
    for idx in result[missing_mask].index:
        season_idx = idx % period
        if season_idx in seasonal_means:
            result.loc[idx] = seasonal_means[season_idx]
    
    return result
```

### 3.3 KNN Imputation

```python
def knn_imputation(df: pd.DataFrame,
                   columns: list = None,
                   n_neighbors: int = 5,
                   weights: str = 'uniform') -> pd.DataFrame:
    """
    K-Nearest Neighbors imputation
    
    Parameters:
    -----------
    columns : list, optional
        Columns to impute (default: all numeric columns)
    n_neighbors : int
        Number of neighbors to use
    weights : str
        'uniform' or 'distance'
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with imputed values
        
    Example:
    --------
    >>> df_imputed = knn_imputation(df, columns=['age', 'income'], n_neighbors=5)
    """
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_result = df.copy()
    
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    
    # Impute numeric columns
    numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if numeric_cols:
        df_result[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    return df_result


def weighted_knn_imputation(df: pd.DataFrame,
                            target_col: str,
                            feature_cols: list,
                            n_neighbors: int = 5) -> pd.Series:
    """
    KNN imputation with custom feature selection
    
    More control over which features to use for finding neighbors
    
    Example:
    --------
    >>> # Impute 'income' using 'age', 'education', 'experience'
    >>> df['income'] = weighted_knn_imputation(
    ...     df, 'income', ['age', 'education', 'experience'], n_neighbors=5
    ... )
    """
    
    # Prepare data
    all_cols = [target_col] + feature_cols
    df_subset = df[all_cols].copy()
    
    # Apply KNN
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
    imputed_data = imputer.fit_transform(df_subset)
    
    # Return only target column
    target_idx = 0  # target_col is first
    return pd.Series(imputed_data[:, target_idx], index=df.index)
```

### 3.4 Iterative Imputer (MICE)

```python
def iterative_imputation(df: pd.DataFrame,
                        columns: list = None,
                        max_iter: int = 10,
                        initial_strategy: str = 'mean',
                        random_state: int = 42) -> pd.DataFrame:
    """
    Multivariate Iterative Imputer (MICE algorithm)
    
    Iteratively models each feature with missing values as a function 
    of other features
    
    Parameters:
    -----------
    max_iter : int
        Maximum number of imputation rounds
    initial_strategy : str
        Strategy for initial imputation ('mean', 'median', 'most_frequent')
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with imputed values
        
    Example:
    --------
    >>> df_imputed = iterative_imputation(df, max_iter=10)
    """
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_result = df.copy()
    
    imputer = IterativeImputer(
        max_iter=max_iter,
        initial_strategy=initial_strategy,
        random_state=random_state,
        verbose=0
    )
    
    # Impute numeric columns
    numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if numeric_cols:
        df_result[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    return df_result


def mice_with_categorical(df: pd.DataFrame,
                         numeric_cols: list,
                         categorical_cols: list,
                         max_iter: int = 10) -> pd.DataFrame:
    """
    MICE imputation handling both numeric and categorical variables
    
    Categorical variables are encoded before imputation
    
    Example:
    --------
    >>> df_imputed = mice_with_categorical(
    ...     df, 
    ...     numeric_cols=['age', 'income'],
    ...     categorical_cols=['education', 'occupation'],
    ...     max_iter=10
    ... )
    """
    
    from sklearn.preprocessing import LabelEncoder
    
    df_encoded = df.copy()
    label_encoders = {}
    
    # Encode categorical columns
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle missing values in encoding
        non_null_mask = df[col].notnull()
        df_encoded.loc[non_null_mask, col] = le.fit_transform(df.loc[non_null_mask, col])
        label_encoders[col] = le
    
    # Apply MICE
    all_cols = numeric_cols + categorical_cols
    imputer = IterativeImputer(max_iter=max_iter, random_state=42)
    df_encoded[all_cols] = imputer.fit_transform(df_encoded[all_cols])
    
    # Decode categorical columns
    for col in categorical_cols:
        le = label_encoders[col]
        # Round and clip to valid range
        df_encoded[col] = df_encoded[col].round().clip(0, len(le.classes_) - 1).astype(int)
        df_encoded[col] = le.inverse_transform(df_encoded[col].astype(int))
    
    return df_encoded
```

### 3.5 대체 품질 검증

```python
def validate_imputation(df_original: pd.DataFrame,
                       df_imputed: pd.DataFrame,
                       columns: list = None) -> pd.DataFrame:
    """
    Validate imputation quality by comparing distributions
    
    Parameters:
    -----------
    df_original : pd.DataFrame
        Original data with missing values
    df_imputed : pd.DataFrame
        Imputed data
    columns : list, optional
        Columns to validate
        
    Returns:
    --------
    pd.DataFrame
        Validation metrics for each column
        
    Example:
    --------
    >>> validation_report = validate_imputation(df_original, df_imputed)
    >>> print(validation_report)
    """
    
    from scipy.stats import ks_2samp, chi2_contingency
    
    if columns is None:
        # Find columns that had missing values
        columns = df_original.columns[df_original.isnull().any()].tolist()
    
    validation_results = []
    
    for col in columns:
        if col not in df_original.columns or col not in df_imputed.columns:
            continue
        
        result = {'column': col, 'dtype': str(df_original[col].dtype)}
        
        # Original non-missing values
        original_complete = df_original[col].dropna()
        
        # Imputed values (only the ones that were missing)
        was_missing = df_original[col].isnull()
        imputed_values = df_imputed.loc[was_missing, col]
        
        result['n_imputed'] = len(imputed_values)
        result['n_original'] = len(original_complete)
        
        if pd.api.types.is_numeric_dtype(df_original[col]):
            # Numeric column validation
            
            # 1. Descriptive statistics comparison
            result['original_mean'] = round(original_complete.mean(), 2)
            result['imputed_mean'] = round(imputed_values.mean(), 2)
            result['mean_diff_pct'] = round(100 * abs(
                result['imputed_mean'] - result['original_mean']
            ) / result['original_mean'], 2) if result['original_mean'] != 0 else 0
            
            result['original_std'] = round(original_complete.std(), 2)
            result['imputed_std'] = round(imputed_values.std(), 2)
            
            # 2. Kolmogorov-Smirnov test (distribution similarity)
            if len(imputed_values) > 0:
                ks_stat, ks_pvalue = ks_2samp(original_complete, imputed_values)
                result['ks_statistic'] = round(ks_stat, 4)
                result['ks_pvalue'] = round(ks_pvalue, 4)
                result['distribution_similar'] = ks_pvalue > 0.05
            
            # 3. Check for unrealistic values
            original_min = original_complete.min()
            original_max = original_complete.max()
            out_of_range = ((imputed_values < original_min) | 
                           (imputed_values > original_max)).sum()
            result['out_of_range_count'] = out_of_range
            result['out_of_range_pct'] = round(100 * out_of_range / len(imputed_values), 2)
        
        else:
            # Categorical column validation
            
            # Value distribution comparison
            original_dist = original_complete.value_counts(normalize=True)
            imputed_dist = imputed_values.value_counts(normalize=True)
            
            # Check if imputed values introduce new categories
            new_categories = set(imputed_values.unique()) - set(original_complete.unique())
            result['new_categories'] = list(new_categories) if new_categories else None
            
            # Mode comparison
            result['original_mode'] = original_complete.mode()[0] if len(original_complete.mode()) > 0 else None
            result['imputed_mode'] = imputed_values.mode()[0] if len(imputed_values.mode()) > 0 else None
        
        # Overall quality grade
        if pd.api.types.is_numeric_dtype(df_original[col]):
            if result.get('distribution_similar', False) and result.get('out_of_range_pct', 100) < 5:
                result['quality_grade'] = 'Excellent'
            elif result.get('mean_diff_pct', 100) < 10 and result.get('out_of_range_pct', 100) < 10:
                result['quality_grade'] = 'Good'
            elif result.get('mean_diff_pct', 100) < 20:
                result['quality_grade'] = 'Fair'
            else:
                result['quality_grade'] = 'Poor'
        else:
            result['quality_grade'] = 'N/A'
        
        validation_results.append(result)
    
    return pd.DataFrame(validation_results)


def visualize_imputation_impact(df_original: pd.DataFrame,
                               df_imputed: pd.DataFrame,
                               column: str) -> None:
    """
    Visualize the impact of imputation on a column
    
    Creates before/after distribution plots
    """
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Original complete values
    original_complete = df_original[column].dropna()
    
    # Imputed values
    was_missing = df_original[column].isnull()
    imputed_values = df_imputed.loc[was_missing, column]
    
    # All imputed data
    all_imputed = df_imputed[column]
    
    if pd.api.types.is_numeric_dtype(df_original[column]):
        # Histogram comparison
        axes[0].hist(original_complete, bins=30, alpha=0.7, label='Original', color='blue', edgecolor='black')
        axes[0].set_title('Original (non-missing)')
        axes[0].set_xlabel(column)
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        
        axes[1].hist(imputed_values, bins=30, alpha=0.7, label='Imputed', color='red', edgecolor='black')
        axes[1].set_title('Imputed Values')
        axes[1].set_xlabel(column)
        axes[1].legend()
        
        axes[2].hist(original_complete, bins=30, alpha=0.5, label='Original', color='blue', edgecolor='black')
        axes[2].hist(imputed_values, bins=30, alpha=0.5, label='Imputed', color='red', edgecolor='black')
        axes[2].set_title('Overlay')
        axes[2].set_xlabel(column)
        axes[2].legend()
    
    else:
        # Bar chart for categorical
        original_counts = original_complete.value_counts().head(10)
        imputed_counts = imputed_values.value_counts().head(10)
        
        axes[0].bar(range(len(original_counts)), original_counts.values, color='blue', alpha=0.7)
        axes[0].set_xticks(range(len(original_counts)))
        axes[0].set_xticklabels(original_counts.index, rotation=45, ha='right')
        axes[0].set_title('Original (non-missing)')
        axes[0].set_ylabel('Count')
        
        axes[1].bar(range(len(imputed_counts)), imputed_counts.values, color='red', alpha=0.7)
        axes[1].set_xticks(range(len(imputed_counts)))
        axes[1].set_xticklabels(imputed_counts.index, rotation=45, ha='right')
        axes[1].set_title('Imputed Values')
        
        # Combined distribution
        combined_counts = all_imputed.value_counts().head(10)
        axes[2].bar(range(len(combined_counts)), combined_counts.values, color='green', alpha=0.7)
        axes[2].set_xticks(range(len(combined_counts)))
        axes[2].set_xticklabels(combined_counts.index, rotation=45, ha='right')
        axes[2].set_title('After Imputation')
    
    plt.tight_layout()
    plt.show()
```

### 3.6 자동 전략 선택

```python
def select_optimal_strategy(df: pd.DataFrame,
                           column: str,
                           strategies: list = None,
                           test_size: float = 0.2) -> Dict[str, Any]:
    """
    Automatically select best imputation strategy by testing on complete data
    
    Parameters:
    -----------
    strategies : list
        List of strategies to test (default: ['mean', 'median', 'knn', 'iterative'])
    test_size : float
        Proportion of complete data to artificially mask for testing
        
    Returns:
    --------
    dict
        Best strategy and performance metrics
        
    Example:
    --------
    >>> best_strategy = select_optimal_strategy(df, 'income')
    >>> print(f"Best strategy: {best_strategy['strategy']}")
    >>> print(f"RMSE: {best_strategy['rmse']:.2f}")
    """
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    if strategies is None:
        strategies = ['mean', 'median', 'knn', 'iterative']
    
    # Use only complete cases for testing
    df_complete = df[df[column].notnull()].copy()
    
    if len(df_complete) < 50:
        return {'error': 'Not enough complete cases for testing'}
    
    # Artificially create missing values
    n_mask = int(len(df_complete) * test_size)
    mask_indices = np.random.choice(df_complete.index, size=n_mask, replace=False)
    
    true_values = df_complete.loc[mask_indices, column].copy()
    df_test = df_complete.copy()
    df_test.loc[mask_indices, column] = np.nan
    
    results = []
    
    for strategy in strategies:
        try:
            # Apply imputation
            if strategy in ['mean', 'median', 'mode']:
                imputer = UnifiedImputer()
                df_imputed = imputer.fit_transform(df_test, {column: strategy})
            
            elif strategy == 'knn':
                df_imputed = knn_imputation(df_test, columns=[column], n_neighbors=5)
            
            elif strategy == 'iterative':
                df_imputed = iterative_imputation(df_test, columns=[column], max_iter=10)
            
            # Calculate error
            imputed_values = df_imputed.loc[mask_indices, column]
            
            if pd.api.types.is_numeric_dtype(df[column]):
                rmse = np.sqrt(mean_squared_error(true_values, imputed_values))
                mae = mean_absolute_error(true_values, imputed_values)
                mape = np.mean(np.abs((true_values - imputed_values) / true_values)) * 100
                
                results.append({
                    'strategy': strategy,
                    'rmse': round(rmse, 4),
                    'mae': round(mae, 4),
                    'mape': round(mape, 2)
                })
        
        except Exception as e:
            print(f"Strategy '{strategy}' failed: {str(e)}")
            continue
    
    if not results:
        return {'error': 'All strategies failed'}
    
    # Select best based on RMSE
    results_df = pd.DataFrame(results).sort_values('rmse')
    best = results_df.iloc[0].to_dict()
    best['all_results'] = results_df
    
    return best
```

---

## 4. 예시: 입출력 샘플

### 4.1 샘플 데이터

```python
# Create sample data with various missing patterns
np.random.seed(42)
n = 1000

df_sample = pd.DataFrame({
    'age': np.random.randint(18, 80, n),
    'income': np.random.uniform(20000, 150000, n),
    'credit_score': np.random.randint(300, 850, n),
    'years_employed': np.random.randint(0, 40, n),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n)
})

# Introduce missing values
df_sample.loc[np.random.choice(df_sample.index, 80), 'age'] = np.nan  # 8%
df_sample.loc[np.random.choice(df_sample.index, 150), 'income'] = np.nan  # 15%
df_sample.loc[np.random.choice(df_sample.index, 50), 'credit_score'] = np.nan  # 5%
df_sample.loc[np.random.choice(df_sample.index, 100), 'education'] = np.nan  # 10%

print("Missing values:")
print(df_sample.isnull().sum())
```

### 4.2 다양한 전략 적용

```python
# Strategy 1: Simple imputation
strategy_simple = {
    'age': 'median',
    'income': 'mean',
    'credit_score': 'median',
    'education': 'mode'
}

imputer_simple = UnifiedImputer()
df_simple = imputer_simple.fit_transform(df_sample, strategy_simple)

print("\n1. Simple Imputation:")
print(df_simple.isnull().sum())

# Strategy 2: KNN imputation
df_knn = knn_imputation(df_sample, columns=['age', 'income', 'credit_score'], n_neighbors=5)

print("\n2. KNN Imputation:")
print(df_knn[['age', 'income', 'credit_score']].isnull().sum())

# Strategy 3: MICE
df_mice = iterative_imputation(df_sample, columns=['age', 'income', 'credit_score'], max_iter=10)

print("\n3. MICE Imputation:")
print(df_mice[['age', 'income', 'credit_score']].isnull().sum())

# Compare strategies
print("\n4. Validation:")
validation = validate_imputation(df_sample, df_mice, columns=['age', 'income'])
print(validation[['column', 'quality_grade', 'mean_diff_pct', 'ks_pvalue']])
```

---

## 5. 에이전트 매핑

### 5.1 Primary: `data-cleaning-specialist`
- 모든 imputation 전략 실행
- 품질 검증 수행

### 5.2 Supporting: `feature-engineering-specialist`
- 고급 imputation (MICE, ML 기반)
- 파생 변수를 활용한 imputation

---

## 6. 필요 라이브러리

```bash
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install scikit-learn>=1.3.0
pip install scipy>=1.10.0
pip install matplotlib>=3.7.0
```

---

## 7. 체크포인트

- [ ] 결측 메커니즘을 고려했는가?
- [ ] 여러 전략을 비교했는가?
- [ ] 대체 후 분포를 검증했는가?
- [ ] 비현실적 값이 생성되지 않았는가?
- [ ] Train/test leakage를 방지했는가?

---

## 8. 트러블슈팅

### 문제 1: KNN이 너무 느림
**해결**: n_neighbors를 줄이거나 샘플링 사용

### 문제 2: MICE가 수렴하지 않음
**해결**: max_iter 증가 또는 initial_strategy 변경

### 문제 3: 비현실적 값 생성
**해결**: 대체 후 값 범위 clipping 적용

---

## 9. 참고 자료

- scikit-learn Imputation: https://scikit-learn.org/stable/modules/impute.html
- "Flexible Imputation of Missing Data" by Stef van Buuren

---

## 10. 요약

**핵심 원칙**:
1. 메커니즘에 맞는 전략 선택
2. 항상 검증 수행
3. 여러 방법 비교
4. 분포 보존 확인

**다음 단계**: Reference 04 (Statistical Outlier Detection)

---

**작성자**: Claude Code  
**마지막 업데이트**: 2025-01-25
