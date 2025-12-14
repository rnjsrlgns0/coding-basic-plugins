# Reference 04: Univariate Analysis

**Version**: 1.0  
**Last Updated**: 2025-01-25  
**Workflow Phase**: Phase 3.1 - Descriptive Statistics (Univariate)  
**Estimated Reading Time**: 22-25 minutes

---

## 1. 개요 (Overview)

### 1.1 목적 (Purpose)

단변량 분석(Univariate Analysis)은 각 변수를 개별적으로 분석하여 그 특성, 분포, 패턴을 이해하는 프로세스입니다. 변수 간 관계를 보기 전에, 먼저 각 변수의 "성격"을 파악하는 것이 중요합니다.

**주요 목적**:
- 변수의 중심 경향성(평균, 중앙값, 최빈값) 파악
- 변수의 산포도(분산, 표준편차, 범위) 이해
- 분포의 형태(정규성, 왜도, 첨도) 분석
- 이상치(Outliers) 탐지
- 각 변수의 특이 패턴 발견

### 1.2 적용 시기 (When to Apply)

단변량 분석은 EDA의 핵심 단계로, 다음 상황에서 반드시 수행해야 합니다:

1. **자동화 프로파일링 후**: 개별 변수에 대한 심층 분석 필요 시
2. **데이터 품질 검증**: 각 변수의 값 범위와 분포 확인
3. **피처 엔지니어링 전**: 변환이 필요한 변수 식별
4. **모델링 전 전처리**: 정규화/표준화 필요성 판단
5. **이상치 처리**: 이상치 탐지 및 처리 전략 수립

### 1.3 분석 유형

| 데이터 타입 | 분석 방법 | 시각화 |
|------------|-----------|--------|
| **연속형 수치** | 기술 통계, 분포 분석, 이상치 탐지 | 히스토그램, 박스플롯, KDE |
| **이산형 수치** | 빈도 분석, 집중도 분석 | 막대 그래프, 파이 차트 |
| **범주형** | 빈도 분석, 모드 분석 | 막대 그래프, 파이 차트 |
| **날짜/시간** | 시간 패턴, 트렌드 분석 | 시계열 플롯, 히트맵 |

---

## 2. 이론적 배경 (Theoretical Background)

### 2.1 기술 통계량 (Descriptive Statistics)

**중심 경향성 측도 (Measures of Central Tendency)**:
- **평균 (Mean)**: 모든 값의 산술 평균, 이상치에 민감
- **중앙값 (Median)**: 중간값, 이상치에 강건
- **최빈값 (Mode)**: 가장 빈번한 값

**산포도 측도 (Measures of Dispersion)**:
- **분산 (Variance)**: 평균으로부터의 제곱 편차 평균
- **표준편차 (Standard Deviation)**: 분산의 제곱근, 원 단위
- **범위 (Range)**: 최대값 - 최소값
- **사분위수 범위 (IQR)**: Q3 - Q1, 이상치 탐지에 사용

**분포 형태 (Shape of Distribution)**:
- **왜도 (Skewness)**: 분포의 비대칭 정도
  - 0: 대칭
  - > 0: 오른쪽 꼬리 (right-skewed, positive skew)
  - < 0: 왼쪽 꼬리 (left-skewed, negative skew)
- **첨도 (Kurtosis)**: 분포의 꼬리 두께
  - 3: 정규분포 (mesokurtic)
  - > 3: 뾰족한 분포 (leptokurtic)
  - < 3: 평평한 분포 (platykurtic)

### 2.2 분포 유형

**정규분포 (Normal Distribution)**:
- 종 모양, 평균=중앙값=최빈값
- 많은 통계 기법의 전제 조건
- 68-95-99.7 규칙 (표준편차 기준)

**왜곡 분포 (Skewed Distribution)**:
- 오른쪽 왜곡: 소득, 가격 등
- 왼쪽 왜곡: 시험 점수(상한 존재), 나이(상한 존재)

**다봉 분포 (Multimodal Distribution)**:
- 여러 개의 피크
- 서로 다른 그룹의 혼합 가능성

**균등 분포 (Uniform Distribution)**:
- 모든 값의 빈도가 유사
- 무작위 데이터 또는 인위적 생성 데이터

### 2.3 이상치 탐지 방법

**1. IQR 방법 (Interquartile Range)**:
```
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
Lower bound = Q1 - 1.5 * IQR
Upper bound = Q3 + 1.5 * IQR
```

**2. Z-Score 방법**:
```
Z = (X - μ) / σ
|Z| > 3: 극단적 이상치
|Z| > 2: 잠재적 이상치
```

**3. Modified Z-Score (Median Absolute Deviation)**:
```
MAD = median(|X_i - median(X)|)
Modified Z = 0.6745 * (X - median(X)) / MAD
```

### 2.4 일반적인 시나리오

**시나리오 1: 정규분포 확인**
- 목적: 모수 검정 가능 여부 판단
- 방법: Q-Q plot, Shapiro-Wilk test, Kolmogorov-Smirnov test

**시나리오 2: 이상치 탐지**
- 목적: 데이터 품질 이슈 또는 특이 케이스 발견
- 방법: 박스플롯, IQR, Z-score

**시나리오 3: 변수 변환 필요성 판단**
- 목적: 모델링을 위한 전처리
- 방법: 왜도 확인, 분포 시각화

**시나리오 4: 범주형 변수 불균형 확인**
- 목적: 샘플링 전략 또는 가중치 필요성 판단
- 방법: 빈도표, 막대 그래프

---

## 3. 구현 (Implementation)

### 3.1 수치형 변수 분석

#### 3.1.1 기술 통계량 계산

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def analyze_numeric_variable(
    df: pd.DataFrame,
    column: str,
    show_plots: bool = True
) -> dict:
    """
    수치형 변수의 포괄적 단변량 분석
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    column : str
        분석할 수치형 컬럼명
    show_plots : bool, default True
        시각화 표시 여부
    
    Returns:
    --------
    dict
        분석 결과 딕셔너리
    
    Examples:
    ---------
    >>> results = analyze_numeric_variable(df, 'age')
    >>> print(results['mean'], results['skewness'])
    """
    if column not in df.columns:
        raise ValueError(f"컬럼 '{column}'이 존재하지 않습니다.")
    
    data = df[column].dropna()
    
    if len(data) == 0:
        raise ValueError(f"컬럼 '{column}'에 유효한 데이터가 없습니다.")
    
    print("=" * 80)
    print(f"수치형 변수 분석: {column}")
    print("=" * 80)
    
    # 1. 기본 정보
    print(f"\n[기본 정보]")
    print(f"  데이터 타입: {df[column].dtype}")
    print(f"  총 관측치: {len(df):,}")
    print(f"  유효 관측치: {len(data):,}")
    print(f"  결측값: {df[column].isnull().sum():,} ({100*df[column].isnull().mean():.2f}%)")
    print(f"  고유값 개수: {data.nunique():,}")
    
    # 2. 중심 경향성
    mean_val = data.mean()
    median_val = data.median()
    mode_val = data.mode().iloc[0] if len(data.mode()) > 0 else np.nan
    
    print(f"\n[중심 경향성]")
    print(f"  평균 (Mean): {mean_val:.4f}")
    print(f"  중앙값 (Median): {median_val:.4f}")
    print(f"  최빈값 (Mode): {mode_val:.4f}")
    print(f"  평균 vs 중앙값 차이: {abs(mean_val - median_val):.4f}")
    
    # 3. 산포도
    std_val = data.std()
    var_val = data.var()
    range_val = data.max() - data.min()
    iqr_val = data.quantile(0.75) - data.quantile(0.25)
    cv = (std_val / mean_val) * 100 if mean_val != 0 else np.inf
    
    print(f"\n[산포도]")
    print(f"  표준편차 (Std): {std_val:.4f}")
    print(f"  분산 (Variance): {var_val:.4f}")
    print(f"  범위 (Range): {range_val:.4f}")
    print(f"  사분위수 범위 (IQR): {iqr_val:.4f}")
    print(f"  변동계수 (CV): {cv:.2f}%")
    
    # 4. 분포 형태
    skewness = stats.skew(data)
    kurtosis_val = stats.kurtosis(data)
    
    print(f"\n[분포 형태]")
    print(f"  왜도 (Skewness): {skewness:.4f}")
    if abs(skewness) < 0.5:
        skew_interp = "대칭 분포"
    elif skewness > 0:
        skew_interp = "오른쪽 꼬리 (양의 왜도)"
    else:
        skew_interp = "왼쪽 꼬리 (음의 왜도)"
    print(f"    해석: {skew_interp}")
    
    print(f"  첨도 (Kurtosis): {kurtosis_val:.4f}")
    if abs(kurtosis_val) < 0.5:
        kurt_interp = "정규분포 수준"
    elif kurtosis_val > 0:
        kurt_interp = "뾰족한 분포 (leptokurtic)"
    else:
        kurt_interp = "평평한 분포 (platykurtic)"
    print(f"    해석: {kurt_interp}")
    
    # 5. 백분위수
    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    print(f"\n[백분위수]")
    for p in percentiles:
        val = data.quantile(p / 100)
        print(f"  {p:3d}%: {val:12.4f}")
    
    # 6. 이상치 탐지 (IQR 방법)
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    n_outliers = len(outliers)
    outlier_pct = 100 * n_outliers / len(data)
    
    print(f"\n[이상치 (IQR 방법)]")
    print(f"  하한: {lower_bound:.4f}")
    print(f"  상한: {upper_bound:.4f}")
    print(f"  이상치 개수: {n_outliers:,} ({outlier_pct:.2f}%)")
    if n_outliers > 0 and n_outliers <= 10:
        print(f"  이상치 값: {sorted(outliers.tolist())}")
    elif n_outliers > 10:
        print(f"  이상치 예시 (하위 5개): {sorted(outliers.tolist())[:5]}")
        print(f"  이상치 예시 (상위 5개): {sorted(outliers.tolist(), reverse=True)[:5]}")
    
    # 7. 정규성 검정
    if len(data) >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(data.sample(min(5000, len(data))))
        print(f"\n[정규성 검정]")
        print(f"  Shapiro-Wilk 통계량: {shapiro_stat:.6f}")
        print(f"  p-value: {shapiro_p:.6f}")
        if shapiro_p > 0.05:
            print(f"  결론: 정규분포를 따른다고 볼 수 있음 (p > 0.05)")
        else:
            print(f"  결론: 정규분포를 따르지 않음 (p <= 0.05)")
    
    # 8. 시각화
    if show_plots:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'단변량 분석: {column}', fontsize=16, fontweight='bold')
        
        # (1) 히스토그램
        axes[0, 0].hist(data, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        axes[0, 0].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        axes[0, 0].set_title('히스토그램', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel(column)
        axes[0, 0].set_ylabel('빈도')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # (2) 박스플롯
        axes[0, 1].boxplot(data, vert=True)
        axes[0, 1].set_title('박스플롯', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel(column)
        axes[0, 1].grid(True, alpha=0.3)
        
        # (3) KDE (커널 밀도 추정)
        data.plot(kind='kde', ax=axes[0, 2], linewidth=2)
        axes[0, 2].axvline(mean_val, color='red', linestyle='--', linewidth=2, label='Mean')
        axes[0, 2].axvline(median_val, color='green', linestyle='--', linewidth=2, label='Median')
        axes[0, 2].set_title('커널 밀도 추정 (KDE)', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel(column)
        axes[0, 2].set_ylabel('밀도')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # (4) Q-Q Plot (정규성 확인)
        stats.probplot(data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (정규분포 비교)', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # (5) 바이올린 플롯
        parts = axes[1, 1].violinplot([data], positions=[0], showmeans=True, showmedians=True)
        axes[1, 1].set_title('바이올린 플롯', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel(column)
        axes[1, 1].set_xticks([0])
        axes[1, 1].set_xticklabels([column])
        axes[1, 1].grid(True, alpha=0.3)
        
        # (6) 누적 분포 함수 (CDF)
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        axes[1, 2].plot(sorted_data, cdf, linewidth=2)
        axes[1, 2].set_title('누적 분포 함수 (CDF)', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel(column)
        axes[1, 2].set_ylabel('누적 확률')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # 결과 딕셔너리 반환
    return {
        'count': len(data),
        'missing': df[column].isnull().sum(),
        'unique': data.nunique(),
        'mean': mean_val,
        'median': median_val,
        'mode': mode_val,
        'std': std_val,
        'variance': var_val,
        'min': data.min(),
        'max': data.max(),
        'range': range_val,
        'iqr': iqr_val,
        'skewness': skewness,
        'kurtosis': kurtosis_val,
        'cv': cv,
        'q25': Q1,
        'q75': Q3,
        'outliers_count': n_outliers,
        'outliers_pct': outlier_pct,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }


# 사용 예시
results = analyze_numeric_variable(df, 'age', show_plots=True)
print(f"\n평균 나이: {results['mean']:.2f}")
print(f"이상치 비율: {results['outliers_pct']:.2f}%")
```

#### 3.1.2 다중 수치형 변수 일괄 분석

```python
def analyze_all_numeric_variables(
    df: pd.DataFrame,
    include_plots: bool = False
) -> pd.DataFrame:
    """
    모든 수치형 변수를 일괄 분석
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    include_plots : bool, default False
        개별 변수 플롯 생성 여부
    
    Returns:
    --------
    pd.DataFrame
        모든 변수의 통계 요약 데이터프레임
    
    Examples:
    ---------
    >>> summary = analyze_all_numeric_variables(df)
    >>> print(summary)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("수치형 변수가 없습니다.")
        return pd.DataFrame()
    
    print(f"총 {len(numeric_cols)}개의 수치형 변수 분석 중...")
    print("=" * 80)
    
    results_list = []
    
    for col in numeric_cols:
        print(f"\n[{col}]")
        try:
            result = analyze_numeric_variable(df, col, show_plots=include_plots)
            result['variable'] = col
            results_list.append(result)
            print(f"  ✓ 분석 완료")
        except Exception as e:
            print(f"  ✗ 분석 실패: {e}")
    
    # 결과를 DataFrame으로 변환
    summary_df = pd.DataFrame(results_list)
    
    # 변수명을 첫 번째 컬럼으로
    cols = ['variable'] + [c for c in summary_df.columns if c != 'variable']
    summary_df = summary_df[cols]
    
    print("\n" + "=" * 80)
    print("분석 완료 요약")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    
    # 주요 발견 사항
    print("\n" + "=" * 80)
    print("주요 발견 사항")
    print("=" * 80)
    
    # 높은 왜도
    high_skew = summary_df[abs(summary_df['skewness']) > 1.0]
    if len(high_skew) > 0:
        print(f"\n[높은 왜도 (|skew| > 1.0)]")
        for _, row in high_skew.iterrows():
            print(f"  - {row['variable']}: {row['skewness']:.2f}")
    
    # 높은 이상치 비율
    high_outliers = summary_df[summary_df['outliers_pct'] > 5.0]
    if len(high_outliers) > 0:
        print(f"\n[높은 이상치 비율 (> 5%)]")
        for _, row in high_outliers.iterrows():
            print(f"  - {row['variable']}: {row['outliers_pct']:.2f}%")
    
    # 높은 변동계수
    high_cv = summary_df[summary_df['cv'] > 100]
    if len(high_cv) > 0:
        print(f"\n[높은 변동계수 (CV > 100%)]")
        for _, row in high_cv.iterrows():
            print(f"  - {row['variable']}: {row['cv']:.2f}%")
    
    return summary_df


# 사용 예시
summary = analyze_all_numeric_variables(df, include_plots=False)

# CSV로 저장
summary.to_csv('numeric_variables_summary.csv', index=False)
print("\n✓ 요약 저장: numeric_variables_summary.csv")
```

### 3.2 범주형 변수 분석

#### 3.2.1 범주형 변수 분석

```python
def analyze_categorical_variable(
    df: pd.DataFrame,
    column: str,
    top_n: int = 10,
    show_plots: bool = True
) -> dict:
    """
    범주형 변수의 포괄적 단변량 분석
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    column : str
        분석할 범주형 컬럼명
    top_n : int, default 10
        상위 N개 카테고리만 시각화
    show_plots : bool, default True
        시각화 표시 여부
    
    Returns:
    --------
    dict
        분석 결과 딕셔너리
    
    Examples:
    ---------
    >>> results = analyze_categorical_variable(df, 'region')
    """
    if column not in df.columns:
        raise ValueError(f"컬럼 '{column}'이 존재하지 않습니다.")
    
    data = df[column].dropna()
    
    print("=" * 80)
    print(f"범주형 변수 분석: {column}")
    print("=" * 80)
    
    # 1. 기본 정보
    print(f"\n[기본 정보]")
    print(f"  데이터 타입: {df[column].dtype}")
    print(f"  총 관측치: {len(df):,}")
    print(f"  유효 관측치: {len(data):,}")
    print(f"  결측값: {df[column].isnull().sum():,} ({100*df[column].isnull().mean():.2f}%)")
    print(f"  고유값 개수: {data.nunique():,}")
    
    # 2. 빈도 분석
    value_counts = data.value_counts()
    value_props = data.value_counts(normalize=True) * 100
    
    print(f"\n[빈도 분석]")
    print(f"  {'카테고리':<30} {'빈도':>10} {'비율(%)':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    
    for i, (cat, count) in enumerate(value_counts.items()):
        if i < 20:  # 상위 20개만 출력
            prop = value_props[cat]
            print(f"  {str(cat):<30} {count:>10,} {prop:>9.2f}%")
    
    if len(value_counts) > 20:
        print(f"  ... (총 {len(value_counts)}개 카테고리)")
    
    # 3. 최빈값 (Mode)
    mode_val = data.mode().iloc[0] if len(data.mode()) > 0 else None
    mode_count = value_counts.iloc[0]
    mode_pct = value_props.iloc[0]
    
    print(f"\n[최빈값]")
    print(f"  최빈값: {mode_val}")
    print(f"  빈도: {mode_count:,} ({mode_pct:.2f}%)")
    
    # 4. 분포 균형성
    # 균등 분포와의 비교
    expected_freq = len(data) / data.nunique()
    chi2_stat = ((value_counts - expected_freq) ** 2 / expected_freq).sum()
    
    print(f"\n[분포 균형성]")
    print(f"  균등 분포 기대 빈도: {expected_freq:.2f}")
    print(f"  카이제곱 통계량: {chi2_stat:.2f}")
    
    # 집중도 (Concentration)
    # 상위 N개가 전체의 몇 %를 차지하는지
    top_3_pct = value_props.iloc[:3].sum()
    top_5_pct = value_props.iloc[:min(5, len(value_props))].sum()
    top_10_pct = value_props.iloc[:min(10, len(value_props))].sum()
    
    print(f"\n[집중도]")
    print(f"  상위 3개 카테고리: {top_3_pct:.2f}%")
    print(f"  상위 5개 카테고리: {top_5_pct:.2f}%")
    print(f"  상위 10개 카테고리: {top_10_pct:.2f}%")
    
    # 5. 희소 카테고리 (Rare categories)
    rare_threshold = 1.0  # 1% 미만
    rare_cats = value_props[value_props < rare_threshold]
    
    if len(rare_cats) > 0:
        print(f"\n[희소 카테고리 (< {rare_threshold}%)]")
        print(f"  개수: {len(rare_cats)}개")
        print(f"  예시: {list(rare_cats.index[:5])}")
    
    # 6. 시각화
    if show_plots:
        # 상위 N개만 시각화
        plot_data = value_counts.iloc[:top_n]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'범주형 변수 분석: {column}', fontsize=16, fontweight='bold')
        
        # (1) 막대 그래프
        plot_data.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
        axes[0].set_title(f'빈도 (상위 {top_n}개)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('카테고리')
        axes[0].set_ylabel('빈도')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 빈도 숫자 표시
        for i, (idx, val) in enumerate(plot_data.items()):
            axes[0].text(i, val, f'{val:,}', ha='center', va='bottom')
        
        # (2) 파이 차트 (카테고리 10개 이하일 때만)
        if data.nunique() <= 10:
            axes[1].pie(
                plot_data.values,
                labels=plot_data.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=plt.cm.Set3(range(len(plot_data)))
            )
            axes[1].set_title('비율 분포', fontsize=12, fontweight='bold')
        else:
            # 막대 그래프 (비율)
            (value_props.iloc[:top_n]).plot(kind='barh', ax=axes[1], color='coral', edgecolor='black')
            axes[1].set_title(f'비율 (상위 {top_n}개)', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('비율 (%)')
            axes[1].set_ylabel('카테고리')
            axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()
    
    # 결과 반환
    return {
        'count': len(data),
        'missing': df[column].isnull().sum(),
        'unique': data.nunique(),
        'mode': mode_val,
        'mode_count': mode_count,
        'mode_pct': mode_pct,
        'top_3_pct': top_3_pct,
        'top_5_pct': top_5_pct,
        'top_10_pct': top_10_pct,
        'rare_categories': len(rare_cats),
        'value_counts': value_counts.to_dict()
    }


# 사용 예시
results = analyze_categorical_variable(df, 'region', top_n=10, show_plots=True)
print(f"\n최빈값: {results['mode']}")
print(f"상위 3개 집중도: {results['top_3_pct']:.2f}%")
```

#### 3.2.2 다중 범주형 변수 일괄 분석

```python
def analyze_all_categorical_variables(
    df: pd.DataFrame,
    include_plots: bool = False
) -> pd.DataFrame:
    """
    모든 범주형 변수를 일괄 분석
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    include_plots : bool, default False
        개별 변수 플롯 생성 여부
    
    Returns:
    --------
    pd.DataFrame
        모든 범주형 변수의 요약 데이터프레임
    
    Examples:
    ---------
    >>> summary = analyze_all_categorical_variables(df)
    """
    # 범주형 컬럼 선택 (object, category)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(cat_cols) == 0:
        print("범주형 변수가 없습니다.")
        return pd.DataFrame()
    
    print(f"총 {len(cat_cols)}개의 범주형 변수 분석 중...")
    print("=" * 80)
    
    results_list = []
    
    for col in cat_cols:
        print(f"\n[{col}]")
        try:
            result = analyze_categorical_variable(df, col, show_plots=include_plots)
            result['variable'] = col
            # value_counts는 제외 (딕셔너리라 DataFrame에 포함 불가)
            result.pop('value_counts', None)
            results_list.append(result)
            print(f"  ✓ 분석 완료")
        except Exception as e:
            print(f"  ✗ 분석 실패: {e}")
    
    # DataFrame으로 변환
    summary_df = pd.DataFrame(results_list)
    
    # 변수명을 첫 번째 컬럼으로
    cols = ['variable'] + [c for c in summary_df.columns if c != 'variable']
    summary_df = summary_df[cols]
    
    print("\n" + "=" * 80)
    print("분석 완료 요약")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    
    # 주요 발견 사항
    print("\n" + "=" * 80)
    print("주요 발견 사항")
    print("=" * 80)
    
    # 높은 카디널리티
    high_card = summary_df[summary_df['unique'] > 50]
    if len(high_card) > 0:
        print(f"\n[높은 카디널리티 (고유값 > 50)]")
        for _, row in high_card.iterrows():
            print(f"  - {row['variable']}: {row['unique']} 개")
    
    # 불균형 분포
    imbalanced = summary_df[summary_df['mode_pct'] > 80]
    if len(imbalanced) > 0:
        print(f"\n[불균형 분포 (최빈값 > 80%)]")
        for _, row in imbalanced.iterrows():
            print(f"  - {row['variable']}: {row['mode']} ({row['mode_pct']:.2f}%)")
    
    # 많은 희소 카테고리
    many_rare = summary_df[summary_df['rare_categories'] > 10]
    if len(many_rare) > 0:
        print(f"\n[많은 희소 카테고리 (> 10개)]")
        for _, row in many_rare.iterrows():
            print(f"  - {row['variable']}: {row['rare_categories']} 개")
    
    return summary_df


# 사용 예시
summary = analyze_all_categorical_variables(df, include_plots=False)

# CSV로 저장
summary.to_csv('categorical_variables_summary.csv', index=False)
print("\n✓ 요약 저장: categorical_variables_summary.csv")
```

### 3.3 통합 단변량 분석

```python
def comprehensive_univariate_analysis(
    df: pd.DataFrame,
    output_dir: str = "univariate_analysis"
) -> dict:
    """
    모든 변수에 대한 포괄적 단변량 분석
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    output_dir : str, default "univariate_analysis"
        출력 디렉토리
    
    Returns:
    --------
    dict
        분석 결과 딕셔너리
    
    Examples:
    ---------
    >>> results = comprehensive_univariate_analysis(df)
    """
    from pathlib import Path
    import time
    
    # 출력 디렉토리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("포괄적 단변량 분석 시작")
    print("=" * 80)
    print(f"데이터 크기: {df.shape}")
    print(f"출력 디렉토리: {output_dir}")
    print("=" * 80)
    
    start_time = time.time()
    
    # 1. 수치형 변수 분석
    print("\n[1/2] 수치형 변수 분석...")
    numeric_summary = analyze_all_numeric_variables(df, include_plots=False)
    
    if len(numeric_summary) > 0:
        numeric_summary.to_csv(f"{output_dir}/numeric_summary.csv", index=False)
        print(f"  ✓ 수치형 변수 요약 저장: {output_dir}/numeric_summary.csv")
    
    # 2. 범주형 변수 분석
    print("\n[2/2] 범주형 변수 분석...")
    categorical_summary = analyze_all_categorical_variables(df, include_plots=False)
    
    if len(categorical_summary) > 0:
        categorical_summary.to_csv(f"{output_dir}/categorical_summary.csv", index=False)
        print(f"  ✓ 범주형 변수 요약 저장: {output_dir}/categorical_summary.csv")
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("단변량 분석 완료")
    print("=" * 80)
    print(f"총 소요 시간: {elapsed:.2f}초")
    print(f"수치형 변수: {len(numeric_summary)}개")
    print(f"범주형 변수: {len(categorical_summary)}개")
    
    return {
        'numeric_summary': numeric_summary,
        'categorical_summary': categorical_summary,
        'elapsed_time': elapsed
    }


# 사용 예시
results = comprehensive_univariate_analysis(df, output_dir="univariate_reports")
```

---

## 4. 예시 (Examples)

### 4.1 전체 워크플로우 예시

```python
import pandas as pd
import numpy as np

# 샘플 데이터 생성
np.random.seed(42)
n = 5000

df = pd.DataFrame({
    'age': np.random.normal(45, 15, n).clip(18, 90).astype(int),
    'income': np.random.lognormal(10.5, 0.8, n),
    'credit_score': np.random.normal(680, 80, n).clip(300, 850).astype(int),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n, p=[0.3, 0.25, 0.25, 0.2]),
    'education': np.random.choice(['HS', 'Bachelor', 'Master', 'PhD'], n, p=[0.3, 0.4, 0.2, 0.1]),
    'product_type': np.random.choice(['A', 'B', 'C', 'D', 'E'], n)
})

# 결측값 추가
df.loc[np.random.choice(df.index, 100), 'income'] = np.nan
df.loc[np.random.choice(df.index, 50), 'education'] = np.nan

print("샘플 데이터 생성 완료:")
print(df.head())
print(f"\n데이터 크기: {df.shape}")

# 포괄적 단변량 분석 실행
results = comprehensive_univariate_analysis(df, output_dir="demo_univariate")

print("\n분석 결과:")
print(f"- 수치형 변수 분석: {len(results['numeric_summary'])}개")
print(f"- 범주형 변수 분석: {len(results['categorical_summary'])}개")
print(f"- 소요 시간: {results['elapsed_time']:.2f}초")
```

---

## 5. 에이전트 매핑 (Agent Mapping)

### 5.1 담당 에이전트

**Primary Agent**: `data-scientist`
- 역할: 단변량 분석 실행 및 해석
- 책임: 통계량 계산, 분포 분석, 이상치 탐지

**Supporting Agent**: `data-visualization-specialist`
- 역할: 고급 시각화
- 책임: 커스텀 플롯, 인터랙티브 시각화

---

## 6. 필요 라이브러리 (Required Libraries)

```bash
uv pip install pandas==2.2.0 numpy==1.26.3 scipy==1.12.0
uv pip install matplotlib==3.8.2 seaborn==0.13.1
```

---

## 7. 체크포인트 (Checkpoints)

- [ ] 모든 변수의 기술 통계량 계산 완료
- [ ] 분포 형태 (왜도, 첨도) 확인
- [ ] 이상치 탐지 및 비율 파악
- [ ] 결측값 패턴 분석
- [ ] 변환 필요 변수 식별

---

## 8. 트러블슈팅 (Troubleshooting)

**문제 1: 왜도가 매우 높음 (> 2)**
- 해결: 로그 변환, 제곱근 변환, Box-Cox 변환

**문제 2: 이상치 비율 > 10%**
- 해결: 데이터 검증, 도메인 전문가 상담, Winsorization

---

## 9. 참고 자료 (References)

1. **Pandas Documentation**: https://pandas.pydata.org/docs/
2. **SciPy Stats**: https://docs.scipy.org/doc/scipy/reference/stats.html
3. **Seaborn**: https://seaborn.pydata.org/

---

**문서 끝**

다음 단계: [05-bivariate-analysis.md](./05-bivariate-analysis.md) - 이변량 분석
