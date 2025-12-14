# Reference 07: Visualization Patterns

**Version**: 1.0  
**Last Updated**: 2025-01-25  
**Workflow Phase**: Phase 4 - Visualization & Pattern Detection  
**Estimated Reading Time**: 25-28 minutes

---

## 1. 개요 (Overview)

### 1.1 목적 (Purpose)

데이터 시각화는 복잡한 데이터를 직관적으로 이해할 수 있는 그래픽 형태로 변환하는 과정입니다. 효과적인 시각화는 패턴 발견, 이상치 탐지, 인사이트 도출을 가속화합니다.

**주요 목적**:
- 데이터 패턴의 직관적 이해
- 복잡한 관계의 시각적 표현
- 발표 및 커뮤니케이션 자료 생성
- 탐색적 분석 가속화
- 데이터 스토리텔링

### 1.2 적용 시기 (When to Apply)

1. **초기 탐색**: 데이터의 전반적 구조 파악
2. **가설 수립**: 패턴과 관계 발견
3. **모델 결과 해석**: 예측, 잔차, 중요도 시각화
4. **리포트 작성**: 발견 사항 커뮤니케이션
5. **대시보드 구축**: 지속적 모니터링

### 1.3 시각화 선택 가이드

| 목적 | 권장 차트 | 사용 시나리오 |
|------|----------|---------------|
| 분포 확인 | 히스토그램, 박스플롯, 바이올린 플롯 | 변수의 분포 형태 파악 |
| 관계 탐색 | 산점도, 히트맵, 페어플롯 | 변수 간 상관관계 |
| 비교 분석 | 막대 그래프, 박스플롯 | 그룹 간 차이 비교 |
| 시간 추이 | 선 그래프, 영역 차트 | 시계열 패턴 |
| 구성 비율 | 파이 차트, 트리맵 | 전체 대비 부분 비율 |
| 지리 데이터 | 지도, 히트맵 | 공간적 분포 |

---

## 2. 이론적 배경 (Theoretical Background)

### 2.1 시각화 디자인 원칙

**1. 명확성 (Clarity)**
- 핵심 메시지가 즉시 전달되어야 함
- 불필요한 요소 제거 (차트 정크 최소화)
- 레이블, 축, 범례가 명확해야 함

**2. 정확성 (Accuracy)**
- 데이터를 왜곡 없이 표현
- 적절한 축 스케일 선택
- 오해의 소지가 없는 색상 사용

**3. 효율성 (Efficiency)**
- 최소한의 노력으로 최대한의 정보 전달
- 적절한 차트 유형 선택
- 인지 부하 최소화

**4. 미학 (Aesthetics)**
- 시각적으로 매력적
- 일관된 색상 팔레트
- 적절한 여백과 레이아웃

### 2.2 색상 이론

**색상 스킴 유형**:

1. **순차형 (Sequential)**: 낮음 → 높음
   - 사용 예: 밀도, 농도, 온도
   - 팔레트: Blues, Greens, YlOrRd

2. **발산형 (Diverging)**: 중간값 기준 양극
   - 사용 예: 상관계수 (-1 ~ +1), 변화율
   - 팔레트: RdBu, BrBG, PiYG

3. **정성형 (Qualitative)**: 범주 구분
   - 사용 예: 지역, 제품 유형
   - 팔레트: Set1, Set2, Pastel

**색맹 고려 (Color Blindness)**:
- 빨강-녹색 색맹 가장 흔함 (8% 남성)
- ColorBrewer 팔레트 사용 권장
- 색상 + 패턴/텍스처 병행

### 2.3 차트 선택 의사결정 트리

```
데이터 변수 개수는?
├── 1개 (단변량)
│   ├── 수치형 → 히스토그램, 박스플롯, KDE
│   └── 범주형 → 막대 그래프, 파이 차트
├── 2개 (이변량)
│   ├── 수치형-수치형 → 산점도, 헥스빈
│   ├── 수치형-범주형 → 박스플롯, 바이올린 플롯
│   └── 범주형-범주형 → 스택 막대, 히트맵
└── 3개 이상 (다변량)
    ├── 수치형들 → 페어플롯, 히트맵, 3D 산점도
    ├── 차원 축소 → PCA 플롯, t-SNE, UMAP
    └── 시계열 + 범주 → 면적 차트, 라인 플롯
```

### 2.4 일반적인 시나리오

**시나리오 1: 데이터 품질 확인**
- 목적: 결측값, 이상치, 분포 이상 탐지
- 차트: 히트맵 (결측값), 박스플롯 (이상치)

**시나리오 2: 피처 중요도 전달**
- 목적: 모델링 결과 해석
- 차트: 막대 그래프 (수평), 워터폴 차트

**시나리오 3: 시계열 패턴 분석**
- 목적: 트렌드, 계절성, 이상 탐지
- 차트: 선 그래프, 캔들스틱, 히트맵 (시간 x 요일)

**시나리오 4: 세그먼트 비교**
- 목적: 그룹 간 차이 시각화
- 차트: 그룹별 박스플롯, 스몰 멀티플

---

## 3. 구현 (Implementation)

### 3.1 분포 시각화

#### 3.1.1 히스토그램 및 밀도 플롯

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Matplotlib 한글 폰트 설정 (선택)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

def plot_distribution(
    df: pd.DataFrame,
    column: str,
    bins: int = 50,
    show_stats: bool = True,
    figsize: tuple = (14, 6)
) -> None:
    """
    변수의 분포를 다양한 방법으로 시각화
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    column : str
        분석할 컬럼명
    bins : int, default 50
        히스토그램 구간 개수
    show_stats : bool, default True
        통계량 표시 여부
    figsize : tuple, default (14, 6)
        그림 크기
    
    Examples:
    ---------
    >>> plot_distribution(df, 'age', bins=30)
    """
    data = df[column].dropna()
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f'Distribution Analysis: {column}', fontsize=16, fontweight='bold')
    
    # 통계량 계산
    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()
    skew_val = stats.skew(data)
    kurt_val = stats.kurtosis(data)
    
    # (1) 히스토그램 + KDE
    axes[0].hist(data, bins=bins, density=True, alpha=0.6, color='steelblue', edgecolor='black')
    
    # KDE 추가
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 200)
    axes[0].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    
    # 평균, 중앙값 선
    axes[0].axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    axes[0].axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    
    axes[0].set_xlabel(column, fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Histogram + KDE', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 통계량 텍스트
    if show_stats:
        stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}\nSkew: {skew_val:.2f}\nKurt: {kurt_val:.2f}'
        axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # (2) 박스플롯 + 스트립 플롯
    box = axes[1].boxplot(data, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2))
    
    # 개별 데이터 포인트 (샘플링)
    if len(data) <= 1000:
        axes[1].scatter(np.ones(len(data)) + np.random.normal(0, 0.04, len(data)),
                        data, alpha=0.3, s=10, color='darkblue')
    
    axes[1].set_ylabel(column, fontsize=12)
    axes[1].set_title('Box Plot', fontsize=12, fontweight='bold')
    axes[1].set_xticks([1])
    axes[1].set_xticklabels([column])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # (3) Q-Q Plot (정규성 검정)
    stats.probplot(data, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# 사용 예시
plot_distribution(df, 'age', bins=40, show_stats=True)
```

#### 3.1.2 다변수 분포 비교

```python
def plot_multiple_distributions(
    df: pd.DataFrame,
    columns: list,
    plot_type: str = 'hist',
    figsize: tuple = (16, 10)
) -> None:
    """
    여러 변수의 분포를 동시에 비교
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    columns : list
        비교할 컬럼 리스트
    plot_type : str, default 'hist'
        플롯 유형 ('hist', 'box', 'violin', 'kde')
    figsize : tuple
        그림 크기
    
    Examples:
    ---------
    >>> plot_multiple_distributions(df, ['age', 'income', 'credit_score'], plot_type='box')
    """
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3  # 3열 그리드
    
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten()
    
    fig.suptitle(f'Distribution Comparison ({plot_type.upper()})', fontsize=16, fontweight='bold')
    
    for idx, col in enumerate(columns):
        data = df[col].dropna()
        
        if plot_type == 'hist':
            axes[idx].hist(data, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
            axes[idx].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
            axes[idx].axvline(data.median(), color='green', linestyle='--', linewidth=2, label='Median')
            axes[idx].set_ylabel('Frequency')
            axes[idx].legend()
            
        elif plot_type == 'box':
            axes[idx].boxplot(data, vert=True, patch_artist=True,
                             boxprops=dict(facecolor='lightblue', alpha=0.7))
            
        elif plot_type == 'violin':
            parts = axes[idx].violinplot([data], positions=[1], showmeans=True, showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor('lightcoral')
                pc.set_alpha(0.7)
            
        elif plot_type == 'kde':
            data.plot(kind='kde', ax=axes[idx], linewidth=2, color='darkblue')
            axes[idx].fill_between(data.plot(kind='kde').get_lines()[0].get_xdata(),
                                   data.plot(kind='kde').get_lines()[0].get_ydata(),
                                   alpha=0.3)
            axes[idx].set_ylabel('Density')
        
        axes[idx].set_xlabel(col)
        axes[idx].set_title(col, fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    # 빈 서브플롯 제거
    for idx in range(n_cols, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()


# 사용 예시
numeric_cols = ['age', 'income', 'credit_score', 'debt_ratio']
plot_multiple_distributions(df, numeric_cols, plot_type='violin')
```

### 3.2 관계 시각화

#### 3.2.1 산점도 및 회귀선

```python
def plot_scatter_with_regression(
    df: pd.DataFrame,
    x_var: str,
    y_var: str,
    hue_var: str = None,
    add_regression: bool = True,
    figsize: tuple = (12, 8)
) -> None:
    """
    산점도 + 회귀선 + 신뢰구간
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    x_var : str
        X축 변수
    y_var : str
        Y축 변수
    hue_var : str, optional
        색상으로 구분할 범주형 변수
    add_regression : bool, default True
        회귀선 추가 여부
    figsize : tuple
        그림 크기
    
    Examples:
    ---------
    >>> plot_scatter_with_regression(df, 'age', 'income', hue_var='region')
    """
    plt.figure(figsize=figsize)
    
    if hue_var:
        # 그룹별 산점도
        sns.scatterplot(data=df, x=x_var, y=y_var, hue=hue_var, 
                        alpha=0.6, s=50, edgecolor='black', linewidth=0.5)
        
        if add_regression:
            # 그룹별 회귀선
            for category in df[hue_var].unique():
                subset = df[df[hue_var] == category]
                sns.regplot(data=subset, x=x_var, y=y_var, 
                            scatter=False, label=f'{category} trend')
    else:
        # 단일 산점도
        sns.scatterplot(data=df, x=x_var, y=y_var, 
                        alpha=0.6, s=50, color='steelblue', edgecolor='black', linewidth=0.5)
        
        if add_regression:
            # 회귀선 + 신뢰구간
            sns.regplot(data=df, x=x_var, y=y_var, 
                        scatter=False, color='red', line_kws={'linewidth': 2})
    
    # 상관계수 계산 및 표시
    valid_data = df[[x_var, y_var]].dropna()
    r, p = stats.pearsonr(valid_data[x_var], valid_data[y_var])
    
    plt.title(f'{y_var} vs {x_var}\nPearson r = {r:.3f}, p = {p:.4f}', 
              fontsize=14, fontweight='bold')
    plt.xlabel(x_var, fontsize=12)
    plt.ylabel(y_var, fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# 사용 예시
plot_scatter_with_regression(df, 'age', 'income', hue_var='region', add_regression=True)
```

#### 3.2.2 상관관계 히트맵

```python
def plot_correlation_heatmap(
    df: pd.DataFrame,
    method: str = 'pearson',
    annot: bool = True,
    figsize: tuple = (12, 10),
    cmap: str = 'coolwarm'
) -> None:
    """
    상관관계 히트맵 (고급 버전)
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    method : str, default 'pearson'
        상관계수 방법 ('pearson', 'spearman', 'kendall')
    annot : bool, default True
        숫자 표시 여부
    figsize : tuple
        그림 크기
    cmap : str, default 'coolwarm'
        색상 맵
    
    Examples:
    ---------
    >>> plot_correlation_heatmap(df, method='spearman')
    """
    # 수치형 변수만 선택
    numeric_df = df.select_dtypes(include=[np.number])
    
    # 상관관계 계산
    corr_matrix = numeric_df.corr(method=method)
    
    # 마스크 (상삼각형)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # 플롯
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=annot,
        fmt='.2f',
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={
            'label': f'{method.capitalize()} Correlation',
            'shrink': 0.8
        },
        vmin=-1,
        vmax=1
    )
    
    plt.title(f'Correlation Matrix ({method.upper()})', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# 사용 예시
plot_correlation_heatmap(df, method='pearson', annot=True)
```

### 3.3 비교 시각화

#### 3.3.1 그룹별 박스플롯/바이올린 플롯

```python
def plot_grouped_comparison(
    df: pd.DataFrame,
    numeric_var: str,
    category_var: str,
    plot_type: str = 'violin',
    figsize: tuple = (12, 6)
) -> None:
    """
    그룹별 수치형 변수 비교
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    numeric_var : str
        수치형 변수
    category_var : str
        범주형 변수 (그룹)
    plot_type : str, default 'violin'
        플롯 유형 ('box', 'violin', 'boxen', 'swarm')
    figsize : tuple
        그림 크기
    
    Examples:
    ---------
    >>> plot_grouped_comparison(df, 'income', 'region', plot_type='violin')
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'{numeric_var} by {category_var}', fontsize=16, fontweight='bold')
    
    # (1) 주요 플롯
    if plot_type == 'box':
        sns.boxplot(data=df, x=category_var, y=numeric_var, ax=axes[0], palette='Set2')
    elif plot_type == 'violin':
        sns.violinplot(data=df, x=category_var, y=numeric_var, ax=axes[0], palette='Set2')
    elif plot_type == 'boxen':
        sns.boxenplot(data=df, x=category_var, y=numeric_var, ax=axes[0], palette='Set2')
    elif plot_type == 'swarm':
        sns.swarmplot(data=df, x=category_var, y=numeric_var, ax=axes[0], palette='Set2', size=3)
    
    axes[0].set_title(f'{plot_type.capitalize()} Plot', fontsize=12, fontweight='bold')
    axes[0].set_xlabel(category_var)
    axes[0].set_ylabel(numeric_var)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # (2) 평균 + 신뢰구간
    group_means = df.groupby(category_var)[numeric_var].mean()
    group_sems = df.groupby(category_var)[numeric_var].sem()  # Standard Error of Mean
    
    axes[1].bar(range(len(group_means)), group_means.values, 
                yerr=1.96*group_sems.values, capsize=5, alpha=0.7, 
                color='steelblue', edgecolor='black')
    axes[1].set_xticks(range(len(group_means)))
    axes[1].set_xticklabels(group_means.index, rotation=45)
    axes[1].set_title('Mean + 95% CI', fontsize=12, fontweight='bold')
    axes[1].set_xlabel(category_var)
    axes[1].set_ylabel(f'Mean {numeric_var}')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


# 사용 예시
plot_grouped_comparison(df, 'income', 'region', plot_type='violin')
```

### 3.4 시계열 시각화

#### 3.4.1 시계열 패턴 플롯

```python
def plot_timeseries_patterns(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str = 'D',
    figsize: tuple = (16, 10)
) -> None:
    """
    시계열 데이터의 다양한 패턴 시각화
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    date_col : str
        날짜 컬럼
    value_col : str
        값 컬럼
    freq : str, default 'D'
        집계 빈도 ('D': 일, 'W': 주, 'M': 월)
    figsize : tuple
        그림 크기
    
    Examples:
    ---------
    >>> plot_timeseries_patterns(df, 'date', 'sales', freq='D')
    """
    # 날짜 인덱스 설정
    ts_data = df.set_index(pd.to_datetime(df[date_col]))[value_col]
    ts_data = ts_data.resample(freq).sum()
    
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    fig.suptitle(f'Time Series Analysis: {value_col}', fontsize=16, fontweight='bold')
    
    # (1) 원본 시계열
    axes[0].plot(ts_data.index, ts_data.values, linewidth=1.5, color='steelblue')
    axes[0].set_title('Raw Time Series', fontsize=12, fontweight='bold')
    axes[0].set_ylabel(value_col)
    axes[0].grid(True, alpha=0.3)
    
    # 이동 평균 추가
    rolling_mean = ts_data.rolling(window=7).mean()
    axes[0].plot(rolling_mean.index, rolling_mean.values, 
                 linewidth=2, color='red', label='7-period MA')
    axes[0].legend()
    
    # (2) 분해 (Decomposition)
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    try:
        decomposition = seasonal_decompose(ts_data, model='additive', period=7)
        
        axes[1].plot(decomposition.trend.index, decomposition.trend.values, 
                     linewidth=2, color='green')
        axes[1].set_title('Trend Component', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Trend')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values, 
                     linewidth=1.5, color='orange')
        axes[2].set_title('Seasonal Component', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Seasonal')
        axes[2].set_xlabel('Date')
        axes[2].grid(True, alpha=0.3)
    except Exception as e:
        print(f"분해 실패: {e}")
        axes[1].text(0.5, 0.5, 'Decomposition failed', ha='center', va='center', 
                     transform=axes[1].transAxes)
        axes[2].text(0.5, 0.5, 'Decomposition failed', ha='center', va='center', 
                     transform=axes[2].transAxes)
    
    plt.tight_layout()
    plt.show()


# 사용 예시 (샘플 데이터)
date_range = pd.date_range('2023-01-01', '2024-12-31', freq='D')
sample_ts = pd.DataFrame({
    'date': date_range,
    'sales': np.random.normal(1000, 200, len(date_range)) + 
             100 * np.sin(np.arange(len(date_range)) * 2 * np.pi / 365)
})

plot_timeseries_patterns(sample_ts, 'date', 'sales', freq='D')
```

### 3.5 고급 시각화

#### 3.5.1 페어플롯 (Pair Plot)

```python
def plot_pairplot_advanced(
    df: pd.DataFrame,
    vars: list = None,
    hue: str = None,
    figsize: tuple = (14, 14)
) -> None:
    """
    고급 페어플롯 (변수 쌍별 관계)
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    vars : list, optional
        분석할 변수 리스트 (None이면 모든 수치형 변수)
    hue : str, optional
        색상으로 구분할 범주형 변수
    figsize : tuple
        그림 크기
    
    Examples:
    ---------
    >>> plot_pairplot_advanced(df, vars=['age', 'income', 'credit_score'], hue='region')
    """
    if vars is None:
        vars = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Seaborn pairplot
    pairplot = sns.pairplot(
        df[vars + ([hue] if hue else [])],
        hue=hue,
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'black', 'linewidth': 0.5},
        diag_kws={'alpha': 0.7, 'linewidth': 2},
        corner=False,
        height=figsize[0] / len(vars)
    )
    
    pairplot.fig.suptitle('Pair Plot: Variable Relationships', 
                          fontsize=16, fontweight='bold', y=1.01)
    plt.show()


# 사용 예시
plot_pairplot_advanced(df, vars=['age', 'income', 'credit_score'], hue='region')
```

#### 3.5.2 결측값 히트맵

```python
def plot_missing_values_heatmap(
    df: pd.DataFrame,
    figsize: tuple = (12, 8)
) -> None:
    """
    결측값 패턴 시각화
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    figsize : tuple
        그림 크기
    
    Examples:
    ---------
    >>> plot_missing_values_heatmap(df)
    """
    # 결측값 매트릭스
    missing_matrix = df.isnull().astype(int)
    
    # 결측값이 있는 컬럼만 선택
    missing_cols = missing_matrix.sum()[missing_matrix.sum() > 0].index
    
    if len(missing_cols) == 0:
        print("결측값이 없습니다.")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle('Missing Values Analysis', fontsize=16, fontweight='bold')
    
    # (1) 히트맵
    sns.heatmap(
        missing_matrix[missing_cols].T,
        cbar=True,
        cmap='YlOrRd',
        yticklabels=True,
        xticklabels=False,
        ax=axes[0],
        cbar_kws={'label': '1 = Missing, 0 = Present'}
    )
    axes[0].set_title('Missing Values Heatmap', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Variables')
    
    # (2) 막대 그래프 (결측 비율)
    missing_pct = (df[missing_cols].isnull().sum() / len(df)) * 100
    missing_pct = missing_pct.sort_values(ascending=False)
    
    axes[1].barh(range(len(missing_pct)), missing_pct.values, color='coral', edgecolor='black')
    axes[1].set_yticks(range(len(missing_pct)))
    axes[1].set_yticklabels(missing_pct.index)
    axes[1].set_xlabel('Missing Percentage (%)')
    axes[1].set_title('Missing Values Percentage', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # 값 레이블 추가
    for i, v in enumerate(missing_pct.values):
        axes[1].text(v + 0.5, i, f'{v:.1f}%', va='center')
    
    plt.tight_layout()
    plt.show()


# 사용 예시
plot_missing_values_heatmap(df)
```

### 3.6 통합 시각화 대시보드

```python
def create_eda_dashboard(
    df: pd.DataFrame,
    target_var: str = None,
    output_file: str = None
) -> None:
    """
    EDA 종합 대시보드 생성
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    target_var : str, optional
        타겟 변수
    output_file : str, optional
        저장할 파일명 (None이면 화면 표시만)
    
    Examples:
    ---------
    >>> create_eda_dashboard(df, target_var='churn', output_file='eda_dashboard.png')
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('EDA Dashboard', fontsize=20, fontweight='bold')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # (1,1) 데이터 개요
    ax1 = fig.add_subplot(gs[0, 0])
    overview_text = f"""
    Dataset Overview:
    -----------------
    Rows: {len(df):,}
    Columns: {len(df.columns)}
    Numeric: {len(numeric_cols)}
    Categorical: {len(cat_cols)}
    Missing: {df.isnull().sum().sum():,}
    Duplicates: {df.duplicated().sum():,}
    Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
    """
    ax1.text(0.1, 0.5, overview_text, fontsize=11, family='monospace',
             verticalalignment='center')
    ax1.axis('off')
    
    # (1,2) 결측값 막대 그래프
    ax2 = fig.add_subplot(gs[0, 1])
    missing = df.isnull().sum().sort_values(ascending=False)[:10]
    if len(missing[missing > 0]) > 0:
        missing[missing > 0].plot(kind='barh', ax=ax2, color='coral')
        ax2.set_title('Top 10 Missing Values', fontweight='bold')
        ax2.set_xlabel('Count')
    else:
        ax2.text(0.5, 0.5, 'No Missing Values', ha='center', va='center')
        ax2.axis('off')
    
    # (1,3) 데이터 타입 파이 차트
    ax3 = fig.add_subplot(gs[0, 2])
    dtype_counts = df.dtypes.value_counts()
    ax3.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', 
            startangle=90, colors=plt.cm.Set3(range(len(dtype_counts))))
    ax3.set_title('Data Types Distribution', fontweight='bold')
    
    # (2,1-3) 상관관계 히트맵
    if len(numeric_cols) >= 2:
        ax4 = fig.add_subplot(gs[1, :])
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=len(numeric_cols) <= 10, fmt='.2f', 
                    cmap='coolwarm', center=0, ax=ax4, square=True)
        ax4.set_title('Correlation Heatmap', fontweight='bold')
    
    # (3,1) 수치형 변수 분포 (첫 번째)
    if len(numeric_cols) >= 1:
        ax5 = fig.add_subplot(gs[2, 0])
        df[numeric_cols[0]].hist(bins=30, ax=ax5, color='steelblue', edgecolor='black')
        ax5.set_title(f'Distribution: {numeric_cols[0]}', fontweight='bold')
        ax5.set_xlabel(numeric_cols[0])
        ax5.set_ylabel('Frequency')
    
    # (3,2) 범주형 변수 막대 (첫 번째)
    if len(cat_cols) >= 1:
        ax6 = fig.add_subplot(gs[2, 1])
        top_cats = df[cat_cols[0]].value_counts()[:10]
        top_cats.plot(kind='bar', ax=ax6, color='lightgreen', edgecolor='black')
        ax6.set_title(f'Top Categories: {cat_cols[0]}', fontweight='bold')
        ax6.set_xlabel(cat_cols[0])
        ax6.set_ylabel('Count')
        ax6.tick_params(axis='x', rotation=45)
    
    # (3,3) 타겟 변수 분포 (있는 경우)
    if target_var and target_var in df.columns:
        ax7 = fig.add_subplot(gs[2, 2])
        df[target_var].value_counts().plot(kind='pie', ax=ax7, autopct='%1.1f%%', 
                                           colors=plt.cm.Pastel1(range(df[target_var].nunique())))
        ax7.set_title(f'Target: {target_var}', fontweight='bold')
        ax7.set_ylabel('')
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Dashboard saved: {output_file}")
    
    plt.show()


# 사용 예시
create_eda_dashboard(df, target_var='churn', output_file='eda_dashboard.png')
```

---

## 4. 예시 (Examples)

### 4.1 완전한 시각화 워크플로우

```python
import pandas as pd
import numpy as np

# 샘플 데이터 생성
np.random.seed(42)
n = 2000

df = pd.DataFrame({
    'age': np.random.normal(40, 12, n).clip(18, 80).astype(int),
    'income': np.random.lognormal(10.5, 0.6, n),
    'credit_score': np.random.normal(700, 50, n).clip(300, 850).astype(int),
    'debt_ratio': np.random.beta(2, 5, n),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n),
    'product': np.random.choice(['A', 'B', 'C'], n),
    'churn': np.random.choice([0, 1], n, p=[0.75, 0.25])
})

# 결측값 추가
df.loc[np.random.choice(df.index, 100), 'income'] = np.nan
df.loc[np.random.choice(df.index, 50), 'credit_score'] = np.nan

print("=" * 80)
print("EDA 시각화 워크플로우 시작")
print("=" * 80)

# 1. 종합 대시보드
print("\n[1/6] Creating EDA Dashboard...")
create_eda_dashboard(df, target_var='churn', output_file='dashboard.png')

# 2. 분포 분석
print("\n[2/6] Distribution Analysis...")
plot_distribution(df, 'income', bins=50)

# 3. 상관관계 분석
print("\n[3/6] Correlation Analysis...")
plot_correlation_heatmap(df, method='pearson')

# 4. 그룹 비교
print("\n[4/6] Group Comparison...")
plot_grouped_comparison(df, 'income', 'region', plot_type='violin')

# 5. 산점도 분석
print("\n[5/6] Scatter Plot Analysis...")
plot_scatter_with_regression(df, 'age', 'income', hue_var='region')

# 6. 결측값 분석
print("\n[6/6] Missing Values Analysis...")
plot_missing_values_heatmap(df)

print("\n" + "=" * 80)
print("시각화 완료!")
print("=" * 80)
```

---

## 5. 에이전트 매핑 (Agent Mapping)

**Primary Agent**: `data-visualization-specialist`
- 역할: 모든 시각화 생성 및 디자인
- 책임: 차트 선택, 색상 설정, 레이아웃 최적화

**Supporting Agent**: `data-scientist`
- 역할: 통계적 해석 및 인사이트 도출

---

## 6. 필요 라이브러리 (Required Libraries)

```bash
# 기본 시각화
uv pip install matplotlib==3.8.2 seaborn==0.13.1

# 통계
uv pip install scipy==1.12.0 statsmodels==0.14.1

# 데이터 처리
uv pip install pandas==2.2.0 numpy==1.26.3

# 인터랙티브 시각화 (선택)
uv pip install plotly==5.18.0 bokeh==3.3.4
```

---

## 7. 체크포인트 (Checkpoints)

- [ ] 모든 주요 변수에 대한 시각화 생성
- [ ] 적절한 차트 유형 선택
- [ ] 레이블, 제목, 범례 명확히 표시
- [ ] 색상 팔레트 일관성 유지
- [ ] 고해상도 이미지 저장 (DPI 300+)

---

## 8. 트러블슈팅 (Troubleshooting)

**문제 1: 한글 폰트 깨짐**
```python
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleGothic'  # Mac
# or
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False
```

**문제 2: 플롯이 너무 작거나 잘림**
```python
plt.tight_layout()
plt.savefig('plot.png', bbox_inches='tight', dpi=300)
```

---

## 9. 참고 자료 (References)

1. **Matplotlib**: https://matplotlib.org/
2. **Seaborn**: https://seaborn.pydata.org/
3. **Edward Tufte's Principles**: "The Visual Display of Quantitative Information"
4. **ColorBrewer**: https://colorbrewer2.org/

---

**문서 끝**

This completes the EDA reference documentation suite.
