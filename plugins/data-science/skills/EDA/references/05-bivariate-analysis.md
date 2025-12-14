# Reference 05: Bivariate Analysis

**Version**: 1.0  
**Last Updated**: 2025-01-25  
**Workflow Phase**: Phase 3.2 - Descriptive Statistics (Bivariate)  
**Estimated Reading Time**: 22-25 minutes

---

## 1. 개요 (Overview)

### 1.1 목적 (Purpose)

이변량 분석(Bivariate Analysis)은 두 변수 간의 관계를 파악하는 프로세스입니다. 단변량 분석에서 개별 변수의 특성을 이해했다면, 이변량 분석에서는 변수들이 어떻게 상호작용하는지 이해합니다.

**주요 목적**:
- 변수 간 상관관계 정량화
- 인과관계 가설 수립
- 예측 변수 식별
- 다중공선성 탐지
- 상호작용 효과 발견

### 1.2 적용 시기 (When to Apply)

1. **Feature Selection 전**: 타겟 변수와 관련 없는 변수 제거
2. **모델링 전**: 다중공선성 문제 해결
3. **가설 검정**: 변수 간 관계의 통계적 유의성 확인
4. **세그먼트 분석**: 그룹별 차이 검증
5. **상호작용 피처 생성**: 변수 조합 효과 탐색

### 1.3 분석 유형

| 변수 1 타입 | 변수 2 타입 | 분석 방법 | 시각화 |
|------------|------------|-----------|--------|
| 수치형 | 수치형 | 상관계수, 회귀분석 | 산점도, 히트맵 |
| 수치형 | 범주형 | 그룹 통계, ANOVA, t-test | 박스플롯, 바이올린 플롯 |
| 범주형 | 범주형 | 교차표, 카이제곱 검정 | 히트맵, 모자이크 플롯 |

---

## 2. 이론적 배경 (Theoretical Background)

### 2.1 상관관계 (Correlation)

**Pearson 상관계수 (Pearson's r)**:
- 선형 관계의 강도와 방향
- 범위: -1 (완전 음의 상관) ~ +1 (완전 양의 상관)
- 가정: 정규분포, 선형관계, 등분산성

**Spearman 순위 상관계수**:
- 단조 관계 (monotonic relationship)
- 비모수적 방법
- 이상치에 강건

**Kendall의 타우 (Kendall's τ)**:
- 순위 기반 상관계수
- 표본 크기가 작을 때 유용
- Spearman보다 계산 비용 높음

**상관계수 해석 기준** (Cohen, 1988):
- 0.00 - 0.19: 매우 약한 상관
- 0.20 - 0.39: 약한 상관
- 0.40 - 0.59: 중간 상관
- 0.60 - 0.79: 강한 상관
- 0.80 - 1.00: 매우 강한 상관

### 2.2 인과관계 vs 상관관계

**주의사항**: "Correlation does not imply causation"

상관관계가 있다고 해서 인과관계가 있는 것은 아닙니다:
- **교란 변수 (Confounding variable)**: 제3의 변수가 두 변수에 모두 영향
- **역인과관계 (Reverse causation)**: 원인과 결과의 방향이 반대
- **우연 (Spurious correlation)**: 통계적 우연

### 2.3 통계적 검정

**t-test (두 그룹 평균 비교)**:
- 독립표본 t-test: 서로 다른 그룹
- 대응표본 t-test: 동일 대상의 전후 비교

**ANOVA (Analysis of Variance)**:
- 세 개 이상 그룹의 평균 비교
- F-통계량 사용
- 사후 검정 (Post-hoc): Tukey HSD, Bonferroni

**카이제곱 검정 (Chi-square test)**:
- 범주형 변수 간 독립성 검정
- 교차표 (Contingency table) 사용

---

## 3. 구현 (Implementation)

### 3.1 수치형 vs 수치형 변수 분석

#### 3.1.1 상관관계 분석

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def analyze_numeric_correlation(
    df: pd.DataFrame,
    var1: str,
    var2: str,
    show_plot: bool = True
) -> dict:
    """
    두 수치형 변수 간 상관관계 분석
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    var1 : str
        첫 번째 수치형 변수
    var2 : str
        두 번째 수치형 변수
    show_plot : bool, default True
        산점도 표시 여부
    
    Returns:
    --------
    dict
        상관관계 분석 결과
    
    Examples:
    ---------
    >>> result = analyze_numeric_correlation(df, 'age', 'income')
    >>> print(f"Pearson r: {result['pearson_r']:.3f}")
    """
    # 유효한 데이터만 추출 (결측값 제거)
    data = df[[var1, var2]].dropna()
    
    if len(data) < 3:
        raise ValueError("유효한 데이터가 3개 미만입니다.")
    
    print("=" * 80)
    print(f"수치형 변수 상관관계 분석: {var1} vs {var2}")
    print("=" * 80)
    
    # 1. 기본 정보
    print(f"\n[기본 정보]")
    print(f"  총 관측치: {len(df):,}")
    print(f"  유효 관측치: {len(data):,}")
    print(f"  결측값 제외: {len(df) - len(data):,}")
    
    # 2. Pearson 상관계수 (선형 상관)
    pearson_r, pearson_p = stats.pearsonr(data[var1], data[var2])
    
    print(f"\n[Pearson 상관계수 (선형 관계)]")
    print(f"  r = {pearson_r:.4f}")
    print(f"  p-value = {pearson_p:.6f}")
    
    if abs(pearson_r) < 0.2:
        strength = "매우 약함"
    elif abs(pearson_r) < 0.4:
        strength = "약함"
    elif abs(pearson_r) < 0.6:
        strength = "중간"
    elif abs(pearson_r) < 0.8:
        strength = "강함"
    else:
        strength = "매우 강함"
    
    direction = "양의 상관" if pearson_r > 0 else "음의 상관"
    print(f"  해석: {strength} {direction}")
    print(f"  유의성: {'유의함 (p < 0.05)' if pearson_p < 0.05 else '유의하지 않음 (p >= 0.05)'}")
    
    # 3. Spearman 상관계수 (단조 관계)
    spearman_r, spearman_p = stats.spearmanr(data[var1], data[var2])
    
    print(f"\n[Spearman 상관계수 (단조 관계)]")
    print(f"  ρ = {spearman_r:.4f}")
    print(f"  p-value = {spearman_p:.6f}")
    print(f"  유의성: {'유의함 (p < 0.05)' if spearman_p < 0.05 else '유의하지 않음 (p >= 0.05)'}")
    
    # 4. Kendall's Tau
    kendall_tau, kendall_p = stats.kendalltau(data[var1], data[var2])
    
    print(f"\n[Kendall's Tau]")
    print(f"  τ = {kendall_tau:.4f}")
    print(f"  p-value = {kendall_p:.6f}")
    
    # 5. 선형 회귀
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(data[var1], data[var2])
    
    print(f"\n[선형 회귀]")
    print(f"  회귀식: {var2} = {slope:.4f} * {var1} + {intercept:.4f}")
    print(f"  R² = {r_value**2:.4f}")
    print(f"  표준오차 = {std_err:.4f}")
    
    # 6. 시각화
    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'상관관계 분석: {var1} vs {var2}', fontsize=16, fontweight='bold')
        
        # (1) 산점도 + 회귀선
        axes[0].scatter(data[var1], data[var2], alpha=0.5, s=30)
        
        # 회귀선
        x_line = np.linspace(data[var1].min(), data[var1].max(), 100)
        y_line = slope * x_line + intercept
        axes[0].plot(x_line, y_line, 'r-', linewidth=2, label=f'y = {slope:.2f}x + {intercept:.2f}')
        
        axes[0].set_xlabel(var1, fontsize=12)
        axes[0].set_ylabel(var2, fontsize=12)
        axes[0].set_title(f'산점도 + 회귀선 (r = {pearson_r:.3f})', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # (2) Hexbin plot (밀도)
        axes[1].hexbin(data[var1], data[var2], gridsize=30, cmap='YlOrRd', mincnt=1)
        axes[1].set_xlabel(var1, fontsize=12)
        axes[1].set_ylabel(var2, fontsize=12)
        axes[1].set_title('밀도 플롯 (Hexbin)', fontsize=12, fontweight='bold')
        plt.colorbar(axes[1].collections[0], ax=axes[1], label='Count')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # 결과 반환
    return {
        'n_valid': len(data),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'kendall_tau': kendall_tau,
        'kendall_p': kendall_p,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2
    }


# 사용 예시
result = analyze_numeric_correlation(df, 'age', 'income', show_plot=True)
```

#### 3.1.2 상관관계 매트릭스 (전체 변수)

```python
def correlation_matrix_analysis(
    df: pd.DataFrame,
    method: str = 'pearson',
    threshold: float = 0.7,
    figsize: tuple = (12, 10)
) -> pd.DataFrame:
    """
    모든 수치형 변수 간 상관관계 매트릭스 생성
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    method : str, default 'pearson'
        상관계수 방법 ('pearson', 'spearman', 'kendall')
    threshold : float, default 0.7
        높은 상관관계 경고 임계값
    figsize : tuple, default (12, 10)
        그림 크기
    
    Returns:
    --------
    pd.DataFrame
        상관관계 매트릭스
    
    Examples:
    ---------
    >>> corr_matrix = correlation_matrix_analysis(df, method='pearson')
    """
    # 수치형 변수만 선택
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        raise ValueError("수치형 변수가 2개 미만입니다.")
    
    print("=" * 80)
    print(f"상관관계 매트릭스 분석 ({method.upper()})")
    print("=" * 80)
    print(f"변수 개수: {numeric_df.shape[1]}")
    
    # 상관관계 매트릭스 계산
    corr_matrix = numeric_df.corr(method=method)
    
    # 높은 상관관계 쌍 찾기
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                high_corr_pairs.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    # 높은 상관관계 출력
    if len(high_corr_pairs) > 0:
        print(f"\n[높은 상관관계 쌍 (|r| >= {threshold})]")
        for pair in high_corr_pairs:
            print(f"  {pair['var1']} <-> {pair['var2']}: {pair['correlation']:.4f}")
        
        print(f"\n⚠ 다중공선성 주의: {len(high_corr_pairs)}개 쌍 발견")
    else:
        print(f"\n✓ 높은 상관관계 (|r| >= {threshold}) 없음")
    
    # 히트맵 시각화
    plt.figure(figsize=figsize)
    
    # 마스크 생성 (상삼각형 제거)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # 히트맵
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': f'{method.capitalize()} Correlation'}
    )
    
    plt.title(f'상관관계 매트릭스 ({method.upper()})', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    return corr_matrix


# 사용 예시
corr_matrix = correlation_matrix_analysis(df, method='pearson', threshold=0.7)

# Spearman (단조 관계)
corr_matrix_spearman = correlation_matrix_analysis(df, method='spearman', threshold=0.7)
```

### 3.2 수치형 vs 범주형 변수 분석

#### 3.2.1 그룹별 비교 분석

```python
def analyze_numeric_by_category(
    df: pd.DataFrame,
    numeric_var: str,
    category_var: str,
    test_type: str = 'auto',
    show_plot: bool = True
) -> dict:
    """
    범주별 수치형 변수 비교 분석
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    numeric_var : str
        수치형 변수
    category_var : str
        범주형 변수
    test_type : str, default 'auto'
        통계 검정 방법 ('ttest', 'anova', 'kruskal', 'auto')
    show_plot : bool, default True
        시각화 표시 여부
    
    Returns:
    --------
    dict
        분석 결과
    
    Examples:
    ---------
    >>> result = analyze_numeric_by_category(df, 'income', 'region')
    """
    # 유효 데이터
    data = df[[numeric_var, category_var]].dropna()
    
    print("=" * 80)
    print(f"수치형 vs 범주형 분석: {numeric_var} by {category_var}")
    print("=" * 80)
    
    # 1. 기본 정보
    n_categories = data[category_var].nunique()
    print(f"\n[기본 정보]")
    print(f"  유효 관측치: {len(data):,}")
    print(f"  카테고리 개수: {n_categories}")
    
    # 2. 그룹별 기술 통계
    group_stats = data.groupby(category_var)[numeric_var].agg([
        ('평균', 'mean'),
        ('중앙값', 'median'),
        ('표준편차', 'std'),
        ('최소', 'min'),
        ('최대', 'max'),
        ('개수', 'count')
    ])
    
    print(f"\n[그룹별 통계]")
    print(group_stats.to_string())
    
    # 3. 통계 검정
    groups = [data[data[category_var] == cat][numeric_var].values 
              for cat in data[category_var].unique()]
    
    # 자동 선택
    if test_type == 'auto':
        if n_categories == 2:
            test_type = 'ttest'
        else:
            test_type = 'anova'
    
    print(f"\n[통계 검정: {test_type.upper()}]")
    
    if test_type == 'ttest' and n_categories == 2:
        # 독립표본 t-test
        t_stat, p_value = stats.ttest_ind(*groups)
        print(f"  t-통계량: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  결론: {'그룹 간 평균 차이 유의함' if p_value < 0.05 else '그룹 간 평균 차이 없음'}")
        
        # 효과 크기 (Cohen's d)
        mean1, mean2 = groups[0].mean(), groups[1].mean()
        std_pooled = np.sqrt((groups[0].var() + groups[1].var()) / 2)
        cohens_d = (mean1 - mean2) / std_pooled
        print(f"  Cohen's d: {cohens_d:.4f}")
        
        result = {'test': 'ttest', 't_stat': t_stat, 'p_value': p_value, 'cohens_d': cohens_d}
        
    elif test_type == 'anova':
        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"  F-통계량: {f_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  결론: {'그룹 간 평균 차이 유의함' if p_value < 0.05 else '그룹 간 평균 차이 없음'}")
        
        # 사후 검정 (Tukey HSD)
        if p_value < 0.05:
            print(f"\n  [사후 검정 필요: Tukey HSD 권장]")
        
        result = {'test': 'anova', 'f_stat': f_stat, 'p_value': p_value}
        
    elif test_type == 'kruskal':
        # Kruskal-Wallis (비모수)
        h_stat, p_value = stats.kruskal(*groups)
        print(f"  H-통계량: {h_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  결론: {'그룹 간 차이 유의함' if p_value < 0.05 else '그룹 간 차이 없음'}")
        
        result = {'test': 'kruskal', 'h_stat': h_stat, 'p_value': p_value}
    
    # 4. 시각화
    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{numeric_var} by {category_var}', fontsize=16, fontweight='bold')
        
        # (1) 박스플롯
        data.boxplot(column=numeric_var, by=category_var, ax=axes[0])
        axes[0].set_title('박스플롯', fontsize=12, fontweight='bold')
        axes[0].set_xlabel(category_var)
        axes[0].set_ylabel(numeric_var)
        plt.sca(axes[0])
        plt.xticks(rotation=45)
        
        # (2) 바이올린 플롯
        sns.violinplot(data=data, x=category_var, y=numeric_var, ax=axes[1])
        axes[1].set_title('바이올린 플롯', fontsize=12, fontweight='bold')
        axes[1].set_xlabel(category_var)
        axes[1].set_ylabel(numeric_var)
        axes[1].tick_params(axis='x', rotation=45)
        
        # (3) 막대 그래프 (평균 + 오차막대)
        group_means = data.groupby(category_var)[numeric_var].mean()
        group_stds = data.groupby(category_var)[numeric_var].std()
        
        axes[2].bar(range(len(group_means)), group_means.values, 
                    yerr=group_stds.values, capsize=5, alpha=0.7, edgecolor='black')
        axes[2].set_xticks(range(len(group_means)))
        axes[2].set_xticklabels(group_means.index, rotation=45)
        axes[2].set_title('평균 + 표준편차', fontsize=12, fontweight='bold')
        axes[2].set_xlabel(category_var)
        axes[2].set_ylabel(f'평균 {numeric_var}')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    result.update({
        'n_categories': n_categories,
        'group_stats': group_stats.to_dict()
    })
    
    return result


# 사용 예시
result = analyze_numeric_by_category(df, 'income', 'region', test_type='auto')
```

### 3.3 범주형 vs 범주형 변수 분석

#### 3.3.1 교차표 및 카이제곱 검정

```python
def analyze_categorical_association(
    df: pd.DataFrame,
    var1: str,
    var2: str,
    show_plot: bool = True
) -> dict:
    """
    두 범주형 변수 간 연관성 분석
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    var1 : str
        첫 번째 범주형 변수
    var2 : str
        두 번째 범주형 변수
    show_plot : bool, default True
        시각화 표시 여부
    
    Returns:
    --------
    dict
        분석 결과
    
    Examples:
    ---------
    >>> result = analyze_categorical_association(df, 'region', 'product_type')
    """
    # 유효 데이터
    data = df[[var1, var2]].dropna()
    
    print("=" * 80)
    print(f"범주형 변수 연관성 분석: {var1} vs {var2}")
    print("=" * 80)
    
    # 1. 교차표 (Contingency Table)
    crosstab = pd.crosstab(data[var1], data[var2])
    
    print(f"\n[교차표 (빈도)]")
    print(crosstab.to_string())
    
    # 비율 (행 기준)
    crosstab_pct = pd.crosstab(data[var1], data[var2], normalize='index') * 100
    
    print(f"\n[교차표 (행 기준 비율, %)]")
    print(crosstab_pct.to_string())
    
    # 2. 카이제곱 검정
    chi2, p_value, dof, expected = stats.chi2_contingency(crosstab)
    
    print(f"\n[카이제곱 검정]")
    print(f"  카이제곱 통계량: {chi2:.4f}")
    print(f"  자유도: {dof}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  결론: {'두 변수는 독립적이지 않음 (연관 있음)' if p_value < 0.05 else '두 변수는 독립적 (연관 없음)'}")
    
    # 3. Cramér's V (효과 크기)
    n = len(data)
    min_dim = min(crosstab.shape[0], crosstab.shape[1]) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    
    print(f"\n[Cramér's V (효과 크기)]")
    print(f"  V = {cramers_v:.4f}")
    
    if cramers_v < 0.1:
        strength = "매우 약함"
    elif cramers_v < 0.3:
        strength = "약함"
    elif cramers_v < 0.5:
        strength = "중간"
    else:
        strength = "강함"
    
    print(f"  해석: {strength} 연관성")
    
    # 4. 시각화
    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{var1} vs {var2}', fontsize=16, fontweight='bold')
        
        # (1) 히트맵 (빈도)
        sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlGnBu', ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title('교차표 히트맵 (빈도)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel(var2)
        axes[0].set_ylabel(var1)
        
        # (2) 스택 막대 그래프 (비율)
        crosstab_pct.plot(kind='bar', stacked=True, ax=axes[1], colormap='Set3', edgecolor='black')
        axes[1].set_title('스택 막대 그래프 (비율)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel(var1)
        axes[1].set_ylabel('비율 (%)')
        axes[1].legend(title=var2, bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    return {
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'cramers_v': cramers_v,
        'crosstab': crosstab.to_dict()
    }


# 사용 예시
result = analyze_categorical_association(df, 'region', 'product_type')
```

### 3.4 통합 이변량 분석

```python
def comprehensive_bivariate_analysis(
    df: pd.DataFrame,
    target_var: str = None,
    output_dir: str = "bivariate_analysis"
) -> dict:
    """
    모든 변수 쌍에 대한 포괄적 이변량 분석
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    target_var : str, optional
        타겟 변수 (지정 시 타겟 중심 분석)
    output_dir : str
        출력 디렉토리
    
    Returns:
    --------
    dict
        분석 결과
    
    Examples:
    ---------
    >>> results = comprehensive_bivariate_analysis(df, target_var='churn')
    """
    from pathlib import Path
    import time
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("포괄적 이변량 분석")
    print("=" * 80)
    print(f"데이터 크기: {df.shape}")
    print(f"타겟 변수: {target_var if target_var else '없음 (모든 쌍 분석)'}")
    print("=" * 80)
    
    start_time = time.time()
    
    results = {}
    
    # 1. 수치형-수치형 상관관계
    print("\n[1/3] 수치형 변수 간 상관관계...")
    corr_matrix = correlation_matrix_analysis(df, method='pearson', threshold=0.7)
    results['correlation_matrix'] = corr_matrix
    corr_matrix.to_csv(f"{output_dir}/correlation_matrix.csv")
    print(f"  ✓ 저장: {output_dir}/correlation_matrix.csv")
    
    # 2. 타겟 변수 중심 분석 (타겟이 있는 경우)
    if target_var and target_var in df.columns:
        print(f"\n[2/3] 타겟 변수 ({target_var}) 중심 분석...")
        
        target_dtype = df[target_var].dtype
        
        # 수치형 변수들과의 상관관계
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != target_var]
        
        if len(numeric_cols) > 0 and target_dtype in [np.int64, np.float64]:
            target_corr = df[numeric_cols + [target_var]].corr()[target_var].drop(target_var)
            target_corr = target_corr.sort_values(ascending=False)
            
            print(f"\n  타겟과의 상관관계 (상위 10개):")
            for var, corr in target_corr.head(10).items():
                print(f"    {var}: {corr:.4f}")
            
            target_corr.to_csv(f"{output_dir}/target_correlations.csv")
            print(f"  ✓ 저장: {output_dir}/target_correlations.csv")
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("이변량 분석 완료")
    print("=" * 80)
    print(f"총 소요 시간: {elapsed:.2f}초")
    
    return results


# 사용 예시
results = comprehensive_bivariate_analysis(df, target_var='churn', output_dir="bivariate_reports")
```

---

## 4. 예시 (Examples)

### 4.1 완전한 워크플로우

```python
import pandas as pd
import numpy as np

# 샘플 데이터
np.random.seed(42)
n = 3000

df = pd.DataFrame({
    'age': np.random.normal(40, 12, n).clip(18, 80).astype(int),
    'income': np.random.lognormal(10.5, 0.6, n),
    'credit_score': np.random.normal(700, 50, n).clip(300, 850).astype(int),
    'debt_ratio': np.random.beta(2, 5, n),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n),
    'product': np.random.choice(['A', 'B', 'C'], n),
    'churn': np.random.choice([0, 1], n, p=[0.75, 0.25])
})

# 포괄적 이변량 분석
results = comprehensive_bivariate_analysis(df, target_var='churn')
```

---

## 5. 에이전트 매핑 (Agent Mapping)

**Primary Agent**: `data-scientist`
**Supporting Agent**: `data-visualization-specialist`

---

## 6. 필요 라이브러리 (Required Libraries)

```bash
uv pip install pandas==2.2.0 numpy==1.26.3 scipy==1.12.0
uv pip install matplotlib==3.8.2 seaborn==0.13.1
```

---

## 7. 체크포인트 (Checkpoints)

- [ ] 상관관계 매트릭스 생성 및 분석
- [ ] 다중공선성 확인 (|r| > 0.7)
- [ ] 타겟 변수와 피처 간 관계 파악
- [ ] 통계적 유의성 검정 완료

---

## 8. 트러블슈팅 (Troubleshooting)

**문제 1: 비선형 관계가 의심됨**
- 해결: Spearman 상관계수 사용, 산점도 육안 확인

**문제 2: 다중공선성 심각 (r > 0.9)**
- 해결: VIF 계산, 주성분 분석 고려

---

## 9. 참고 자료 (References)

1. **SciPy Stats**: https://docs.scipy.org/doc/scipy/reference/stats.html
2. **Seaborn**: https://seaborn.pydata.org/

---

**문서 끝**

다음 단계: [07-visualization-patterns.md](./07-visualization-patterns.md) - 시각화 패턴
