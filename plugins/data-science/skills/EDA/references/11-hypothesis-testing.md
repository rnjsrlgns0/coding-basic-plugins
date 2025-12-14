# 11. Hypothesis Testing (가설 검정)

## 1. 개요

### 1.1 목적
가설 검정(Hypothesis Testing)은 데이터를 기반으로 모집단에 대한 가설이 통계적으로 유의미한지를 판단하는 추론 기법입니다. A/B 테스트, 그룹 간 차이 검증, 효과성 검증 등에 필수적으로 사용됩니다.

### 1.2 적용 시기
- 두 그룹 간 차이가 통계적으로 유의미한지 확인할 때
- A/B 테스트 결과 분석 (신규 기능 효과 검증)
- 정책/전략 변경 전후 비교
- 제품/서비스 개선 효과 측정
- 과학적 주장의 통계적 근거 제시

### 1.3 주요 검정 방법

**모수 검정 (Parametric Tests)**:
- **t-test**: 평균 비교 (2그룹)
- **ANOVA**: 평균 비교 (3그룹 이상)
- **Z-test**: 대표본 평균 비교
- **F-test**: 분산 비교

**비모수 검정 (Non-parametric Tests)**:
- **Mann-Whitney U**: t-test 대체 (비정규)
- **Kruskal-Wallis H**: ANOVA 대체 (비정규)
- **Wilcoxon Signed-Rank**: 대응표본 t-test 대체
- **Chi-square**: 범주형 변수 독립성 검정

---

## 2. 이론적 배경

### 2.1 가설 검정의 기본 개념

#### 귀무가설 vs 대립가설
```
귀무가설 (H₀): "차이가 없다", "효과가 없다"
대립가설 (H₁): "차이가 있다", "효과가 있다"

예시 1: 신약 효과 검증
H₀: 신약과 위약의 효과가 같다 (μ₁ = μ₂)
H₁: 신약의 효과가 위약보다 크다 (μ₁ > μ₂)

예시 2: 웹사이트 A/B 테스트
H₀: 신규 디자인과 기존 디자인의 전환율이 같다
H₁: 신규 디자인의 전환율이 더 높다
```

#### P-value (유의확률)
```
정의: 귀무가설이 참일 때, 관측된 결과(또는 더 극단적인 결과)가 나올 확률

해석:
- p < 0.05: 귀무가설 기각 (통계적으로 유의미한 차이)
- p ≥ 0.05: 귀무가설 채택 (차이 없음)

주의:
- p-value는 효과 크기가 아님
- p < 0.05는 임의의 기준 (도메인에 따라 조정 가능)
- "유의하다" ≠ "중요하다" (practical significance)
```

#### 제1종 오류 vs 제2종 오류
```
                실제 H₀ 참    실제 H₀ 거짓
H₀ 기각         제1종 오류    올바른 결정
                (α)          (1-β: 검정력)
H₀ 채택         올바른 결정   제2종 오류
                             (β)

제1종 오류 (α): "거짓 양성" (False Positive)
- 실제로 차이가 없는데 있다고 판단
- 보통 α = 0.05로 설정

제2종 오류 (β): "거짓 음성" (False Negative)
- 실제로 차이가 있는데 없다고 판단
- 검정력 = 1 - β (보통 0.8 이상 목표)
```

### 2.2 검정 방법 선택 가이드

```
                    정규성 만족?
                   /           \
                 Yes            No
                 /               \
          모수 검정           비모수 검정
         /    |    \          /    |    \
      t-test ANOVA F-test  Mann-W K-W  Wilcoxon

추가 고려사항:
1. 샘플 크기
   - n < 30: 정규성 중요 (비모수 고려)
   - n ≥ 30: 중심극한정리로 정규성 완화
2. 분산 동질성
   - 만족: Student's t-test
   - 불만족: Welch's t-test
3. 대응 여부
   - 독립표본: Independent t-test
   - 대응표본: Paired t-test
```

### 2.3 시나리오

**시나리오 1: 신규 기능 A/B 테스트**
```
상황: 웹사이트 결제 버튼 색상 변경
- A그룹 (기존): 파란색 버튼, 1000명, 전환율 5.2%
- B그룹 (신규): 빨간색 버튼, 1000명, 전환율 6.1%

질문: 차이가 통계적으로 유의미한가?

분석:
1. 가설 설정
   H₀: p_A = p_B (전환율 동일)
   H₁: p_A ≠ p_B (전환율 다름)
2. 검정 선택: Two-proportion z-test
3. 결과: p = 0.032 < 0.05
4. 결론: 유의미한 차이 → 빨간색 버튼 채택

액션: 전체 사용자에게 빨간색 버튼 적용
```

**시나리오 2: 약물 효능 비교 (3그룹)**
```
상황: 신약 A, 신약 B, 위약 비교
- 각 그룹 30명, 혈압 감소량 측정

분석:
1. 가설
   H₀: μ_A = μ_B = μ_위약
   H₁: 적어도 하나는 다름
2. 검정: One-way ANOVA
3. 결과: F = 8.5, p = 0.001
4. Post-hoc: Tukey HSD
   - A vs 위약: p < 0.001 (유의)
   - B vs 위약: p = 0.023 (유의)
   - A vs B: p = 0.412 (비유의)

결론: 두 신약 모두 효과 있음, 서로 비슷
```

**시나리오 3: 정규성 불만족 (비모수 검정)**
```
상황: 고객 만족도 (1-5점 척도)
- 기존 서비스: [3, 4, 3, 5, 4, 3, ...]
- 개선 서비스: [4, 5, 4, 5, 5, 4, ...]

문제: 정규성 불만족 (Shapiro-Wilk p = 0.003)

해결:
1. t-test 대신 Mann-Whitney U test 사용
2. 결과: U = 450, p = 0.018
3. 결론: 개선 서비스가 유의미하게 높은 만족도
```

---

## 3. 구현

### 3.1 환경 설정

```python
# 필수 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import (
    ttest_ind, ttest_rel, ttest_1samp,
    f_oneway, kruskal,
    mannwhitneyu, wilcoxon,
    chi2_contingency, fisher_exact,
    shapiro, normaltest, levene,
    pearsonr, spearmanr
)
import warnings
warnings.filterwarnings('ignore')

# 통계 모델링
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower

# 시각화
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
%matplotlib inline

# 한글 폰트 (선택)
plt.rcParams['font.family'] = 'AppleGothic'  # Mac
plt.rcParams['axes.unicode_minus'] = False

# 출력 옵션
np.set_printoptions(precision=4, suppress=True)
pd.set_option('display.precision', 4)
```

### 3.2 샘플 데이터 생성

```python
def generate_ab_test_data(n_per_group=500, effect_size=0.3, seed=42):
    """
    A/B 테스트용 샘플 데이터 생성
    
    Parameters:
    -----------
    n_per_group : int
        각 그룹 샘플 크기
    effect_size : float
        효과 크기 (Cohen's d)
    seed : int
    
    Returns:
    --------
    df : DataFrame
    """
    np.random.seed(seed)
    
    # 그룹 A (Control)
    group_a = np.random.normal(loc=100, scale=15, size=n_per_group)
    
    # 그룹 B (Treatment) - effect_size만큼 평균 증가
    mean_increase = effect_size * 15  # effect_size * std
    group_b = np.random.normal(loc=100 + mean_increase, scale=15, size=n_per_group)
    
    # DataFrame 생성
    df = pd.DataFrame({
        'group': ['A'] * n_per_group + ['B'] * n_per_group,
        'value': np.concatenate([group_a, group_b])
    })
    
    print(f"=" * 70)
    print(f"📊 A/B 테스트 데이터 생성")
    print(f"=" * 70)
    print(f"그룹 A (Control):  n={n_per_group}, μ={group_a.mean():.2f}, σ={group_a.std():.2f}")
    print(f"그룹 B (Treatment): n={n_per_group}, μ={group_b.mean():.2f}, σ={group_b.std():.2f}")
    print(f"Effect Size (Cohen's d): {effect_size}")
    print(f"\n기술 통계:")
    print(df.groupby('group')['value'].describe())
    
    return df

# 데이터 생성
df_ab = generate_ab_test_data(n_per_group=500, effect_size=0.3)
```

### 3.3 정규성 검정 (Normality Test)

```python
def check_normality(data, alpha=0.05):
    """
    정규성 검정 (Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov)
    
    Parameters:
    -----------
    data : array-like
        검정할 데이터
    alpha : float
        유의수준
    
    Returns:
    --------
    is_normal : bool
    """
    print(f"\n" + "=" * 70)
    print(f"📊 정규성 검정 (Normality Tests)")
    print(f"=" * 70)
    print(f"샘플 크기: {len(data)}")
    print(f"평균: {data.mean():.4f}, 표준편차: {data.std():.4f}")
    
    # 1. Shapiro-Wilk test (n < 5000)
    if len(data) <= 5000:
        stat_sw, p_sw = shapiro(data)
        print(f"\n1. Shapiro-Wilk Test:")
        print(f"   통계량: {stat_sw:.4f}")
        print(f"   P-value: {p_sw:.4f}")
        if p_sw > alpha:
            print(f"   결론: 정규분포를 따름 (p > {alpha})")
        else:
            print(f"   결론: 정규분포를 따르지 않음 (p ≤ {alpha})")
    
    # 2. Anderson-Darling test
    result_ad = stats.anderson(data, dist='norm')
    print(f"\n2. Anderson-Darling Test:")
    print(f"   통계량: {result_ad.statistic:.4f}")
    print(f"   Critical values: {result_ad.critical_values}")
    print(f"   Significance levels: {result_ad.significance_level}%")
    
    # 3. Kolmogorov-Smirnov test
    stat_ks, p_ks = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
    print(f"\n3. Kolmogorov-Smirnov Test:")
    print(f"   통계량: {stat_ks:.4f}")
    print(f"   P-value: {p_ks:.4f}")
    if p_ks > alpha:
        print(f"   결론: 정규분포를 따름 (p > {alpha})")
    else:
        print(f"   결론: 정규분포를 따르지 않음 (p ≤ {alpha})")
    
    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    # 히스토그램 + 정규분포 곡선
    axes[0].hist(data, bins=30, density=True, alpha=0.7, edgecolor='black')
    mu, sigma = data.mean(), data.std()
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    axes[0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Dist')
    axes[0].set_xlabel('Value', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('Histogram + Normal Curve', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q Plot
    stats.probplot(data, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Boxplot
    axes[2].boxplot(data, vert=True)
    axes[2].set_ylabel('Value', fontsize=11)
    axes[2].set_title('Boxplot', fontsize=12)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # 종합 결론
    print(f"\n💡 종합 결론:")
    if len(data) <= 5000:
        is_normal = p_sw > alpha
    else:
        is_normal = p_ks > alpha
    
    if is_normal:
        print(f"  ✅ 데이터가 정규분포를 따름 → 모수 검정 (t-test, ANOVA) 사용 가능")
    else:
        print(f"  ⚠️  데이터가 정규분포를 따르지 않음 → 비모수 검정 (Mann-Whitney, Kruskal-Wallis) 권장")
    
    return is_normal

# 정규성 검정
group_a_data = df_ab[df_ab['group'] == 'A']['value']
is_normal = check_normality(group_a_data, alpha=0.05)
```

### 3.4 독립표본 t-test (Independent t-test)

```python
def perform_independent_ttest(group1, group2, alpha=0.05, equal_var=True):
    """
    독립표본 t-test 수행
    
    Parameters:
    -----------
    group1, group2 : array-like
        비교할 두 그룹 데이터
    alpha : float
        유의수준
    equal_var : bool
        분산 동질성 가정 (True: Student's, False: Welch's)
    
    Returns:
    --------
    result : dict
    """
    print(f"\n" + "=" * 70)
    print(f"📊 독립표본 t-test (Independent Samples t-test)")
    print(f"=" * 70)
    
    # 기술 통계
    print(f"\n기술 통계:")
    print(f"  Group 1: n={len(group1)}, μ={group1.mean():.4f}, σ={group1.std():.4f}")
    print(f"  Group 2: n={len(group2)}, μ={group2.mean():.4f}, σ={group2.std():.4f}")
    print(f"  평균 차이: {group2.mean() - group1.mean():.4f}")
    
    # 등분산 검정 (Levene's test)
    stat_levene, p_levene = levene(group1, group2)
    print(f"\n등분산 검정 (Levene's test):")
    print(f"  통계량: {stat_levene:.4f}, P-value: {p_levene:.4f}")
    if p_levene > alpha:
        print(f"  결론: 등분산 가정 만족 (p > {alpha}) → Student's t-test")
        equal_var = True
    else:
        print(f"  결론: 등분산 가정 불만족 (p ≤ {alpha}) → Welch's t-test")
        equal_var = False
    
    # t-test 수행
    t_stat, p_value = ttest_ind(group1, group2, equal_var=equal_var)
    
    # 자유도
    if equal_var:
        df = len(group1) + len(group2) - 2
    else:
        # Welch-Satterthwaite equation
        s1, s2 = group1.std(ddof=1), group2.std(ddof=1)
        n1, n2 = len(group1), len(group2)
        df = ((s1**2/n1 + s2**2/n2)**2) / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
    
    # 효과 크기 (Cohen's d)
    pooled_std = np.sqrt(((len(group1)-1)*group1.std()**2 + (len(group2)-1)*group2.std()**2) / (len(group1)+len(group2)-2))
    cohens_d = (group2.mean() - group1.mean()) / pooled_std
    
    # 신뢰구간
    diff_mean = group2.mean() - group1.mean()
    se = np.sqrt(group1.var()/len(group1) + group2.var()/len(group2))
    t_crit = stats.t.ppf(1 - alpha/2, df)
    ci_lower = diff_mean - t_crit * se
    ci_upper = diff_mean + t_crit * se
    
    # 결과 출력
    print(f"\nt-test 결과:")
    print(f"  H₀: μ₁ = μ₂ (두 그룹의 평균이 같다)")
    print(f"  H₁: μ₁ ≠ μ₂ (두 그룹의 평균이 다르다)")
    print(f"\n  t-통계량: {t_stat:.4f}")
    print(f"  자유도: {df:.2f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  유의수준: {alpha}")
    
    if p_value < alpha:
        print(f"\n  ✅ 결론: 귀무가설 기각 (p < {alpha})")
        print(f"     → 두 그룹 간 평균에 유의미한 차이가 있음")
    else:
        print(f"\n  ❌ 결론: 귀무가설 채택 (p ≥ {alpha})")
        print(f"     → 두 그룹 간 평균에 유의미한 차이가 없음")
    
    print(f"\n효과 크기 (Cohen's d): {cohens_d:.4f}")
    if abs(cohens_d) < 0.2:
        effect_size_interp = "작음 (small)"
    elif abs(cohens_d) < 0.5:
        effect_size_interp = "중간 (medium)"
    elif abs(cohens_d) < 0.8:
        effect_size_interp = "큼 (large)"
    else:
        effect_size_interp = "매우 큼 (very large)"
    print(f"  해석: {effect_size_interp}")
    
    print(f"\n평균 차이의 95% 신뢰구간: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 박스플롯
    data_plot = pd.DataFrame({
        'Group': ['Group 1']*len(group1) + ['Group 2']*len(group2),
        'Value': np.concatenate([group1, group2])
    })
    sns.boxplot(data=data_plot, x='Group', y='Value', ax=axes[0], palette='Set2')
    axes[0].set_title(f't-test: p-value = {p_value:.4f}', fontsize=12)
    axes[0].set_ylabel('Value', fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 분포 비교
    axes[1].hist(group1, bins=30, alpha=0.6, label='Group 1', density=True)
    axes[1].hist(group2, bins=30, alpha=0.6, label='Group 2', density=True)
    axes[1].axvline(group1.mean(), color='blue', linestyle='--', linewidth=2, label=f'μ₁={group1.mean():.2f}')
    axes[1].axvline(group2.mean(), color='orange', linestyle='--', linewidth=2, label=f'μ₂={group2.mean():.2f}')
    axes[1].set_xlabel('Value', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].set_title('Distribution Comparison', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'df': df,
        'cohens_d': cohens_d,
        'ci': (ci_lower, ci_upper),
        'significant': p_value < alpha
    }

# 독립표본 t-test 수행
group_a = df_ab[df_ab['group'] == 'A']['value'].values
group_b = df_ab[df_ab['group'] == 'B']['value'].values
ttest_result = perform_independent_ttest(group_a, group_b, alpha=0.05)
```

### 3.5 대응표본 t-test (Paired t-test)

```python
def perform_paired_ttest(before, after, alpha=0.05):
    """
    대응표본 t-test 수행 (전후 비교)
    
    Parameters:
    -----------
    before, after : array-like
        전후 데이터 (같은 개체)
    alpha : float
    
    Returns:
    --------
    result : dict
    """
    print(f"\n" + "=" * 70)
    print(f"📊 대응표본 t-test (Paired Samples t-test)")
    print(f"=" * 70)
    
    # 차이 계산
    diff = after - before
    
    # 기술 통계
    print(f"\n기술 통계:")
    print(f"  Before: n={len(before)}, μ={before.mean():.4f}, σ={before.std():.4f}")
    print(f"  After:  n={len(after)}, μ={after.mean():.4f}, σ={after.std():.4f}")
    print(f"  Difference: μ_diff={diff.mean():.4f}, σ_diff={diff.std():.4f}")
    
    # Paired t-test 수행
    t_stat, p_value = ttest_rel(before, after)
    
    # 자유도
    df = len(before) - 1
    
    # 효과 크기 (Cohen's d for paired)
    cohens_d = diff.mean() / diff.std()
    
    # 신뢰구간
    se = diff.std() / np.sqrt(len(diff))
    t_crit = stats.t.ppf(1 - alpha/2, df)
    ci_lower = diff.mean() - t_crit * se
    ci_upper = diff.mean() + t_crit * se
    
    # 결과 출력
    print(f"\nPaired t-test 결과:")
    print(f"  H₀: μ_diff = 0 (전후 차이가 없다)")
    print(f"  H₁: μ_diff ≠ 0 (전후 차이가 있다)")
    print(f"\n  t-통계량: {t_stat:.4f}")
    print(f"  자유도: {df}")
    print(f"  P-value: {p_value:.4f}")
    
    if p_value < alpha:
        print(f"\n  ✅ 결론: 귀무가설 기각 (p < {alpha})")
        print(f"     → 전후 유의미한 변화가 있음")
        if diff.mean() > 0:
            print(f"     → After가 Before보다 유의미하게 높음")
        else:
            print(f"     → After가 Before보다 유의미하게 낮음")
    else:
        print(f"\n  ❌ 결론: 귀무가설 채택 (p ≥ {alpha})")
        print(f"     → 전후 유의미한 변화가 없음")
    
    print(f"\n효과 크기 (Cohen's d): {cohens_d:.4f}")
    print(f"평균 차이의 95% 신뢰구간: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    # 전후 비교 (연결선)
    for i in range(min(50, len(before))):  # 최대 50개만 표시
        axes[0].plot([0, 1], [before[i], after[i]], 'o-', alpha=0.3, color='gray')
    axes[0].plot([0, 1], [before.mean(), after.mean()], 'ro-', linewidth=3, markersize=10, label='Mean')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Before', 'After'])
    axes[0].set_ylabel('Value', fontsize=11)
    axes[0].set_title('Before vs After', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 차이 분포
    axes[1].hist(diff, bins=30, alpha=0.7, edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='μ_diff=0')
    axes[1].axvline(diff.mean(), color='green', linestyle='--', linewidth=2, label=f'μ_diff={diff.mean():.2f}')
    axes[1].set_xlabel('Difference (After - Before)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Distribution of Differences', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Boxplot
    data_plot = pd.DataFrame({
        'Condition': ['Before']*len(before) + ['After']*len(after),
        'Value': np.concatenate([before, after])
    })
    sns.boxplot(data=data_plot, x='Condition', y='Value', ax=axes[2], palette='Set2')
    axes[2].set_title(f'Paired t-test: p-value = {p_value:.4f}', fontsize=12)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'df': df,
        'cohens_d': cohens_d,
        'mean_diff': diff.mean(),
        'ci': (ci_lower, ci_upper),
        'significant': p_value < alpha
    }

# 대응표본 데이터 생성 및 검정
np.random.seed(42)
before_treatment = np.random.normal(100, 15, 200)
after_treatment = before_treatment + np.random.normal(5, 10, 200)  # 평균 5 증가

paired_result = perform_paired_ttest(before_treatment, after_treatment, alpha=0.05)
```

### 3.6 ANOVA (Analysis of Variance) - 3그룹 이상

```python
def perform_anova(groups, group_names, alpha=0.05):
    """
    일원 분산분석 (One-way ANOVA)
    
    Parameters:
    -----------
    groups : list of arrays
        비교할 그룹들 (3개 이상)
    group_names : list of str
        그룹 이름
    alpha : float
    
    Returns:
    --------
    result : dict
    """
    print(f"\n" + "=" * 70)
    print(f"📊 일원 분산분석 (One-way ANOVA)")
    print(f"=" * 70)
    
    # 기술 통계
    print(f"\n기술 통계:")
    for i, (group, name) in enumerate(zip(groups, group_names)):
        print(f"  {name}: n={len(group)}, μ={group.mean():.4f}, σ={group.std():.4f}")
    
    # ANOVA 수행
    f_stat, p_value = f_oneway(*groups)
    
    # 자유도
    k = len(groups)  # 그룹 수
    n = sum([len(g) for g in groups])  # 총 샘플 수
    df_between = k - 1
    df_within = n - k
    
    # 효과 크기 (Eta-squared)
    grand_mean = np.mean(np.concatenate(groups))
    ss_between = sum([len(g) * (g.mean() - grand_mean)**2 for g in groups])
    ss_within = sum([((g - g.mean())**2).sum() for g in groups])
    ss_total = ss_between + ss_within
    eta_squared = ss_between / ss_total
    
    # 결과 출력
    print(f"\nANOVA 결과:")
    print(f"  H₀: μ₁ = μ₂ = ... = μₖ (모든 그룹의 평균이 같다)")
    print(f"  H₁: 적어도 하나의 그룹 평균이 다르다")
    print(f"\n  F-통계량: {f_stat:.4f}")
    print(f"  자유도 (between): {df_between}")
    print(f"  자유도 (within): {df_within}")
    print(f"  P-value: {p_value:.4f}")
    
    if p_value < alpha:
        print(f"\n  ✅ 결론: 귀무가설 기각 (p < {alpha})")
        print(f"     → 적어도 하나의 그룹 평균이 다름")
        print(f"     → Post-hoc 검정 필요 (Tukey HSD)")
    else:
        print(f"\n  ❌ 결론: 귀무가설 채택 (p ≥ {alpha})")
        print(f"     → 모든 그룹의 평균이 같음")
    
    print(f"\n효과 크기 (Eta-squared): {eta_squared:.4f}")
    if eta_squared < 0.01:
        effect_size_interp = "작음"
    elif eta_squared < 0.06:
        effect_size_interp = "중간"
    else:
        effect_size_interp = "큼"
    print(f"  해석: {effect_size_interp}")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 박스플롯
    data_plot = pd.DataFrame({
        'Group': np.concatenate([[name]*len(g) for name, g in zip(group_names, groups)]),
        'Value': np.concatenate(groups)
    })
    sns.boxplot(data=data_plot, x='Group', y='Value', ax=axes[0], palette='Set3')
    axes[0].set_title(f'ANOVA: F={f_stat:.2f}, p-value={p_value:.4f}', fontsize=12)
    axes[0].set_ylabel('Value', fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 평균 비교
    means = [g.mean() for g in groups]
    sems = [g.std() / np.sqrt(len(g)) for g in groups]
    axes[1].bar(group_names, means, yerr=sems, alpha=0.7, capsize=10, edgecolor='black')
    axes[1].set_ylabel('Mean Value', fontsize=11)
    axes[1].set_title('Group Means (±SEM)', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Post-hoc 검정 (ANOVA 유의 시)
    if p_value < alpha:
        perform_posthoc_tukey(groups, group_names, alpha)
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'df_between': df_between,
        'df_within': df_within,
        'eta_squared': eta_squared,
        'significant': p_value < alpha
    }

def perform_posthoc_tukey(groups, group_names, alpha=0.05):
    """
    Tukey HSD post-hoc test
    """
    print(f"\n" + "=" * 70)
    print(f"📊 Post-hoc: Tukey HSD")
    print(f"=" * 70)
    
    # 데이터 준비
    data_all = np.concatenate(groups)
    labels_all = np.concatenate([[name]*len(g) for name, g in zip(group_names, groups)])
    
    # Tukey HSD
    tukey_result = pairwise_tukeyhsd(data_all, labels_all, alpha=alpha)
    
    print(tukey_result)
    
    # 시각화
    plt.figure(figsize=(8, 6))
    tukey_result.plot_simultaneous(ylabel='Group', xlabel='Mean Value')
    plt.title('Tukey HSD: 95% Confidence Intervals', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    print(f"\n💡 해석:")
    print(f"  - reject=True: 두 그룹 간 유의미한 차이")
    print(f"  - 신뢰구간이 0을 포함하지 않으면 유의미한 차이")

# 3그룹 데이터 생성 및 ANOVA
np.random.seed(42)
group1 = np.random.normal(100, 15, 150)
group2 = np.random.normal(105, 15, 150)
group3 = np.random.normal(110, 15, 150)

anova_result = perform_anova(
    [group1, group2, group3],
    ['Control', 'Treatment A', 'Treatment B'],
    alpha=0.05
)
```

### 3.7 Mann-Whitney U Test (비모수 t-test 대체)

```python
def perform_mann_whitney(group1, group2, alpha=0.05):
    """
    Mann-Whitney U test (비모수 독립표본 검정)
    
    정규성 가정 불필요, 순위 기반
    """
    print(f"\n" + "=" * 70)
    print(f"📊 Mann-Whitney U Test (비모수 검정)")
    print(f"=" * 70)
    
    # 기술 통계
    print(f"\n기술 통계:")
    print(f"  Group 1: n={len(group1)}, median={np.median(group1):.4f}, IQR={stats.iqr(group1):.4f}")
    print(f"  Group 2: n={len(group2)}, median={np.median(group2):.4f}, IQR={stats.iqr(group2):.4f}")
    
    # Mann-Whitney U test
    u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
    
    # 효과 크기 (Rank-biserial correlation)
    n1, n2 = len(group1), len(group2)
    r = 1 - (2*u_stat) / (n1 * n2)  # Rank-biserial correlation
    
    # 결과 출력
    print(f"\nMann-Whitney U Test 결과:")
    print(f"  H₀: 두 그룹의 분포가 같다")
    print(f"  H₁: 두 그룹의 분포가 다르다")
    print(f"\n  U-통계량: {u_stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    
    if p_value < alpha:
        print(f"\n  ✅ 결론: 귀무가설 기각 (p < {alpha})")
        print(f"     → 두 그룹 간 유의미한 차이가 있음")
    else:
        print(f"\n  ❌ 결론: 귀무가설 채택 (p ≥ {alpha})")
        print(f"     → 두 그룹 간 유의미한 차이가 없음")
    
    print(f"\n효과 크기 (Rank-biserial correlation): {r:.4f}")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 박스플롯
    data_plot = pd.DataFrame({
        'Group': ['Group 1']*len(group1) + ['Group 2']*len(group2),
        'Value': np.concatenate([group1, group2])
    })
    sns.boxplot(data=data_plot, x='Group', y='Value', ax=axes[0], palette='Set2')
    axes[0].set_title(f'Mann-Whitney U: p-value = {p_value:.4f}', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 분포 비교
    axes[1].hist(group1, bins=30, alpha=0.6, label='Group 1', density=True)
    axes[1].hist(group2, bins=30, alpha=0.6, label='Group 2', density=True)
    axes[1].axvline(np.median(group1), color='blue', linestyle='--', linewidth=2, label=f'Med₁={np.median(group1):.2f}')
    axes[1].axvline(np.median(group2), color='orange', linestyle='--', linewidth=2, label=f'Med₂={np.median(group2):.2f}')
    axes[1].set_xlabel('Value', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].set_title('Distribution Comparison', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'u_statistic': u_stat,
        'p_value': p_value,
        'effect_size_r': r,
        'significant': p_value < alpha
    }

# 비정규 데이터 생성 (지수분포)
np.random.seed(42)
non_normal_1 = np.random.exponential(scale=2.0, size=200)
non_normal_2 = np.random.exponential(scale=2.5, size=200)

mann_whitney_result = perform_mann_whitney(non_normal_1, non_normal_2, alpha=0.05)
```

### 3.8 Chi-Square Test (범주형 변수 독립성 검정)

```python
def perform_chi_square_test(contingency_table, alpha=0.05):
    """
    Chi-square test of independence (범주형 변수 독립성 검정)
    
    Parameters:
    -----------
    contingency_table : DataFrame or array
        교차표 (contingency table)
    alpha : float
    
    Returns:
    --------
    result : dict
    """
    print(f"\n" + "=" * 70)
    print(f"📊 Chi-square Test of Independence")
    print(f"=" * 70)
    
    # 교차표 출력
    print(f"\n교차표 (Contingency Table):")
    print(contingency_table)
    
    # Chi-square 검정
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # 효과 크기 (Cramér's V)
    n = contingency_table.sum().sum() if isinstance(contingency_table, pd.DataFrame) else contingency_table.sum()
    min_dim = min(contingency_table.shape[0], contingency_table.shape[1]) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    
    # 결과 출력
    print(f"\nChi-square Test 결과:")
    print(f"  H₀: 두 변수가 독립적이다 (관련 없음)")
    print(f"  H₁: 두 변수가 독립적이지 않다 (관련 있음)")
    print(f"\n  χ² 통계량: {chi2:.4f}")
    print(f"  자유도: {dof}")
    print(f"  P-value: {p_value:.4f}")
    
    if p_value < alpha:
        print(f"\n  ✅ 결론: 귀무가설 기각 (p < {alpha})")
        print(f"     → 두 변수 간 유의미한 관련성이 있음")
    else:
        print(f"\n  ❌ 결론: 귀무가설 채택 (p ≥ {alpha})")
        print(f"     → 두 변수 간 관련성이 없음 (독립적)")
    
    print(f"\n효과 크기 (Cramér's V): {cramers_v:.4f}")
    if cramers_v < 0.1:
        effect_interp = "작음"
    elif cramers_v < 0.3:
        effect_interp = "중간"
    else:
        effect_interp = "큼"
    print(f"  해석: {effect_interp}")
    
    print(f"\n기대빈도 (Expected Frequencies):")
    print(pd.DataFrame(expected, 
                       index=contingency_table.index if isinstance(contingency_table, pd.DataFrame) else range(contingency_table.shape[0]),
                       columns=contingency_table.columns if isinstance(contingency_table, pd.DataFrame) else range(contingency_table.shape[1])))
    
    # 기대빈도 < 5 경고
    if (expected < 5).any():
        print(f"\n⚠️  경고: 기대빈도가 5 미만인 셀이 있습니다.")
        print(f"   → Fisher's exact test 고려 (2x2 표의 경우)")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 히트맵 (관측빈도)
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title(f'Observed Frequencies\nχ²={chi2:.2f}, p={p_value:.4f}', fontsize=12)
    
    # 히트맵 (기대빈도)
    expected_df = pd.DataFrame(expected,
                               index=contingency_table.index if isinstance(contingency_table, pd.DataFrame) else range(contingency_table.shape[0]),
                               columns=contingency_table.columns if isinstance(contingency_table, pd.DataFrame) else range(contingency_table.shape[1]))
    sns.heatmap(expected_df, annot=True, fmt='.1f', cmap='Blues', ax=axes[1], cbar_kws={'label': 'Expected Count'})
    axes[1].set_title('Expected Frequencies', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'chi2_statistic': chi2,
        'p_value': p_value,
        'dof': dof,
        'cramers_v': cramers_v,
        'expected': expected,
        'significant': p_value < alpha
    }

# 범주형 데이터 예시
contingency_table = pd.DataFrame(
    [[50, 30, 20],
     [45, 40, 15],
     [35, 35, 30]],
    index=['Group A', 'Group B', 'Group C'],
    columns=['Category 1', 'Category 2', 'Category 3']
)

chi_square_result = perform_chi_square_test(contingency_table, alpha=0.05)
```

### 3.9 검정력 분석 (Power Analysis)

```python
def perform_power_analysis(effect_size, alpha=0.05, power=0.8, test_type='t-test'):
    """
    검정력 분석: 필요한 샘플 크기 계산
    
    Parameters:
    -----------
    effect_size : float
        효과 크기 (Cohen's d)
    alpha : float
        제1종 오류 확률
    power : float
        검정력 (1 - β)
    test_type : str
        't-test' or 'anova'
    
    Returns:
    --------
    sample_size : int
    """
    print(f"\n" + "=" * 70)
    print(f"📊 검정력 분석 (Power Analysis)")
    print(f"=" * 70)
    
    if test_type == 't-test':
        analysis = TTestIndPower()
        sample_size = analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            ratio=1.0,  # 그룹 크기 비율
            alternative='two-sided'
        )
        
        print(f"\n독립표본 t-test:")
        print(f"  효과 크기 (Cohen's d): {effect_size}")
        print(f"  유의수준 (α): {alpha}")
        print(f"  검정력 (1-β): {power}")
        print(f"\n  ✅ 필요한 샘플 크기 (각 그룹): {int(np.ceil(sample_size))}명")
        print(f"     → 총 샘플: {int(np.ceil(sample_size)) * 2}명")
    
    # 효과 크기별 샘플 크기 시각화
    effect_sizes = np.linspace(0.1, 1.0, 50)
    sample_sizes = [analysis.solve_power(es, alpha, power, 1.0, 'two-sided') for es in effect_sizes]
    
    plt.figure(figsize=(10, 6))
    plt.plot(effect_sizes, sample_sizes, linewidth=2)
    plt.axhline(y=sample_size, color='r', linestyle='--', label=f'Current: n={int(np.ceil(sample_size))}')
    plt.axvline(x=effect_size, color='r', linestyle='--')
    plt.xlabel('Effect Size (Cohen\'s d)', fontsize=12)
    plt.ylabel('Required Sample Size (per group)', fontsize=12)
    plt.title(f'Power Analysis\n(α={alpha}, power={power})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\n💡 해석:")
    print(f"  - 효과 크기가 클수록 필요한 샘플 크기 감소")
    print(f"  - 검정력을 높이려면 샘플 크기 증가 필요")
    print(f"  - 작은 효과(d=0.2)는 큰 샘플(n≈400/group) 필요")
    
    return int(np.ceil(sample_size))

# 검정력 분석
required_n = perform_power_analysis(effect_size=0.3, alpha=0.05, power=0.8, test_type='t-test')
```

### 3.10 종합 검정 요약 함수

```python
def comprehensive_hypothesis_test(group1, group2, alpha=0.05, paired=False):
    """
    종합 가설 검정: 정규성 확인 후 적절한 검정 선택
    
    Parameters:
    -----------
    group1, group2 : array-like
    alpha : float
    paired : bool
        대응표본 여부
    """
    print(f"\n" + "=" * 80)
    print(f"🔬 종합 가설 검정 (Comprehensive Hypothesis Test)")
    print(f"=" * 80)
    
    # 1. 정규성 검정
    _, p_norm1 = shapiro(group1) if len(group1) <= 5000 else stats.kstest(group1, 'norm')
    _, p_norm2 = shapiro(group2) if len(group2) <= 5000 else stats.kstest(group2, 'norm')
    
    is_normal = (p_norm1 > 0.05) and (p_norm2 > 0.05)
    
    print(f"\n1단계: 정규성 검정")
    print(f"  Group 1: p={p_norm1:.4f} {'(정규)' if p_norm1 > 0.05 else '(비정규)'}")
    print(f"  Group 2: p={p_norm2:.4f} {'(정규)' if p_norm2 > 0.05 else '(비정규)'}")
    
    # 2. 검정 선택 및 수행
    print(f"\n2단계: 검정 수행")
    
    if is_normal:
        if paired:
            print(f"  선택: Paired t-test (대응표본, 정규)")
            result = perform_paired_ttest(group1, group2, alpha)
        else:
            print(f"  선택: Independent t-test (독립표본, 정규)")
            result = perform_independent_ttest(group1, group2, alpha)
    else:
        if paired:
            print(f"  선택: Wilcoxon Signed-Rank test (대응표본, 비정규)")
            stat, p_val = wilcoxon(group1, group2)
            print(f"  통계량: {stat:.4f}, P-value: {p_val:.4f}")
            result = {'p_value': p_val, 'significant': p_val < alpha}
        else:
            print(f"  선택: Mann-Whitney U test (독립표본, 비정규)")
            result = perform_mann_whitney(group1, group2, alpha)
    
    # 3. 최종 결론
    print(f"\n3단계: 최종 결론")
    if result['significant']:
        print(f"  ✅ 두 그룹 간 통계적으로 유의미한 차이가 있습니다 (p < {alpha})")
    else:
        print(f"  ❌ 두 그룹 간 통계적으로 유의미한 차이가 없습니다 (p ≥ {alpha})")
    
    return result

# 종합 검정 실행
comprehensive_result = comprehensive_hypothesis_test(group_a, group_b, alpha=0.05, paired=False)
```

---

## 4. 예시

(Due to length constraints, I'll continue in a separate message with sections 4-10)

### 4.1 실전 예제: A/B 테스트 (전환율 비교)

```python
print("=" * 70)
print("📈 비즈니스 시나리오: 웹사이트 A/B 테스트")
print("=" * 70)

print("\n목표:")
print("- 신규 결제 페이지 vs 기존 결제 페이지")
print("- 전환율 비교 및 유의성 검정")

print("\n데이터:")
print("- A그룹 (기존): 10,000명, 전환 520명 (5.2%)")
print("- B그룹 (신규): 10,000명, 전환 610명 (6.1%)")

print("\n분석:")
print("1. 가설 설정")
print("   H₀: p_A = p_B")
print("   H₁: p_B > p_A (one-tailed)")
print("\n2. Two-proportion z-test 수행")
print("   z = 3.21, p = 0.0007")
print("\n3. 효과 크기")
print("   절대 차이: 0.9%p")
print("   상대 차이: 17.3% 증가")

print("\n결론:")
print("  ✅ 신규 페이지가 통계적으로 유의미하게 높은 전환율")
print("  ✅ 실무적으로도 의미 있는 개선 (17% 증가)")
print("\n액션:")
print("  → 신규 페이지 전면 적용")
print("  → 연간 추가 수익: 약 $50,000 예상")
```

---

## 5. 에이전트 매핑

### 5.1 담당 에이전트

| 작업 | Primary Agent | Supporting Agents |
|------|--------------|-------------------|
| 가설 설정 | `data-scientist` | - |
| 정규성 검정 | `data-scientist` | - |
| t-test, ANOVA | `data-scientist` | - |
| 비모수 검정 | `data-scientist` | - |
| 검정력 분석 | `data-scientist` | - |
| 결과 해석 및 보고 | `data-scientist` | `technical-documentation-writer` |

### 5.2 관련 스킬

**Scientific Skills**:
- `scipy.stats` (모든 검정 함수)
- `statsmodels` (post-hoc, power analysis)
- `pandas`, `numpy` (데이터 처리)
- `matplotlib`, `seaborn` (시각화)

---

## 6. 필요 라이브러리

### 6.1 필수 라이브러리

```bash
# 통계 분석
pip install scipy==1.12.0
pip install statsmodels==0.14.1

# 데이터 처리
pip install pandas==2.2.0
pip install numpy==1.26.3

# 시각화
pip install matplotlib==3.8.2
pip install seaborn==0.13.1
```

---

## 7. 체크포인트

### 7.1 분석 전 체크리스트

- [ ] **가설 명확화**
  - [ ] 귀무가설 (H₀) 정의
  - [ ] 대립가설 (H₁) 정의
  - [ ] 단측 or 양측 검정?

- [ ] **데이터 준비**
  - [ ] 샘플 크기 충분한가?
  - [ ] 결측값 처리 완료
  - [ ] 이상치 확인

### 7.2 분석 중 체크리스트

- [ ] **가정 확인**
  - [ ] 정규성 검정 (모수 검정 시)
  - [ ] 등분산 검정 (t-test 시)
  - [ ] 독립성 확인

- [ ] **적절한 검정 선택**
  - [ ] 정규성 만족 → 모수 검정
  - [ ] 정규성 불만족 → 비모수 검정

### 7.3 분석 후 체크리스트

- [ ] **결과 해석**
  - [ ] P-value 확인 (α와 비교)
  - [ ] 효과 크기 계산
  - [ ] 실무적 의미 고려

- [ ] **보고**
  - [ ] 결론 명확히 작성
  - [ ] 시각화 포함
  - [ ] 제한점 언급

---

## 8. 트러블슈팅

### 8.1 일반적 오류

**문제 1: P-value는 유의하지만 효과 크기가 작음**

```python
# 원인: 샘플이 너무 큼 → 작은 차이도 유의
# 해결: 효과 크기(Cohen's d, eta-squared) 함께 보고

if p_value < 0.05 and abs(cohens_d) < 0.2:
    print("통계적으로 유의하지만 실무적 의미는 작음")
```

**문제 2: 정규성 검정에서 모두 거부됨 (대표본)**

```python
# 원인: n이 크면 작은 편차도 유의
# 해결: Q-Q plot으로 시각적 확인, 중심극한정리 적용

if n > 100:
    print("대표본: 정규성 완화, t-test 사용 가능")
else:
    print("소표본: 비모수 검정 권장")
```

**문제 3: 다중 비교 문제 (Multiple Testing)**

```python
# 원인: 여러 번 검정 → 제1종 오류 증가
# 해결: Bonferroni 보정, FDR 보정

alpha_corrected = 0.05 / n_comparisons  # Bonferroni
```

### 8.2 해석 관련

**Q1: P-value 0.051은 유의하지 않은가요?**

```
A: α=0.05 기준으로는 유의하지 않지만...
- 0.05는 임의의 기준 (절대적 아님)
- 도메인에 따라 0.1 또는 0.01 사용 가능
- P-value보다 효과 크기와 실무적 의미 중요

권장:
- P-value 정확히 보고 (0.051)
- 효과 크기 함께 제시
- 실무 판단은 종합적으로
```

**Q2: 통계적 유의성 vs 실무적 유의성?**

```
A:
통계적 유의성: p < 0.05
- 차이가 우연이 아니다

실무적 유의성: 효과 크기
- 차이가 의미 있는가?

예시:
웹사이트 전환율: 5.0% → 5.1%
- 통계적: p=0.001 (유의)
- 실무적: 0.1%p 증가는 미미

→ 둘 다 고려하여 판단
```

**Q3: 검정력(power)이 낮으면?**

```
A: 제2종 오류(β) 위험 증가
- 실제 차이가 있는데 못 찾을 확률 ↑
- 검정력 < 0.8: 샘플 부족 가능성

해결:
1. 샘플 크기 증가
2. 효과 크기가 작으면 많은 샘플 필요
3. 사전 검정력 분석으로 필요 샘플 계산
```

### 8.3 검정 선택 플로우

```
                   정규성 만족?
                   /         \
                 Yes          No
                 /             \
            대응표본?        대응표본?
            /    \           /    \
          Yes    No        Yes    No
          /       \         /      \
    Paired-t  Independent-t  Wilcoxon  Mann-Whitney
              /      \
          등분산?    
          /    \
        Yes    No
        /       \
   Student's  Welch's
```

---

## 9. 참고 자료

### 9.1 공식 문서

- **SciPy Stats**: https://docs.scipy.org/doc/scipy/reference/stats.html
- **Statsmodels**: https://www.statsmodels.org/stable/index.html
- **Statistical Power**: https://www.stat.ubc.ca/~rollin/stats/ssize/

### 9.2 베스트 프랙티스

1. **가설 검정 체크리스트**
   ```
   1. 연구 질문 명확화
   2. 가설 설정 (H₀, H₁)
   3. 유의수준 결정 (α)
   4. 검정 선택 (정규성, 독립성)
   5. 샘플 크기 확인 (검정력)
   6. 검정 수행
   7. P-value 해석
   8. 효과 크기 계산
   9. 결론 도출 (통계+실무)
   ```

2. **보고 형식**
   ```
   "독립표본 t-test 결과, 그룹 B(M=105.3, SD=15.2)가 
   그룹 A(M=100.1, SD=14.8)보다 통계적으로 유의미하게 
   높았다, t(998)=3.21, p=.001, d=0.34. 
   이는 중간 정도의 효과 크기이다."
   ```

### 9.3 추가 학습 자료

- **통계 검정 선택 가이드**: https://www.graphpad.com/guides/prism/latest/statistics/stat_choosing_a_test.htm
- **P-value 오해와 진실**: https://www.nature.com/articles/d41586-019-00857-9
- **Effect Size 가이드**: https://www.statisticshowto.com/effect-size/

---

## 10. 요약

### 10.1 핵심 메시지

가설 검정은 데이터 기반 의사결정의 과학적 근거를 제공합니다. P-value만이 아닌 효과 크기, 신뢰구간, 실무적 의미를 종합적으로 고려하여 판단해야 합니다. 정규성 등 가정을 확인하고 적절한 검정을 선택하는 것이 중요합니다.

### 10.2 검정 방법 선택 가이드

| 상황 | 추천 검정 |
|------|----------|
| 2그룹 평균 비교 (정규) | Independent t-test |
| 2그룹 평균 비교 (비정규) | Mann-Whitney U |
| 전후 비교 (정규) | Paired t-test |
| 전후 비교 (비정규) | Wilcoxon Signed-Rank |
| 3그룹+ 평균 비교 (정규) | ANOVA + Tukey |
| 3그룹+ 평균 비교 (비정규) | Kruskal-Wallis |
| 범주형 독립성 | Chi-square |
| 비율 비교 | Two-proportion z-test |

### 10.3 다음 단계

- **신뢰구간**: `12-statistical-inference.md` 참고
- **회귀 분석**: 변수 간 관계 모델링
- **베이지안 통계**: 사전 정보 활용
- **실험 설계**: A/B 테스트 최적화

---

**작성일**: 2025-01-25  
**버전**: 1.0  
**난이도**: ⭐⭐⭐ (고급)  
**예상 소요 시간**: 3-4시간 (학습 및 실습)
