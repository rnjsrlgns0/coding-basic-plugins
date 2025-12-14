# 12. Statistical Inference (통계적 추론)

## 1. 개요

### 1.1 목적
통계적 추론(Statistical Inference)은 표본 데이터를 바탕으로 모집단의 특성(모수)을 추정하고, 그 불확실성을 정량화하는 분석 기법입니다. 신뢰구간, 부트스트랩, 베이지안 추론 등을 통해 데이터 기반 의사결정의 신뢰도를 제공합니다.

### 1.2 적용 시기
- 표본으로부터 모집단의 평균, 비율 등을 추정할 때
- 추정값의 불확실성(uncertainty)을 정량화하고 싶을 때
- 신뢰구간(Confidence Interval)을 계산할 때
- 예측의 불확실성을 표현할 때 (예측 구간)
- 소표본에서 안정적인 추정이 필요할 때 (Bootstrap)

### 1.3 주요 기법
- **신뢰구간 (Confidence Interval)**: 모수의 가능한 범위 추정
- **Bootstrap**: 리샘플링을 통한 비모수적 추정
- **베이지안 추론**: 사전 정보와 데이터를 결합한 추정
- **예측 구간 (Prediction Interval)**: 미래 관측값의 범위
- **표준오차 (Standard Error)**: 추정량의 변동성

---

## 2. 이론적 배경

### 2.1 통계적 추론의 핵심 개념

#### 모수 vs 통계량
```
모수 (Parameter):
- 모집단의 특성 (μ, σ, p 등)
- 고정된 값 (알 수 없음)
- 추정의 대상

통계량 (Statistic):
- 표본의 특성 (x̄, s, p̂ 등)
- 표본에 따라 변함
- 모수의 추정치

예시:
모집단: 한국 성인 남성의 평균 키 (μ = 173.5cm)
표본: 100명 조사 → 평균 172.8cm (x̄)
→ x̄로 μ를 추정
```

#### 신뢰구간의 의미
```
95% 신뢰구간 [170.2, 175.4]의 의미:

올바른 해석:
"이 방법으로 100번 신뢰구간을 구하면, 
 약 95번은 진짜 모수(μ)를 포함한다"

잘못된 해석:
"μ가 이 구간에 있을 확률이 95%"
→ μ는 고정값, 확률 변수가 아님

실무적 해석:
"μ가 170.2~175.4 사이에 있다고 
 95% 신뢰할 수 있다"
```

#### 표준오차 (Standard Error)
```
정의: 통계량의 표준편차

평균의 표준오차:
SE(x̄) = σ / √n

의미:
- SE가 작을수록 추정이 정확
- 샘플 크기(n)가 클수록 SE 감소
- 정밀도 ∝ √n

예시:
σ = 15, n = 100 → SE = 15/√100 = 1.5
σ = 15, n = 400 → SE = 15/√400 = 0.75
→ 4배 샘플 = 2배 정밀도
```

### 2.2 신뢰구간의 종류

#### 1. 모수적 신뢰구간 (Parametric CI)
```
가정: 모집단 분포를 알고 있음 (보통 정규분포)

평균의 신뢰구간 (정규분포, σ 알려짐):
CI = x̄ ± z_(α/2) × (σ/√n)

평균의 신뢰구간 (정규분포, σ 모름):
CI = x̄ ± t_(α/2, n-1) × (s/√n)

비율의 신뢰구간:
CI = p̂ ± z_(α/2) × √(p̂(1-p̂)/n)
```

#### 2. 비모수적 신뢰구간 (Bootstrap CI)
```
가정 없음, 리샘플링 기반

장점:
- 분포 가정 불필요
- 복잡한 통계량에도 적용 가능
- 중앙값, 분산, 비율 등 모두 가능

단점:
- 계산 비용 높음
- 원본 데이터가 모집단을 잘 대표해야 함
```

### 2.3 시나리오

**시나리오 1: 고객 만족도 추정**
```
상황: 신규 서비스 론칭 후 고객 만족도 조사
- 표본: 200명
- 평균 만족도: 7.8/10
- 표준편차: 1.5

분석:
1. 95% 신뢰구간 계산
   CI = 7.8 ± 1.96 × (1.5/√200)
   CI = [7.59, 8.01]

2. 해석
   "모집단 평균 만족도가 7.59~8.01 사이에 있다고
    95% 신뢰할 수 있다"

3. 의사결정
   목표: 7.5 이상
   → 95% CI 하한(7.59) > 7.5
   → 목표 달성 확신
```

**시나리오 2: 전환율 추정 (Bootstrap)**
```
상황: 웹사이트 전환율 측정 (비정규 분포)
- 표본: 500명
- 전환: 53명 (10.6%)

문제: 비율 분포가 비대칭 (작은 p)
→ 정규근사 부적합

해결: Bootstrap 신뢰구간
1. 원본 데이터에서 500명 리샘플링 (복원추출)
2. 리샘플의 전환율 계산
3. 10,000번 반복
4. 2.5%, 97.5% 백분위수 = 95% CI

결과: [8.2%, 13.4%]
→ 비대칭 구간 (정규근사와 다름)
```

**시나리오 3: 예측 구간 (개별 예측)**
```
상황: 부동산 가격 예측 모델
- 예측 평균: 3억원
- 모델 오차: RMSE = 5천만원

질문: 개별 집의 가격 범위는?

차이:
- 신뢰구간: 평균 가격의 불확실성
- 예측 구간: 개별 가격의 불확실성 (더 넓음)

95% 예측 구간:
PI = 3억 ± 1.96 × 5천만
PI = [2.02억, 3.98억]

→ 신뢰구간보다 훨씬 넓음 (개별 변동성 포함)
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
from scipy.stats import t, norm, chi2
import warnings
warnings.filterwarnings('ignore')

# Bootstrap
from sklearn.utils import resample

# 베이지안 (선택)
# pip install pymc3  # 필요 시
# import pymc3 as pm

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
def generate_sample_data(n=200, mu=100, sigma=15, seed=42):
    """
    통계적 추론용 샘플 데이터 생성
    
    Parameters:
    -----------
    n : int
        샘플 크기
    mu : float
        모집단 평균 (참값)
    sigma : float
        모집단 표준편차
    seed : int
    
    Returns:
    --------
    data : array
    """
    np.random.seed(seed)
    data = np.random.normal(mu, sigma, n)
    
    print(f"=" * 70)
    print(f"📊 샘플 데이터 생성")
    print(f"=" * 70)
    print(f"모집단 모수 (참값):")
    print(f"  μ (평균): {mu}")
    print(f"  σ (표준편차): {sigma}")
    print(f"\n표본 통계량:")
    print(f"  n (샘플 크기): {n}")
    print(f"  x̄ (표본 평균): {data.mean():.4f}")
    print(f"  s (표본 표준편차): {data.std(ddof=1):.4f}")
    print(f"  SE (표준오차): {data.std(ddof=1) / np.sqrt(n):.4f}")
    
    return data

# 데이터 생성
sample_data = generate_sample_data(n=200, mu=100, sigma=15)
```

### 3.3 평균의 신뢰구간 (t-분포)

```python
def calculate_mean_ci(data, confidence=0.95):
    """
    평균의 신뢰구간 계산 (t-분포 기반)
    
    Parameters:
    -----------
    data : array-like
        샘플 데이터
    confidence : float
        신뢰수준 (0.95 = 95%)
    
    Returns:
    --------
    ci : tuple
        (하한, 상한)
    """
    print(f"\n" + "=" * 70)
    print(f"📊 평균의 {confidence*100:.0f}% 신뢰구간 (t-분포)")
    print(f"=" * 70)
    
    # 기본 통계량
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / np.sqrt(n)
    
    # t-분포 임계값
    alpha = 1 - confidence
    df = n - 1
    t_crit = t.ppf(1 - alpha/2, df)
    
    # 신뢰구간
    margin_of_error = t_crit * se
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    
    # 결과 출력
    print(f"\n기본 정보:")
    print(f"  샘플 크기 (n): {n}")
    print(f"  표본 평균 (x̄): {mean:.4f}")
    print(f"  표본 표준편차 (s): {std:.4f}")
    print(f"  표준오차 (SE): {se:.4f}")
    
    print(f"\n신뢰구간 계산:")
    print(f"  신뢰수준: {confidence*100:.0f}%")
    print(f"  자유도 (df): {df}")
    print(f"  t-임계값: {t_crit:.4f}")
    print(f"  오차한계 (ME): {margin_of_error:.4f}")
    
    print(f"\n{confidence*100:.0f}% 신뢰구간:")
    print(f"  [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    print(f"\n💡 해석:")
    print(f"  이 방법으로 100번 신뢰구간을 구하면,")
    print(f"  약 {confidence*100:.0f}번은 진짜 모평균(μ)을 포함합니다.")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 샘플 분포 + 신뢰구간
    axes[0].hist(data, bins=30, alpha=0.7, edgecolor='black', density=True)
    axes[0].axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
    axes[0].axvline(ci_lower, color='orange', linestyle='--', linewidth=2, label=f'CI Lower: {ci_lower:.2f}')
    axes[0].axvline(ci_upper, color='orange', linestyle='--', linewidth=2, label=f'CI Upper: {ci_upper:.2f}')
    axes[0].set_xlabel('Value', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('Sample Distribution + CI', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 표본 평균의 표집분포
    x = np.linspace(mean - 4*se, mean + 4*se, 1000)
    y = t.pdf((x - mean) / se, df) / se
    axes[1].plot(x, y, 'b-', linewidth=2, label='Sampling Distribution')
    axes[1].axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
    axes[1].fill_between(x, 0, y, where=(x >= ci_lower) & (x <= ci_upper), 
                          alpha=0.3, color='orange', label=f'{confidence*100:.0f}% CI')
    axes[1].set_xlabel('Sample Mean', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].set_title('Sampling Distribution of Mean', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return (ci_lower, ci_upper), mean, se

# 신뢰구간 계산
ci_95, sample_mean, se = calculate_mean_ci(sample_data, confidence=0.95)

# 다양한 신뢰수준 비교
print(f"\n" + "=" * 70)
print(f"다양한 신뢰수준 비교")
print(f"=" * 70)

for conf in [0.90, 0.95, 0.99]:
    ci, _, _ = calculate_mean_ci(sample_data, confidence=conf)
    width = ci[1] - ci[0]
    print(f"{conf*100:.0f}% CI: [{ci[0]:.2f}, {ci[1]:.2f}] (폭: {width:.2f})")

print(f"\n💡 신뢰수준 ↑ → 구간 폭 ↑ (더 보수적)")
```

### 3.4 비율의 신뢰구간

```python
def calculate_proportion_ci(successes, n, confidence=0.95, method='wilson'):
    """
    비율의 신뢰구간 계산
    
    Parameters:
    -----------
    successes : int
        성공 횟수
    n : int
        시행 횟수
    confidence : float
    method : str
        'normal', 'wilson', 'clopper-pearson'
    
    Returns:
    --------
    ci : tuple
    """
    print(f"\n" + "=" * 70)
    print(f"📊 비율의 {confidence*100:.0f}% 신뢰구간 ({method.upper()})")
    print(f="=" * 70)
    
    # 표본 비율
    p_hat = successes / n
    
    print(f"\n기본 정보:")
    print(f"  성공 횟수: {successes}")
    print(f"  시행 횟수: {n}")
    print(f"  표본 비율 (p̂): {p_hat:.4f} ({p_hat*100:.2f}%)")
    
    alpha = 1 - confidence
    z = norm.ppf(1 - alpha/2)
    
    if method == 'normal':
        # Wald 방법 (정규근사)
        se = np.sqrt(p_hat * (1 - p_hat) / n)
        ci_lower = p_hat - z * se
        ci_upper = p_hat + z * se
        
        print(f"\n정규근사 방법 (Wald):")
        print(f"  표준오차: {se:.4f}")
        print(f"  z-임계값: {z:.4f}")
        
        # 주의사항
        if n * p_hat < 10 or n * (1-p_hat) < 10:
            print(f"\n⚠️  경고: n×p 또는 n×(1-p) < 10")
            print(f"   정규근사 부정확할 수 있음 → Wilson 방법 권장")
    
    elif method == 'wilson':
        # Wilson score 방법 (더 정확)
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2*n)) / denominator
        margin = z * np.sqrt(p_hat * (1-p_hat) / n + z**2 / (4*n**2)) / denominator
        
        ci_lower = center - margin
        ci_upper = center + margin
        
        print(f"\nWilson Score 방법 (더 정확):")
        print(f"  중심 조정: {center:.4f}")
    
    elif method == 'clopper-pearson':
        # Clopper-Pearson (정확 방법, 보수적)
        from scipy.stats import beta
        ci_lower = beta.ppf(alpha/2, successes, n - successes + 1) if successes > 0 else 0
        ci_upper = beta.ppf(1 - alpha/2, successes + 1, n - successes) if successes < n else 1
        
        print(f"\nClopper-Pearson (정확 방법):")
        print(f"  베타 분포 기반")
    
    # 신뢰구간 (0~1 범위로 제한)
    ci_lower = max(0, ci_lower)
    ci_upper = min(1, ci_upper)
    
    print(f"\n{confidence*100:.0f}% 신뢰구간:")
    print(f"  [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  ([{ci_lower*100:.2f}%, {ci_upper*100:.2f}%])")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 비율 시각화
    categories = ['Successes', 'Failures']
    counts = [successes, n - successes]
    colors = ['green', 'red']
    axes[0].bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title(f'Proportion: {p_hat*100:.2f}% ({successes}/{n})', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 신뢰구간 시각화
    axes[1].errorbar([0], [p_hat], 
                      yerr=[[p_hat - ci_lower], [ci_upper - p_hat]],
                      fmt='o', markersize=10, capsize=10, capthick=2,
                      color='blue', label=f'{confidence*100:.0f}% CI')
    axes[1].axhline(p_hat, color='blue', linestyle='--', alpha=0.5)
    axes[1].set_ylim([0, 1])
    axes[1].set_ylabel('Proportion', fontsize=11)
    axes[1].set_title(f'{confidence*100:.0f}% Confidence Interval', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_xticks([])
    
    plt.tight_layout()
    plt.show()
    
    return (ci_lower, ci_upper)

# 비율 신뢰구간 계산
# 예: 500명 중 53명 전환 (10.6%)
ci_prop_wilson = calculate_proportion_ci(53, 500, confidence=0.95, method='wilson')

# 방법 비교
print(f"\n" + "=" * 70)
print(f"비율 신뢰구간 방법 비교")
print(f="=" * 70)

for method in ['normal', 'wilson', 'clopper-pearson']:
    ci = calculate_proportion_ci(53, 500, confidence=0.95, method=method)
    print(f"{method.capitalize():20s}: [{ci[0]*100:5.2f}%, {ci[1]*100:5.2f}%]")
```

### 3.5 Bootstrap 신뢰구간

```python
def bootstrap_ci(data, statistic_func, n_iterations=10000, confidence=0.95, seed=42):
    """
    Bootstrap 신뢰구간 계산
    
    Parameters:
    -----------
    data : array-like
        원본 데이터
    statistic_func : function
        계산할 통계량 함수 (예: np.mean, np.median)
    n_iterations : int
        Bootstrap 반복 횟수
    confidence : float
    seed : int
    
    Returns:
    --------
    ci : tuple
        (하한, 상한)
    bootstrap_dist : array
        Bootstrap 분포
    """
    print(f"\n" + "=" * 70)
    print(f"📊 Bootstrap {confidence*100:.0f}% 신뢰구간")
    print(f"=" * 70)
    
    np.random.seed(seed)
    
    # 원본 통계량
    original_stat = statistic_func(data)
    
    print(f"\n원본 통계량: {original_stat:.4f}")
    print(f"Bootstrap 반복: {n_iterations:,}회")
    print(f"Bootstrap 샘플 크기: {len(data)}")
    
    # Bootstrap
    print(f"\n⏳ Bootstrap 실행 중...")
    bootstrap_stats = []
    
    for i in range(n_iterations):
        # 복원추출 리샘플링
        bootstrap_sample = resample(data, replace=True, n_samples=len(data))
        bootstrap_stat = statistic_func(bootstrap_sample)
        bootstrap_stats.append(bootstrap_stat)
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    print(f"✅ 완료!")
    
    # 백분위수 방법 (Percentile Method)
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
    
    # Bootstrap 통계
    boot_mean = np.mean(bootstrap_stats)
    boot_std = np.std(bootstrap_stats)
    boot_bias = boot_mean - original_stat
    
    print(f"\nBootstrap 분포:")
    print(f"  평균: {boot_mean:.4f}")
    print(f"  표준편차: {boot_std:.4f}")
    print(f"  Bias: {boot_bias:.4f} (원본 대비 편향)")
    
    print(f"\n{confidence*100:.0f}% 신뢰구간 (Percentile Method):")
    print(f"  [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bootstrap 분포
    axes[0].hist(bootstrap_stats, bins=50, alpha=0.7, edgecolor='black', density=True)
    axes[0].axvline(original_stat, color='red', linestyle='--', linewidth=2, 
                    label=f'Original: {original_stat:.2f}')
    axes[0].axvline(ci_lower, color='orange', linestyle='--', linewidth=2, 
                    label=f'CI Lower: {ci_lower:.2f}')
    axes[0].axvline(ci_upper, color='orange', linestyle='--', linewidth=2, 
                    label=f'CI Upper: {ci_upper:.2f}')
    axes[0].set_xlabel('Statistic Value', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('Bootstrap Distribution', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q Plot
    stats.probplot(bootstrap_stats, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Normality Check)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n💡 Bootstrap 장점:")
    print(f"  - 분포 가정 불필요")
    print(f"  - 중앙값, 분산 등 복잡한 통계량에도 적용")
    print(f"  - 소표본에서도 안정적")
    
    return (ci_lower, ci_upper), bootstrap_stats

# Bootstrap 평균 신뢰구간
ci_boot_mean, boot_dist_mean = bootstrap_ci(
    sample_data, 
    statistic_func=np.mean,
    n_iterations=10000,
    confidence=0.95
)

# Bootstrap 중앙값 신뢰구간 (비대칭 통계량)
ci_boot_median, boot_dist_median = bootstrap_ci(
    sample_data,
    statistic_func=np.median,
    n_iterations=10000,
    confidence=0.95
)

# 표준편차 신뢰구간
ci_boot_std, boot_dist_std = bootstrap_ci(
    sample_data,
    statistic_func=lambda x: np.std(x, ddof=1),
    n_iterations=10000,
    confidence=0.95
)
```

### 3.6 예측 구간 (Prediction Interval)

```python
def calculate_prediction_interval(model_predictions, actual_values, new_X, confidence=0.95):
    """
    예측 구간 계산 (개별 예측의 불확실성)
    
    Parameters:
    -----------
    model_predictions : array
        모델의 예측값 (train/test)
    actual_values : array
        실제값
    new_X : float or array
        새로운 입력값 (예측할 점)
    confidence : float
    
    Returns:
    --------
    pi : tuple
        (하한, 상한)
    """
    print(f"\n" + "=" * 70)
    print(f"📊 예측 구간 (Prediction Interval) - {confidence*100:.0f}%")
    print(f"=" * 70)
    
    # 잔차 계산
    residuals = actual_values - model_predictions
    
    # 잔차의 표준편차 (MSE의 제곱근)
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    
    print(f"\n모델 성능:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {np.mean(np.abs(residuals)):.4f}")
    
    # 새로운 점의 예측값 (여기서는 평균으로 근사)
    prediction = np.mean(model_predictions)
    
    # 예측 구간
    # PI = ŷ ± t_(α/2) × RMSE × √(1 + 1/n)
    n = len(model_predictions)
    alpha = 1 - confidence
    t_crit = t.ppf(1 - alpha/2, n - 2)  # 자유도 n-2 (회귀)
    
    # 예측 구간은 신뢰구간보다 넓음 (개별 변동성 포함)
    pi_margin = t_crit * rmse * np.sqrt(1 + 1/n)
    pi_lower = prediction - pi_margin
    pi_upper = prediction + pi_margin
    
    # 신뢰구간 (평균 예측)
    ci_margin = t_crit * rmse / np.sqrt(n)
    ci_lower = prediction - ci_margin
    ci_upper = prediction + ci_margin
    
    print(f"\n예측값: {prediction:.4f}")
    
    print(f"\n신뢰구간 (평균 예측의 불확실성):")
    print(f"  [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  폭: {ci_upper - ci_lower:.4f}")
    
    print(f"\n예측 구간 (개별 예측의 불확실성):")
    print(f"  [{pi_lower:.4f}, {pi_upper:.4f}]")
    print(f"  폭: {pi_upper - pi_lower:.4f}")
    
    print(f"\n💡 차이:")
    print(f"  예측 구간이 신뢰구간보다 {(pi_upper - pi_lower) / (ci_upper - ci_lower):.1f}배 넓음")
    print(f"  이유: 개별 관측값의 변동성 포함")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 잔차 분포
    axes[0].hist(residuals, bins=30, alpha=0.7, edgecolor='black', density=True)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Mean=0')
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[0].plot(x, norm.pdf(x, 0, rmse), 'g-', linewidth=2, label=f'N(0, {rmse:.2f})')
    axes[0].set_xlabel('Residuals', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('Residual Distribution', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 신뢰구간 vs 예측 구간
    axes[1].errorbar([0], [prediction],
                      yerr=[[prediction - ci_lower], [ci_upper - prediction]],
                      fmt='o', markersize=10, capsize=10, capthick=2,
                      color='blue', label=f'{confidence*100:.0f}% CI (Mean)')
    axes[1].errorbar([0.2], [prediction],
                      yerr=[[prediction - pi_lower], [pi_upper - prediction]],
                      fmt='s', markersize=10, capsize=10, capthick=2,
                      color='red', label=f'{confidence*100:.0f}% PI (Individual)')
    axes[1].set_ylabel('Value', fontsize=11)
    axes[1].set_title('Confidence Interval vs Prediction Interval', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_xticks([])
    
    plt.tight_layout()
    plt.show()
    
    return (pi_lower, pi_upper), (ci_lower, ci_upper)

# 예측 구간 예시
# 가상의 모델 예측 및 실제값
np.random.seed(42)
true_values = np.random.normal(100, 15, 200)
predictions = true_values + np.random.normal(0, 10, 200)  # RMSE ≈ 10

pi, ci = calculate_prediction_interval(predictions, true_values, new_X=100, confidence=0.95)
```

### 3.7 표본 크기와 정밀도의 관계

```python
def analyze_sample_size_effect(population_std=15, confidence=0.95):
    """
    표본 크기가 신뢰구간 폭에 미치는 영향 분석
    """
    print(f"\n" + "=" * 70)
    print(f"📊 표본 크기와 신뢰구간 폭의 관계")
    print(f"=" * 70)
    
    # 다양한 표본 크기
    sample_sizes = np.array([10, 30, 50, 100, 200, 500, 1000, 2000, 5000])
    
    # 각 표본 크기에 대한 신뢰구간 폭 계산
    alpha = 1 - confidence
    
    ci_widths = []
    for n in sample_sizes:
        se = population_std / np.sqrt(n)
        t_crit = t.ppf(1 - alpha/2, n - 1)
        margin = t_crit * se
        width = 2 * margin
        ci_widths.append(width)
        
        print(f"n={n:5d}: SE={se:6.3f}, 폭={width:6.3f}")
    
    ci_widths = np.array(ci_widths)
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 샘플 크기 vs 신뢰구간 폭
    axes[0].plot(sample_sizes, ci_widths, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Sample Size (n)', fontsize=11)
    axes[0].set_ylabel('CI Width', fontsize=11)
    axes[0].set_title(f'{confidence*100:.0f}% CI Width vs Sample Size', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 로그 스케일
    axes[1].plot(sample_sizes, ci_widths, 'o-', linewidth=2, markersize=8)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Sample Size (n, log scale)', fontsize=11)
    axes[1].set_ylabel('CI Width', fontsize=11)
    axes[1].set_title('CI Width vs Sample Size (Log Scale)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n💡 핵심 인사이트:")
    print(f"  - 정밀도 ∝ 1/√n")
    print(f"  - n을 4배 → 폭이 1/2 (정밀도 2배)")
    print(f"  - n=100: 폭={ci_widths[np.where(sample_sizes==100)[0][0]]:.2f}")
    print(f"  - n=400: 폭={ci_widths[np.where(sample_sizes==500)[0][0]]:.2f} (약 1/2)")
    print(f"  - 큰 n에서는 개선 효과 둔화")

# 표본 크기 효과 분석
analyze_sample_size_effect(population_std=15, confidence=0.95)
```

### 3.8 신뢰구간 시뮬레이션 (이해 돕기)

```python
def simulate_confidence_intervals(n_simulations=100, sample_size=50, 
                                   population_mean=100, population_std=15,
                                   confidence=0.95, seed=42):
    """
    신뢰구간 시뮬레이션: 100번 중 몇 번이 진짜 모수를 포함하는가?
    """
    print(f"\n" + "=" * 70)
    print(f"📊 신뢰구간 시뮬레이션 (Coverage 확인)")
    print(f="=" * 70)
    
    np.random.seed(seed)
    
    # 진짜 모수
    true_mean = population_mean
    
    # 시뮬레이션
    coverage_count = 0
    intervals = []
    
    for i in range(n_simulations):
        # 표본 추출
        sample = np.random.normal(true_mean, population_std, sample_size)
        
        # 신뢰구간 계산
        mean = np.mean(sample)
        std = np.std(sample, ddof=1)
        se = std / np.sqrt(sample_size)
        
        alpha = 1 - confidence
        t_crit = t.ppf(1 - alpha/2, sample_size - 1)
        
        ci_lower = mean - t_crit * se
        ci_upper = mean + t_crit * se
        
        # 진짜 모수 포함 여부
        contains_true = ci_lower <= true_mean <= ci_upper
        if contains_true:
            coverage_count += 1
        
        intervals.append((ci_lower, ci_upper, mean, contains_true))
    
    # Coverage rate
    coverage_rate = coverage_count / n_simulations
    
    print(f"\n모집단:")
    print(f"  μ (진짜 평균): {true_mean}")
    print(f"  σ (진짜 표준편차): {population_std}")
    
    print(f"\n시뮬레이션 설정:")
    print(f"  반복 횟수: {n_simulations}")
    print(f"  표본 크기: {sample_size}")
    print(f"  신뢰수준: {confidence*100:.0f}%")
    
    print(f"\n결과:")
    print(f"  진짜 μ를 포함한 신뢰구간: {coverage_count}/{n_simulations}")
    print(f"  Coverage rate: {coverage_rate:.2%}")
    print(f"  이론적 예상: {confidence:.2%}")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 신뢰구간 플롯 (처음 50개만)
    display_n = min(50, n_simulations)
    for i in range(display_n):
        ci_lower, ci_upper, mean, contains = intervals[i]
        color = 'green' if contains else 'red'
        axes[0].plot([ci_lower, ci_upper], [i, i], color=color, linewidth=1.5, alpha=0.7)
        axes[0].plot(mean, i, 'o', color=color, markersize=4)
    
    axes[0].axvline(true_mean, color='blue', linestyle='--', linewidth=2, label=f'True μ={true_mean}')
    axes[0].set_xlabel('Value', fontsize=11)
    axes[0].set_ylabel('Simulation #', fontsize=11)
    axes[0].set_title(f'First {display_n} Confidence Intervals\n(Green: Contains μ, Red: Misses μ)', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Coverage histogram
    axes[1].bar(['Contains μ', 'Misses μ'], 
                [coverage_count, n_simulations - coverage_count],
                color=['green', 'red'], alpha=0.7, edgecolor='black')
    axes[1].axhline(n_simulations * confidence, color='blue', linestyle='--', 
                    linewidth=2, label=f'Expected: {confidence*100:.0f}%')
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title(f'Coverage Summary ({coverage_rate:.1%})', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n💡 해석:")
    print(f"  '{confidence*100:.0f}% 신뢰구간'의 의미:")
    print(f"  → 이 방법으로 100번 구간을 만들면,")
    print(f"    약 {confidence*100:.0f}번은 진짜 μ를 포함한다")
    print(f"  → 개별 구간이 μ를 포함할 '확률'이 아님!")

# 시뮬레이션 실행
simulate_confidence_intervals(
    n_simulations=100,
    sample_size=50,
    population_mean=100,
    population_std=15,
    confidence=0.95
)
```

### 3.9 베이지안 추론 (간단 예시)

```python
def bayesian_inference_simple(prior_mean, prior_std, data, likelihood_std):
    """
    간단한 베이지안 추론 (정규-정규 conjugate)
    
    사전분포 + 데이터 → 사후분포
    
    Parameters:
    -----------
    prior_mean : float
        사전 평균
    prior_std : float
        사전 표준편차
    data : array
        관측 데이터
    likelihood_std : float
        우도 표준편차 (알려진 경우)
    """
    print(f"\n" + "=" * 70)
    print(f"📊 베이지안 추론 (Bayesian Inference)")
    print(f="=" * 70)
    
    # 데이터 통계량
    n = len(data)
    data_mean = np.mean(data)
    
    # 사후분포 계산 (정규-정규 conjugate)
    prior_precision = 1 / prior_std**2
    likelihood_precision = n / likelihood_std**2
    
    posterior_precision = prior_precision + likelihood_precision
    posterior_std = 1 / np.sqrt(posterior_precision)
    
    posterior_mean = (prior_precision * prior_mean + likelihood_precision * data_mean) / posterior_precision
    
    # Credible Interval (베이지안 신뢰구간)
    ci_lower = posterior_mean - 1.96 * posterior_std
    ci_upper = posterior_mean + 1.96 * posterior_std
    
    print(f"\n사전분포 (Prior):")
    print(f"  평균: {prior_mean:.4f}")
    print(f"  표준편차: {prior_std:.4f}")
    
    print(f"\n데이터 (Likelihood):")
    print(f"  샘플 크기: {n}")
    print(f"  표본 평균: {data_mean:.4f}")
    
    print(f"\n사후분포 (Posterior):")
    print(f"  평균: {posterior_mean:.4f}")
    print(f"  표준편차: {posterior_std:.4f}")
    
    print(f"\n95% Credible Interval:")
    print(f"  [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    print(f"\n💡 베이지안 해석:")
    print(f"  모수가 이 구간에 있을 확률이 95%")
    print(f"  (빈도주의 신뢰구간과 해석이 다름!)")
    
    # 시각화
    x = np.linspace(prior_mean - 4*prior_std, prior_mean + 4*prior_std, 1000)
    
    plt.figure(figsize=(12, 6))
    
    # 사전분포
    prior_pdf = norm.pdf(x, prior_mean, prior_std)
    plt.plot(x, prior_pdf, 'b-', linewidth=2, label='Prior')
    
    # 우도 (데이터)
    likelihood_pdf = norm.pdf(x, data_mean, likelihood_std / np.sqrt(n))
    plt.plot(x, likelihood_pdf, 'g-', linewidth=2, label='Likelihood')
    
    # 사후분포
    posterior_pdf = norm.pdf(x, posterior_mean, posterior_std)
    plt.plot(x, posterior_pdf, 'r-', linewidth=2, label='Posterior')
    
    # Credible interval
    mask = (x >= ci_lower) & (x <= ci_upper)
    plt.fill_between(x[mask], 0, posterior_pdf[mask], alpha=0.3, color='red', 
                      label='95% Credible Interval')
    
    plt.xlabel('μ', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Bayesian Inference: Prior → Posterior', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\n사전 → 사후 변화:")
    print(f"  평균: {prior_mean:.2f} → {posterior_mean:.2f}")
    print(f"  표준편차: {prior_std:.2f} → {posterior_std:.2f}")
    print(f"  불확실성 감소: {(1 - posterior_std/prior_std)*100:.1f}%")

# 베이지안 추론 예시
# 사전: μ ~ N(95, 10) (이전 연구 기반)
# 데이터: 새로운 실험 결과
bayesian_inference_simple(
    prior_mean=95,
    prior_std=10,
    data=sample_data,
    likelihood_std=15
)
```

### 3.10 종합 비교: 신뢰구간 방법들

```python
def compare_ci_methods(data, confidence=0.95):
    """
    다양한 신뢰구간 방법 비교
    """
    print(f"\n" + "=" * 70)
    print(f"📊 신뢰구간 방법 종합 비교")
    print(f"=" * 70)
    
    # 1. 모수적 방법 (t-분포)
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / np.sqrt(n)
    
    alpha = 1 - confidence
    t_crit = t.ppf(1 - alpha/2, n - 1)
    
    ci_parametric = (mean - t_crit * se, mean + t_crit * se)
    
    # 2. Bootstrap (백분위수)
    np.random.seed(42)
    bootstrap_means = []
    for _ in range(10000):
        sample = resample(data, n_samples=len(data))
        bootstrap_means.append(np.mean(sample))
    
    ci_bootstrap = (np.percentile(bootstrap_means, alpha/2 * 100),
                    np.percentile(bootstrap_means, (1 - alpha/2) * 100))
    
    # 3. Bootstrap (BCa - Bias-Corrected and Accelerated)
    # 간단 버전
    ci_bootstrap_bca = ci_bootstrap  # 실제로는 더 복잡
    
    # 결과 출력
    print(f"\n표본 통계량:")
    print(f"  n = {n}")
    print(f"  mean = {mean:.4f}")
    print(f"  std = {std:.4f}")
    print(f"  SE = {se:.4f}")
    
    print(f"\n{confidence*100:.0f}% 신뢰구간 비교:")
    print(f"-" * 70)
    print(f"{'Method':<25} {'Lower':<12} {'Upper':<12} {'Width':<12}")
    print(f"-" * 70)
    
    methods = [
        ('t-분포 (모수적)', ci_parametric),
        ('Bootstrap 백분위수', ci_bootstrap),
        ('Bootstrap BCa', ci_bootstrap_bca)
    ]
    
    for method_name, (lower, upper) in methods:
        width = upper - lower
        print(f"{method_name:<25} {lower:<12.4f} {upper:<12.4f} {width:<12.4f}")
    
    # 시각화
    plt.figure(figsize=(10, 6))
    
    y_pos = range(len(methods))
    for i, (method_name, (lower, upper)) in enumerate(methods):
        plt.plot([lower, upper], [i, i], 'o-', linewidth=3, markersize=8, label=method_name)
        plt.plot(mean, i, 's', markersize=10, color='red')
    
    plt.axvline(mean, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'Mean={mean:.2f}')
    plt.yticks(y_pos, [m[0] for m in methods])
    plt.xlabel('Value', fontsize=12)
    plt.title(f'{confidence*100:.0f}% Confidence Intervals Comparison', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    print(f"\n💡 방법 선택 가이드:")
    print(f"  - t-분포: 정규성 만족 시 (빠름, 이론적)")
    print(f"  - Bootstrap: 정규성 불확실 시 (안전, 유연)")
    print(f"  - Bootstrap BCa: 비대칭 분포 시 (더 정확)")

# 종합 비교
compare_ci_methods(sample_data, confidence=0.95)
```

---

## 4. 예시

### 4.1 실전 예제: 신약 임상시험 결과 보고

```python
print("=" * 70)
print("📈 비즈니스 시나리오: 신약 임상시험")
print("=" * 70)

print("\n목표:")
print("- 신약의 혈압 강하 효과 추정")
print("- 95% 신뢰구간으로 효과 범위 제시")

print("\n데이터:")
print("- 환자 100명")
print("- 평균 혈압 감소: 8.5 mmHg")
print("- 표준편차: 4.2 mmHg")

print("\n분석:")
print("1. 95% 신뢰구간 계산")
print("   SE = 4.2 / √100 = 0.42")
print("   CI = 8.5 ± 1.98 × 0.42")
print("   CI = [7.67, 9.33] mmHg")

print("\n2. 해석")
print("   모집단 평균 혈압 감소량이")
print("   7.67~9.33 mmHg 사이에 있다고")
print("   95% 신뢰할 수 있다")

print("\n3. 의사결정")
print("   목표: 5 mmHg 이상 감소")
print("   → 95% CI 하한(7.67) > 5")
print("   → 목표 달성 확신")

print("\n결론:")
print("  ✅ 신약은 통계적으로 유의미하고")
print("  ✅ 임상적으로도 의미 있는 효과")
```

---

## 5. 에이전트 매핑

### 5.1 담당 에이전트

| 작업 | Primary Agent | Supporting Agents |
|------|--------------|-------------------|
| 신뢰구간 계산 | `data-scientist` | - |
| Bootstrap 분석 | `data-scientist` | - |
| 예측 구간 계산 | `data-scientist` | `ml-modeling-specialist` |
| 베이지안 추론 | `data-scientist` | - |
| 결과 해석 및 보고 | `data-scientist` | `technical-documentation-writer` |

### 5.2 관련 스킬

**Scientific Skills**:
- `scipy.stats` (t, norm, bootstrap)
- `sklearn.utils` (resample)
- `pandas`, `numpy` (데이터 처리)
- `matplotlib`, `seaborn` (시각화)

---

## 6. 필요 라이브러리

### 6.1 필수 라이브러리

```bash
# 통계 분석
pip install scipy==1.12.0
pip install statsmodels==0.14.1

# 머신러닝 (Bootstrap)
pip install scikit-learn==1.4.0

# 데이터 처리
pip install pandas==2.2.0
pip install numpy==1.26.3

# 시각화
pip install matplotlib==3.8.2
pip install seaborn==0.13.1

# 베이지안 (선택)
# pip install pymc3
```

---

## 7. 체크포인트

### 7.1 분석 전 체크리스트

- [ ] **목적 명확화**
  - [ ] 모수 추정인가 예측인가?
  - [ ] 신뢰수준 결정 (90%, 95%, 99%)

- [ ] **데이터 확인**
  - [ ] 샘플 크기 충분한가?
  - [ ] 정규성 가정 확인 (모수적 방법 시)

### 7.2 분석 중 체크리스트

- [ ] **방법 선택**
  - [ ] 정규성 만족 → t-분포
  - [ ] 정규성 불만족 → Bootstrap
  - [ ] 복잡한 통계량 → Bootstrap

- [ ] **계산 정확성**
  - [ ] 자유도 확인
  - [ ] 표준오차 계산 확인

### 7.3 분석 후 체크리스트

- [ ] **해석**
  - [ ] 올바른 해석인가?
  - [ ] 실무적 의미 고려

- [ ] **보고**
  - [ ] 신뢰구간과 점추정 모두 제시
  - [ ] 시각화 포함

---

## 8. 트러블슈팅

### 8.1 일반적 오류

**문제 1: 신뢰구간이 너무 넓음**

```python
# 원인 1: 샘플 크기 부족
# 해결: 샘플 증가 (CI 폭 ∝ 1/√n)

# 원인 2: 데이터 변동성 큼
# 해결: 측정 방법 개선, 층화추출
```

**문제 2: Bootstrap이 너무 느림**

```python
# 해결 1: 반복 횟수 조정
n_iterations = 1000  # 10000 → 1000

# 해결 2: 병렬 처리
from joblib import Parallel, delayed
bootstrap_stats = Parallel(n_jobs=-1)(
    delayed(calculate_stat)(resample(data)) 
    for _ in range(n_iterations)
)
```

**문제 3: 신뢰구간 vs 예측구간 혼동**

```python
# 신뢰구간: 평균의 불확실성 (좁음)
ci = mean ± t × (s/√n)

# 예측구간: 개별값의 불확실성 (넓음)
pi = mean ± t × s × √(1 + 1/n)

# 선택: 목적에 따라
# - 모평균 추정 → 신뢰구간
# - 개별 예측 → 예측구간
```

### 8.2 해석 관련

**Q1: 95% 신뢰구간 [98, 102]의 의미는?**

```
A: 자주 오해되는 개념

❌ 잘못된 해석:
"μ가 [98, 102]에 있을 확률이 95%"
→ μ는 고정값, 확률 변수 아님

✅ 올바른 해석:
"이 방법으로 100번 신뢰구간을 구하면,
 약 95번은 진짜 μ를 포함한다"

✅ 실무적 해석:
"μ가 98~102 사이에 있다고
 95% 신뢰할 수 있다"
```

**Q2: 신뢰수준을 99%로 높이면?**

```
A: 구간이 더 넓어짐

99% 신뢰구간 > 95% 신뢰구간

이유:
- 더 보수적 (더 확신하기 위해)
- z-값 증가: 1.96 → 2.58

Trade-off:
- 높은 신뢰수준 → 넓은 구간
- 정밀도 ↓ vs 확신 ↑
```

**Q3: Bootstrap이 항상 더 좋은가?**

```
A: 상황에 따라 다름

Bootstrap 장점:
✅ 분포 가정 불필요
✅ 복잡한 통계량 가능
✅ 비대칭 분포 처리

Bootstrap 단점:
❌ 계산 비용 높음
❌ 원본 데이터가 모집단 잘 대표해야
❌ 매우 작은 샘플(n<30)에서 불안정

권장:
- 정규성 만족 + 단순 통계량 → t-분포
- 정규성 불만족 or 복잡한 통계량 → Bootstrap
```

### 8.3 샘플 크기 결정

```python
def required_sample_size(margin_of_error, std, confidence=0.95):
    """
    원하는 오차한계를 위한 필요 샘플 크기
    """
    z = norm.ppf(1 - (1-confidence)/2)
    n = (z * std / margin_of_error)**2
    return int(np.ceil(n))

# 예시: ME=2, σ=15, 95% CI
n = required_sample_size(margin_of_error=2, std=15, confidence=0.95)
print(f"필요 샘플 크기: {n}")  # ≈ 217
```

---

## 9. 참고 자료

### 9.1 공식 문서

- **SciPy Stats**: https://docs.scipy.org/doc/scipy/reference/stats.html
- **Scikit-learn Bootstrap**: https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
- **Statsmodels**: https://www.statsmodels.org/stable/index.html

### 9.2 베스트 프랙티스

1. **신뢰구간 보고 형식**
   ```
   "평균 혈압 감소는 8.5 mmHg (95% CI: 7.67-9.33)였다."
   
   또는
   
   "평균 = 8.5, 95% CI [7.67, 9.33]"
   ```

2. **방법 선택 가이드**
   ```
   1. 정규성 확인 (Shapiro-Wilk, Q-Q plot)
   2. 정규성 만족 + 단순 통계량 → t-분포
   3. 정규성 불만족 or 복잡 통계량 → Bootstrap
   4. 사전 정보 활용 → 베이지안
   ```

3. **신뢰구간 폭 최소화**
   ```
   1. 샘플 크기 증가 (CI ∝ 1/√n)
   2. 측정 정밀도 향상 (σ 감소)
   3. 층화추출 (분산 감소)
   ```

### 9.3 추가 학습 자료

- **신뢰구간 직관적 이해**: https://rpsychologist.com/d3/ci/
- **Bootstrap 설명**: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
- **베이지안 vs 빈도주의**: https://www.probabilisticworld.com/frequentist-bayesian-approaches-inferential-statistics/

---

## 10. 요약

### 10.1 핵심 메시지

통계적 추론은 표본으로부터 모집단을 추정하고 그 불확실성을 정량화하는 과학입니다. 신뢰구간, Bootstrap, 예측구간 등 다양한 방법을 적절히 선택하여 데이터 기반 의사결정의 신뢰도를 높일 수 있습니다.

### 10.2 방법 선택 가이드

| 목적 | 추천 방법 | 특징 |
|------|----------|------|
| 평균 추정 (정규) | t-분포 CI | 빠름, 이론적 |
| 평균 추정 (비정규) | Bootstrap | 안전, 유연 |
| 복잡한 통계량 | Bootstrap | 범용 |
| 개별 예측 | Prediction Interval | 넓은 구간 |
| 사전 정보 활용 | 베이지안 | 확률적 해석 |

### 10.3 다음 단계

- **가설 검정**: `11-hypothesis-testing.md` 참고
- **회귀 분석**: 신뢰구간과 예측구간
- **베이지안 통계**: PyMC3, Stan 활용
- **실험 설계**: 최적 샘플 크기 결정

---

**작성일**: 2025-01-25  
**버전**: 1.0  
**난이도**: ⭐⭐⭐ (고급)  
**예상 소요 시간**: 3-4시간 (학습 및 실습)
