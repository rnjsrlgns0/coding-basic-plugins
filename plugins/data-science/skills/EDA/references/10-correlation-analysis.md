# 10. Correlation Analysis (상관관계 분석)

## 1. 개요

### 1.1 목적
상관관계 분석(Correlation Analysis)은 두 개 이상의 변수 간 선형/비선형 관계의 강도와 방향을 정량화하는 분석 기법입니다. 변수 간 관계를 이해하고, 다중공선성을 탐지하며, feature selection의 기초를 제공합니다.

### 1.2 적용 시기
- 변수 간 관계의 강도를 정량적으로 측정하고 싶을 때
- 타겟 변수와 강하게 연관된 features를 찾을 때
- 다중공선성(multicollinearity) 문제를 진단할 때
- Feature selection 및 차원 축소 전 변수 간 중복성 확인
- 인과관계(causation)가 아닌 연관성(association) 탐색

### 1.3 주요 기법
- **Pearson 상관계수**: 선형 관계 측정
- **Spearman 상관계수**: 단조 관계 측정 (순위 기반)
- **Kendall Tau**: 순위 일치도 측정
- **부분 상관관계**: 제3변수 통제 후 관계
- **VIF (분산팽창계수)**: 다중공선성 진단

---

## 2. 이론적 배경

### 2.1 상관관계 vs 인과관계

**핵심 원칙**: **"Correlation does not imply causation"**

```
예시 1: 아이스크림 판매량 ↔ 수영장 익사 사고
- 강한 양의 상관관계 (r=0.85)
- 하지만 인과관계 없음
- 제3변수: 여름 날씨 (confounding variable)

예시 2: 흡연 ↔ 폐암
- 강한 양의 상관관계
- 인과관계 존재 (흡연이 폐암 유발)
- 수십 년의 연구로 인과관계 입증
```

**상관관계 분석의 역할**:
- ✅ 변수 간 연관성 발견 (가설 생성)
- ✅ 예측 모델의 feature selection
- ❌ 인과관계 입증 (추가 연구 필요)

### 2.2 상관계수의 종류

#### Pearson 상관계수 (r)
- **측정**: 선형 관계의 강도와 방향
- **범위**: -1 ≤ r ≤ 1
- **가정**: 
  - 변수가 연속형
  - 선형 관계
  - 정규분포 (검정 시)
  - 이상치에 민감

#### Spearman 상관계수 (ρ)
- **측정**: 단조 관계 (순위 기반)
- **범위**: -1 ≤ ρ ≤ 1
- **장점**:
  - 비선형 단조 관계 탐지
  - 이상치에 강건
  - 정규성 가정 불필요

#### Kendall Tau (τ)
- **측정**: 순위 일치도
- **범위**: -1 ≤ τ ≤ 1
- **장점**:
  - 작은 샘플에서 더 정확
  - 해석이 직관적
  - 동점(tie)이 많을 때 유리

### 2.3 상관계수 해석 기준

```
|r| 값         강도          해석
0.0 - 0.1      없음          관계 없음
0.1 - 0.3      약함          약한 관계
0.3 - 0.5      중간          중간 정도 관계
0.5 - 0.7      강함          강한 관계
0.7 - 0.9      매우 강함     매우 강한 관계
0.9 - 1.0      거의 완벽     거의 완벽한 관계

주의: 도메인에 따라 기준이 다름
- 사회과학: |r| > 0.3이면 의미 있음
- 물리/공학: |r| > 0.7 정도는 되어야 의미 있음
```

### 2.4 시나리오

**시나리오 1: Feature Selection**
```
상황: 100개 features, 1개 target
목표: 모델 성능 향상 및 과적합 방지

분석:
1. 각 feature와 target의 상관계수 계산
2. |r| > 0.3인 features만 선택 (20개로 축소)
3. 선택된 features 간 상관관계 확인
4. |r| > 0.8인 쌍 중 하나 제거 (다중공선성)

결과:
- 최종 15개 features 선택
- 모델 정확도 유지하면서 학습 속도 3배 향상
```

**시나리오 2: 다중공선성 탐지**
```
상황: 회귀 모델에서 계수가 불안정
증상: 변수 추가 시 다른 변수 계수가 크게 변함

진단:
1. 상관관계 행렬 확인 → age & years_experience: r=0.95
2. VIF 계산 → age: VIF=18 (기준 10 초과)

해결:
- 두 변수 중 하나 제거 또는
- PCA로 통합하여 하나의 변수로 축소
```

**시나리오 3: 비선형 관계 탐지**
```
상황: 산점도에서 명확한 곡선 패턴
분석:
- Pearson r = 0.05 (약함)
- Spearman ρ = 0.78 (강함)

해석:
- 선형 관계는 약하지만 단조 증가 패턴 존재
- 로그 변환 또는 다항식 features 고려
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
from scipy.stats import pearsonr, spearmanr, kendalltau
import warnings
warnings.filterwarnings('ignore')

# 통계 모델링
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# 시각화 설정
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("coolwarm")
%matplotlib inline

# 한글 폰트 (선택)
plt.rcParams['font.family'] = 'AppleGothic'  # Mac
# plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False

# 출력 옵션
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 3)
```

### 3.2 샘플 데이터 생성

```python
# 다양한 상관관계 패턴을 가진 데이터 생성
np.random.seed(42)

def generate_correlation_data(n=500):
    """
    다양한 상관관계 패턴을 가진 데이터 생성
    """
    # 기본 변수
    x1 = np.random.normal(100, 15, n)
    
    # 강한 양의 선형 관계 (r ≈ 0.9)
    x2 = x1 + np.random.normal(0, 5, n)
    
    # 중간 양의 선형 관계 (r ≈ 0.5)
    x3 = x1 * 0.5 + np.random.normal(50, 20, n)
    
    # 약한 음의 선형 관계 (r ≈ -0.2)
    x4 = -0.2 * x1 + np.random.normal(100, 30, n)
    
    # 비선형 관계 (2차)
    x5 = 0.01 * (x1 - 100)**2 + np.random.normal(0, 10, n)
    
    # 로그 관계
    x6 = 20 * np.log(x1) + np.random.normal(0, 5, n)
    
    # 관계 없음 (r ≈ 0)
    x7 = np.random.normal(50, 15, n)
    
    # 이상치 포함 변수
    x8 = x1 * 0.6 + np.random.normal(0, 10, n)
    outlier_idx = np.random.choice(n, 10, replace=False)
    x8[outlier_idx] = np.random.uniform(200, 300, 10)
    
    df = pd.DataFrame({
        'x1_base': x1,
        'x2_strong_pos': x2,
        'x3_medium_pos': x3,
        'x4_weak_neg': x4,
        'x5_quadratic': x5,
        'x6_log': x6,
        'x7_no_corr': x7,
        'x8_outliers': x8
    })
    
    return df

# 데이터 생성
df = generate_correlation_data(500)
print(f"데이터 크기: {df.shape}")
print(f"\n기본 통계:")
print(df.describe())
```

### 3.3 Pearson 상관계수

```python
def calculate_pearson_correlation(df, method='pearson'):
    """
    Pearson 상관계수 행렬 계산 및 시각화
    
    Parameters:
    -----------
    df : DataFrame
        수치형 변수만 포함
    method : str
        'pearson', 'spearman', 'kendall'
    
    Returns:
    --------
    corr_matrix : DataFrame
        상관계수 행렬
    """
    # 상관계수 행렬 계산
    corr_matrix = df.corr(method=method)
    
    print(f"=" * 70)
    print(f"📊 {method.upper()} 상관계수 행렬")
    print(f"=" * 70)
    print(corr_matrix.round(3))
    
    # 히트맵 시각화
    plt.figure(figsize=(12, 10))
    
    # Mask for upper triangle (대각선 위쪽 제거)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8, "label": f"{method.capitalize()} Correlation"}
    )
    
    plt.title(f'{method.capitalize()} Correlation Heatmap', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()
    
    return corr_matrix

# Pearson 상관계수 계산
pearson_corr = calculate_pearson_correlation(df, method='pearson')
```

### 3.4 Spearman 및 Kendall 상관계수

```python
# Spearman 상관계수 (단조 관계)
spearman_corr = calculate_pearson_correlation(df, method='spearman')

# Kendall Tau (순위 일치도)
kendall_corr = calculate_pearson_correlation(df, method='kendall')

# 3가지 방법 비교
def compare_correlation_methods(df, var1, var2):
    """
    3가지 상관계수 방법 비교
    """
    x = df[var1]
    y = df[var2]
    
    # 각 상관계수 계산
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
    kendall_r, kendall_p = kendalltau(x, y)
    
    print(f"\n" + "=" * 70)
    print(f"📊 상관계수 비교: {var1} vs {var2}")
    print(f"=" * 70)
    print(f"{'Method':<15} {'Coefficient':<15} {'P-value':<15} {'Interpretation'}")
    print(f"-" * 70)
    print(f"{'Pearson':<15} {pearson_r:<15.3f} {pearson_p:<15.3e} {'선형 관계'}")
    print(f"{'Spearman':<15} {spearman_r:<15.3f} {spearman_p:<15.3e} {'단조 관계'}")
    print(f"{'Kendall':<15} {kendall_r:<15.3f} {kendall_p:<15.3e} {'순위 일치도'}")
    
    # 산점도 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 산점도
    axes[0].scatter(x, y, alpha=0.5, s=30)
    axes[0].set_xlabel(var1, fontsize=11)
    axes[0].set_ylabel(var2, fontsize=11)
    axes[0].set_title(f'Scatter Plot\nPearson r={pearson_r:.3f}', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 회귀선 추가
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    axes[0].plot(x, p(x), "r--", alpha=0.8, linewidth=2, label='Linear Fit')
    axes[0].legend()
    
    # 순위 산점도 (Spearman용)
    rank_x = stats.rankdata(x)
    rank_y = stats.rankdata(y)
    axes[1].scatter(rank_x, rank_y, alpha=0.5, s=30, color='green')
    axes[1].set_xlabel(f'{var1} (Rank)', fontsize=11)
    axes[1].set_ylabel(f'{var2} (Rank)', fontsize=11)
    axes[1].set_title(f'Rank Plot\nSpearman ρ={spearman_r:.3f}', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 비선형 관계 예시
compare_correlation_methods(df, 'x1_base', 'x5_quadratic')

# 선형 관계 예시
compare_correlation_methods(df, 'x1_base', 'x2_strong_pos')

# 관계 없음 예시
compare_correlation_methods(df, 'x1_base', 'x7_no_corr')
```

### 3.5 타겟 변수와의 상관관계 분석

```python
def analyze_target_correlation(df, target_col, top_n=10):
    """
    타겟 변수와 가장 상관관계가 높은 features 식별
    
    Parameters:
    -----------
    df : DataFrame
    target_col : str
        타겟 변수 이름
    top_n : int
        상위 몇 개 변수를 표시할지
    
    Returns:
    --------
    top_features : DataFrame
        상위 상관관계 변수 목록
    """
    # 타겟 변수와의 상관계수
    target_corr = df.corr()[target_col].drop(target_col).sort_values(
        key=abs, ascending=False
    )
    
    print(f"\n" + "=" * 70)
    print(f"📊 '{target_col}'와 가장 상관관계가 높은 변수 (Top {top_n})")
    print(f"=" * 70)
    print(f"{'Feature':<25} {'Correlation':<15} {'Strength'}")
    print(f"-" * 70)
    
    for feature, corr in target_corr.head(top_n).items():
        strength = get_correlation_strength(abs(corr))
        direction = "↑" if corr > 0 else "↓"
        print(f"{feature:<25} {direction} {corr:>6.3f}          {strength}")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 막대 그래프
    top_features = target_corr.head(top_n)
    colors = ['green' if x > 0 else 'red' for x in top_features.values]
    axes[0].barh(range(len(top_features)), top_features.values, color=colors, alpha=0.7)
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features.index)
    axes[0].set_xlabel('Correlation Coefficient', fontsize=11)
    axes[0].set_title(f'Top {top_n} Features Correlated with {target_col}', fontsize=12)
    axes[0].axvline(x=0, color='black', linewidth=0.8)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # 산점도 매트릭스 (상위 4개)
    top_4_features = target_corr.head(4).index.tolist()
    selected_cols = top_4_features + [target_col]
    
    # Pairplot
    sns.pairplot(
        df[selected_cols],
        diag_kind='kde',
        plot_kws={'alpha': 0.5, 's': 30},
        height=2
    )
    plt.suptitle(f'Pairplot: Top 4 Features vs {target_col}', y=1.02, fontsize=12)
    plt.tight_layout()
    plt.show()
    
    return target_corr.head(top_n)

def get_correlation_strength(abs_corr):
    """상관계수 강도 해석"""
    if abs_corr < 0.1:
        return "없음"
    elif abs_corr < 0.3:
        return "약함"
    elif abs_corr < 0.5:
        return "중간"
    elif abs_corr < 0.7:
        return "강함"
    elif abs_corr < 0.9:
        return "매우 강함"
    else:
        return "거의 완벽"

# 타겟 변수와의 상관관계 분석 (예시: x2_strong_pos를 타겟으로)
top_features = analyze_target_correlation(df, target_col='x2_strong_pos', top_n=7)
```

### 3.6 부분 상관관계 (Partial Correlation)

```python
def calculate_partial_correlation(df, x_col, y_col, control_cols):
    """
    부분 상관관계 계산: 제3변수를 통제한 후 x와 y의 관계
    
    Parameters:
    -----------
    df : DataFrame
    x_col : str
        관심 변수 1
    y_col : str
        관심 변수 2
    control_cols : list
        통제할 변수 목록
    
    Returns:
    --------
    partial_corr : float
        부분 상관계수
    """
    from scipy.stats import pearsonr
    
    # 잔차 계산 (제3변수의 영향 제거)
    def get_residuals(df, target, predictors):
        X = df[predictors]
        y = df[target]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        return model.resid
    
    # x와 y의 잔차
    resid_x = get_residuals(df, x_col, control_cols)
    resid_y = get_residuals(df, y_col, control_cols)
    
    # 잔차 간 상관계수 = 부분 상관계수
    partial_corr, p_value = pearsonr(resid_x, resid_y)
    
    # 일반 상관계수
    simple_corr, _ = pearsonr(df[x_col], df[y_col])
    
    print(f"\n" + "=" * 70)
    print(f"📊 부분 상관관계 분석")
    print(f"=" * 70)
    print(f"X: {x_col}")
    print(f"Y: {y_col}")
    print(f"통제 변수: {', '.join(control_cols)}")
    print(f"\n일반 상관계수 (Simple):   {simple_corr:.3f}")
    print(f"부분 상관계수 (Partial):  {partial_corr:.3f}")
    print(f"P-value:                   {p_value:.3e}")
    
    print(f"\n💡 해석:")
    if abs(partial_corr) < abs(simple_corr) * 0.5:
        print(f"  → 관계가 크게 약해짐: 통제 변수가 X-Y 관계를 매개")
    elif abs(partial_corr) > abs(simple_corr) * 1.5:
        print(f"  → 관계가 강해짐: 통제 변수가 관계를 억압(suppression)")
    else:
        print(f"  → 관계가 유지됨: X-Y는 통제 변수와 독립적")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 원래 관계
    axes[0].scatter(df[x_col], df[y_col], alpha=0.5, s=30)
    axes[0].set_xlabel(x_col, fontsize=11)
    axes[0].set_ylabel(y_col, fontsize=11)
    axes[0].set_title(f'Simple Correlation\nr = {simple_corr:.3f}', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 잔차 관계 (부분 상관)
    axes[1].scatter(resid_x, resid_y, alpha=0.5, s=30, color='green')
    axes[1].set_xlabel(f'{x_col} (residuals)', fontsize=11)
    axes[1].set_ylabel(f'{y_col} (residuals)', fontsize=11)
    axes[1].set_title(f'Partial Correlation\nr = {partial_corr:.3f}', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return partial_corr, p_value

# 부분 상관관계 예시
# x1과 x3의 관계를 x2를 통제한 후 확인
partial_corr, p_val = calculate_partial_correlation(
    df,
    x_col='x1_base',
    y_col='x3_medium_pos',
    control_cols=['x2_strong_pos']
)
```

### 3.7 다중공선성 진단 (VIF)

```python
def calculate_vif(df, exclude_cols=None):
    """
    VIF (Variance Inflation Factor) 계산
    
    VIF > 10: 심각한 다중공선성
    VIF > 5: 주의 필요
    VIF < 5: 문제 없음
    
    Parameters:
    -----------
    df : DataFrame
        수치형 변수만 포함
    exclude_cols : list
        제외할 컬럼 (예: 타겟 변수)
    
    Returns:
    --------
    vif_df : DataFrame
        VIF 값 테이블
    """
    # 변수 선택
    cols = df.columns.tolist()
    if exclude_cols:
        cols = [c for c in cols if c not in exclude_cols]
    
    X = df[cols]
    
    # VIF 계산
    vif_data = pd.DataFrame()
    vif_data["Feature"] = cols
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(len(cols))
    ]
    
    # 정렬
    vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)
    
    print(f"\n" + "=" * 70)
    print(f"📊 VIF (Variance Inflation Factor) 분석")
    print(f"=" * 70)
    print(f"{'Feature':<25} {'VIF':<10} {'Status'}")
    print(f"-" * 70)
    
    for idx, row in vif_data.iterrows():
        feature = row['Feature']
        vif = row['VIF']
        
        if vif > 10:
            status = "🚨 심각 (제거 권장)"
            color = '\033[91m'  # Red
        elif vif > 5:
            status = "⚠️  주의"
            color = '\033[93m'  # Yellow
        else:
            status = "✅ 정상"
            color = '\033[92m'  # Green
        
        print(f"{feature:<25} {vif:<10.2f} {color}{status}\033[0m")
    
    print(f"\n💡 다중공선성 해결 방법:")
    print(f"  1. VIF > 10인 변수 중 하나 제거")
    print(f"  2. 변수 결합 (예: PCA)")
    print(f"  3. Regularization (Ridge, Lasso)")
    
    # 시각화
    plt.figure(figsize=(10, 6))
    colors = ['red' if v > 10 else 'orange' if v > 5 else 'green' 
              for v in vif_data['VIF']]
    plt.barh(vif_data['Feature'], vif_data['VIF'], color=colors, alpha=0.7)
    plt.xlabel('VIF', fontsize=12)
    plt.title('VIF Analysis (다중공선성 진단)', fontsize=14)
    plt.axvline(x=5, color='orange', linestyle='--', linewidth=2, label='Threshold: 5')
    plt.axvline(x=10, color='red', linestyle='--', linewidth=2, label='Threshold: 10')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    return vif_data

# VIF 계산
vif_results = calculate_vif(df)
```

### 3.8 상관관계 행렬 필터링

```python
def filter_high_correlations(df, threshold=0.8, method='pearson'):
    """
    높은 상관관계를 가진 변수 쌍 식별
    
    Parameters:
    -----------
    df : DataFrame
    threshold : float
        상관계수 임계값 (절댓값)
    method : str
        상관계수 방법
    
    Returns:
    --------
    high_corr_pairs : DataFrame
        높은 상관관계 변수 쌍
    """
    # 상관계수 행렬
    corr_matrix = df.corr(method=method)
    
    # 상삼각 행렬만 (중복 제거)
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # 높은 상관관계 쌍 찾기
    high_corr_pairs = []
    for column in upper_tri.columns:
        high_corr = upper_tri[column][abs(upper_tri[column]) > threshold]
        for idx, corr in high_corr.items():
            high_corr_pairs.append({
                'Feature 1': idx,
                'Feature 2': column,
                'Correlation': corr,
                'Abs_Correlation': abs(corr)
            })
    
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values(
        'Abs_Correlation', ascending=False
    )
    
    if len(high_corr_df) == 0:
        print(f"\n✅ |r| > {threshold}인 변수 쌍이 없습니다.")
        return high_corr_df
    
    print(f"\n" + "=" * 70)
    print(f"🔍 높은 상관관계 변수 쌍 (|r| > {threshold})")
    print(f"=" * 70)
    print(high_corr_df.to_string(index=False))
    
    print(f"\n💡 권장 조치:")
    print(f"  - 두 변수 중 하나 제거 고려")
    print(f"  - 타겟과 더 높은 상관관계를 가진 변수 유지")
    print(f"  - 또는 PCA/feature engineering으로 결합")
    
    return high_corr_df

# 높은 상관관계 변수 쌍 찾기
high_corr_pairs = filter_high_correlations(df, threshold=0.8)
```

### 3.9 이상치의 영향 분석

```python
def analyze_outlier_effect_on_correlation(df, x_col, y_col, outlier_threshold=3):
    """
    이상치가 상관계수에 미치는 영향 분석
    
    Parameters:
    -----------
    df : DataFrame
    x_col, y_col : str
        분석할 변수
    outlier_threshold : float
        Z-score 임계값
    """
    x = df[x_col].copy()
    y = df[y_col].copy()
    
    # Z-score 계산
    z_x = np.abs(stats.zscore(x))
    z_y = np.abs(stats.zscore(y))
    
    # 이상치 마스크
    outlier_mask = (z_x > outlier_threshold) | (z_y > outlier_threshold)
    
    # 이상치 포함/제외 상관계수
    corr_with_outliers, _ = pearsonr(x, y)
    corr_without_outliers, _ = pearsonr(
        x[~outlier_mask], y[~outlier_mask]
    )
    
    # Spearman (이상치 강건)
    spearman_with, _ = spearmanr(x, y)
    
    print(f"\n" + "=" * 70)
    print(f"📊 이상치 영향 분석: {x_col} vs {y_col}")
    print(f"=" * 70)
    print(f"전체 샘플: {len(x)}개")
    print(f"이상치: {outlier_mask.sum()}개")
    print(f"\nPearson (이상치 포함):  {corr_with_outliers:.3f}")
    print(f"Pearson (이상치 제외):  {corr_without_outliers:.3f}")
    print(f"Spearman (강건):        {spearman_with:.3f}")
    
    # 차이 해석
    diff = abs(corr_with_outliers - corr_without_outliers)
    if diff > 0.2:
        print(f"\n⚠️  이상치가 상관계수를 크게 왜곡합니다 (Δr = {diff:.3f})")
        print(f"  → Spearman 상관계수 또는 이상치 제거 고려")
    else:
        print(f"\n✅ 이상치의 영향이 크지 않습니다 (Δr = {diff:.3f})")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 이상치 포함
    axes[0].scatter(x, y, alpha=0.5, s=30, label='Normal')
    axes[0].scatter(
        x[outlier_mask], y[outlier_mask],
        color='red', s=100, alpha=0.7, label='Outliers', edgecolors='black'
    )
    axes[0].set_xlabel(x_col, fontsize=11)
    axes[0].set_ylabel(y_col, fontsize=11)
    axes[0].set_title(f'With Outliers\nPearson r={corr_with_outliers:.3f}', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 이상치 제외
    axes[1].scatter(x[~outlier_mask], y[~outlier_mask], alpha=0.5, s=30, color='green')
    axes[1].set_xlabel(x_col, fontsize=11)
    axes[1].set_ylabel(y_col, fontsize=11)
    axes[1].set_title(f'Without Outliers\nPearson r={corr_without_outliers:.3f}', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 이상치 영향 분석
analyze_outlier_effect_on_correlation(df, 'x1_base', 'x8_outliers')
```

### 3.10 상관관계 종합 리포트

```python
def generate_correlation_report(df, target_col=None, output_file='correlation_report.txt'):
    """
    종합 상관관계 분석 리포트 생성
    
    Parameters:
    -----------
    df : DataFrame
    target_col : str
        타겟 변수 (있는 경우)
    output_file : str
        저장할 파일명
    """
    report = []
    report.append("=" * 70)
    report.append("📊 CORRELATION ANALYSIS REPORT")
    report.append("=" * 70)
    report.append(f"\n생성일시: {pd.Timestamp.now()}")
    report.append(f"데이터 크기: {df.shape[0]} rows × {df.shape[1]} columns\n")
    
    # 1. 기본 상관계수 행렬
    report.append("\n" + "=" * 70)
    report.append("1. 상관계수 행렬 (Pearson)")
    report.append("=" * 70)
    corr_matrix = df.corr()
    report.append(corr_matrix.to_string())
    
    # 2. 타겟 변수와의 상관관계
    if target_col:
        report.append("\n" + "=" * 70)
        report.append(f"2. '{target_col}'와 상관관계 Top 10")
        report.append("=" * 70)
        target_corr = corr_matrix[target_col].drop(target_col).sort_values(
            key=abs, ascending=False
        ).head(10)
        for feat, corr in target_corr.items():
            report.append(f"{feat-<30} {corr:>8.3f}")
    
    # 3. 높은 상관관계 변수 쌍
    report.append("\n" + "=" * 70)
    report.append("3. 높은 상관관계 변수 쌍 (|r| > 0.7)")
    report.append("=" * 70)
    high_corr = filter_high_correlations(df, threshold=0.7, method='pearson')
    if len(high_corr) > 0:
        report.append(high_corr.to_string(index=False))
    else:
        report.append("없음")
    
    # 4. VIF 분석
    report.append("\n" + "=" * 70)
    report.append("4. VIF 분석 (다중공선성)")
    report.append("=" * 70)
    vif_df = calculate_vif(df, exclude_cols=[target_col] if target_col else None)
    report.append(vif_df.to_string(index=False))
    
    # 5. 권장 사항
    report.append("\n" + "=" * 70)
    report.append("5. 권장 사항")
    report.append("=" * 70)
    
    # VIF 기반 권장
    high_vif = vif_df[vif_df['VIF'] > 10]
    if len(high_vif) > 0:
        report.append("\n⚠️  다중공선성 문제:")
        for feat in high_vif['Feature']:
            report.append(f"  - {feat} 변수 제거 또는 결합 고려")
    else:
        report.append("\n✅ 다중공선성 문제 없음")
    
    # 타겟 상관 기반 권장
    if target_col:
        weak_corr = target_corr[abs(target_corr) < 0.1]
        if len(weak_corr) > 0:
            report.append(f"\n💡 약한 상관관계 변수 ({len(weak_corr)}개):")
            report.append(f"  → Feature selection 시 제거 고려")
    
    # 리포트 저장
    report_text = "\n".join(report)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✅ 리포트 저장: {output_file}")
    
    return report_text

# 종합 리포트 생성
report = generate_correlation_report(df, target_col='x2_strong_pos')
```

---

## 4. 예시

### 4.1 실전 예제: 부동산 가격 예측 Feature Selection

```python
print("=" * 70)
print("📈 비즈니스 시나리오: 부동산 가격 예측 모델")
print("=" * 70)
print("\n목표:")
print("- 50개 features에서 중요 변수 선택")
print("- 다중공선성 제거")
print("- 모델 성능 향상")

print("\n🔄 분석 프로세스:")
print("-" * 70)
print("1단계: 타겟(가격)과 상관관계 분석")
print("   → Top 20개 features 선택 (|r| > 0.3)")
print("\n2단계: 선택된 features 간 상관관계 확인")
print("   → '면적'과 '방개수': r=0.92 (높은 상관)")
print("   → '건물나이'와 '리모델링연도': r=-0.88")
print("\n3단계: VIF 계산")
print("   → '면적': VIF=15 (심각)")
print("   → '방개수': VIF=12 (심각)")
print("\n4단계: 변수 제거 전략")
print("   → '면적' 유지 (타겟 상관 0.75)")
print("   → '방개수' 제거 (타겟 상관 0.68)")
print("   → '리모델링연도' 유지 (타겟 상관 0.42)")
print("   → '건물나이' 제거 (타겟 상관 -0.40)")

print("\n✅ 최종 결과:")
print("-" * 70)
print("선택된 features: 15개")
print("모델 성능:")
print("  - Before: R² = 0.78, Training time = 25s")
print("  - After:  R² = 0.80, Training time = 8s")
print("  → 성능 향상 + 3배 빠른 학습")
```

### 4.2 입출력 예시

```python
# 입력: 원본 데이터
print("\n📥 입력 데이터:")
print(df.head())

# 출력 1: 상관계수 행렬
print("\n📤 출력 1: 상관계수 행렬")
print(pearson_corr.round(3))

# 출력 2: 타겟 상관관계
print("\n📤 출력 2: 타겟 변수 상관관계")
print(top_features)

# 출력 3: VIF 테이블
print("\n📤 출력 3: VIF 분석 결과")
print(vif_results)

# 출력 4: 높은 상관관계 쌍
print("\n📤 출력 4: 높은 상관관계 변수 쌍")
print(high_corr_pairs)
```

---

## 5. 에이전트 매핑

### 5.1 담당 에이전트

| 작업 | Primary Agent | Supporting Agents |
|------|--------------|-------------------|
| 상관계수 계산 및 해석 | `data-scientist` | - |
| 히트맵 시각화 | `data-visualization-specialist` | `data-scientist` |
| VIF 분석 | `data-scientist` | `feature-engineering-specialist` |
| Feature selection | `feature-engineering-specialist` | `data-scientist` |
| 부분 상관관계 분석 | `data-scientist` | - |

### 5.2 관련 스킬

**Scientific Skills**:
- `scipy` (pearsonr, spearmanr, kendalltau)
- `pandas` (corr 메서드)
- `seaborn` (heatmap, pairplot)
- `statsmodels` (VIF, OLS)
- `matplotlib` (시각화)

---

## 6. 필요 라이브러리

### 6.1 필수 라이브러리

```bash
# 데이터 처리
pip install pandas==2.2.0
pip install numpy==1.26.3

# 통계 분석
pip install scipy==1.12.0
pip install statsmodels==0.14.1

# 시각화
pip install matplotlib==3.8.2
pip install seaborn==0.13.1
```

### 6.2 라이브러리 버전 확인

```python
import pandas as pd
import numpy as np
import scipy
import statsmodels
import matplotlib
import seaborn as sns

print("라이브러리 버전:")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"scipy: {scipy.__version__}")
print(f"statsmodels: {statsmodels.__version__}")
print(f"matplotlib: {matplotlib.__version__}")
print(f"seaborn: {sns.__version__}")
```

---

## 7. 체크포인트

### 7.1 분석 전 체크리스트

- [ ] **데이터 전처리**
  - [ ] 결측값 처리 완료
  - [ ] 이상치 확인 완료
  - [ ] 변수가 수치형인지 확인

- [ ] **분석 목적 명확화**
  - [ ] Feature selection용인가?
  - [ ] 다중공선성 진단용인가?
  - [ ] 탐색적 분석용인가?

### 7.2 분석 중 체크리스트

- [ ] **상관계수 선택**
  - [ ] 선형 관계: Pearson
  - [ ] 단조 관계: Spearman
  - [ ] 이상치 많음: Spearman 또는 Kendall

- [ ] **해석**
  - [ ] 상관계수 크기 확인 (|r| 값)
  - [ ] P-value 확인 (통계적 유의성)
  - [ ] 실질적 의미(practical significance) 고려

### 7.3 분석 후 체크리스트

- [ ] **다중공선성**
  - [ ] VIF > 10인 변수 처리
  - [ ] |r| > 0.8인 변수 쌍 처리

- [ ] **Feature Selection**
  - [ ] 타겟과 약한 상관 변수 제거
  - [ ] 중복 변수 제거
  - [ ] 최종 변수 목록 문서화

---

## 8. 트러블슈팅

### 8.1 일반적 오류

**문제 1: `ValueError: Input contains NaN`**

```python
# 원인: 결측값 존재
# 해결:
df_clean = df.dropna()  # 또는
df_clean = df.fillna(df.mean())
```

**문제 2: 상관계수가 NaN**

```python
# 원인: 변수의 표준편차가 0 (상수)
# 해결: 상수 변수 제거
df_clean = df.loc[:, df.std() > 0]
```

**문제 3: VIF 계산 시 무한대(`inf`) 발생**

```python
# 원인: 완벽한 다중공선성 (r=1.0)
# 해결: 중복 변수 하나 제거
high_corr_pairs = filter_high_correlations(df, threshold=0.99)
# 하나씩 제거 후 재계산
```

### 8.2 해석 관련

**Q1: 상관관계가 있으면 인과관계가 있나요?**

```
A: 아닙니다.
- 상관관계: X와 Y가 함께 변한다
- 인과관계: X가 Y를 야기한다

인과관계 입증 방법:
1. 시간 순서 (X가 Y보다 먼저)
2. 메커니즘 설명 가능
3. 제3변수 통제 (실험 또는 통계적 통제)
4. 반증 가능성 배제
```

**Q2: P-value는 무엇을 의미하나요?**

```
A: 귀무가설(상관계수=0) 하에서 관측된 결과가 나올 확률

해석:
- p < 0.05: 통계적으로 유의미한 상관관계
- p ≥ 0.05: 우연에 의한 상관관계일 가능성

주의:
- p-value는 효과 크기(|r|)와 다름
- 샘플이 크면 작은 r도 유의미할 수 있음
→ 실질적 의미(practical significance) 함께 고려
```

**Q3: Pearson vs Spearman 언제 사용하나요?**

```
A: 데이터 특성에 따라 선택

Pearson:
✅ 선형 관계
✅ 정규분포 (검정 시)
✅ 이상치 없음

Spearman:
✅ 비선형 단조 관계
✅ 순서형 데이터
✅ 이상치 많음
✅ 정규성 가정 불필요

실무 전략:
1. 산점도로 관계 확인
2. 선형이면 Pearson
3. 곡선이지만 단조면 Spearman
4. 둘 다 계산하여 비교
```

### 8.3 다중공선성 해결 전략

```python
# 전략 1: 변수 제거
# VIF > 10인 변수 중 타겟 상관 낮은 것 제거

# 전략 2: PCA로 통합
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
X_combined = pca.fit_transform(df[['var1', 'var2']])

# 전략 3: Regularization
from sklearn.linear_model import Ridge, Lasso
# Ridge/Lasso는 다중공선성에 강건

# 전략 4: Domain knowledge
# 비즈니스 관점에서 더 중요한 변수 유지
```

---

## 9. 참고 자료

### 9.1 공식 문서

- **Pandas Correlation**: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html
- **SciPy Stats**: https://docs.scipy.org/doc/scipy/reference/stats.html
- **Statsmodels VIF**: https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html
- **Seaborn Heatmap**: https://seaborn.pydata.org/generated/seaborn.heatmap.html

### 9.2 베스트 프랙티스

1. **상관관계 분석 파이프라인**
   ```
   1. 산점도로 관계 시각화
   2. 적절한 상관계수 선택 (Pearson/Spearman)
   3. P-value로 통계적 유의성 확인
   4. 효과 크기(|r|)로 실질적 의미 판단
   5. 인과관계와 혼동하지 않기
   ```

2. **Feature Selection 전략**
   ```
   1. 타겟과 상관관계 계산
   2. 약한 상관 변수 제거 (|r| < 0.1)
   3. Features 간 상관관계 확인
   4. 높은 상관 쌍 중 하나 제거
   5. VIF로 다중공선성 최종 확인
   ```

3. **다중공선성 관리**
   ```
   - VIF < 5: 문제 없음
   - 5 < VIF < 10: 주의, 모니터링
   - VIF > 10: 즉시 조치 필요
   ```

### 9.3 추가 학습 자료

- **상관관계 직관적 이해**: https://rpsychologist.com/correlation/
- **Correlation vs Causation**: https://www.tylervigen.com/spurious-correlations
- **Partial Correlation**: https://en.wikipedia.org/wiki/Partial_correlation
- **VIF 설명**: https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/

---

## 10. 요약

### 10.1 핵심 메시지

상관관계 분석은 변수 간 선형/비선형 연관성을 정량화하는 필수 기법입니다. Pearson, Spearman, Kendall 등 다양한 상관계수를 데이터 특성에 맞게 선택하고, VIF로 다중공선성을 진단하여 효과적인 feature selection과 모델링 전략을 수립할 수 있습니다.

**주의**: 상관관계는 인과관계가 아닙니다!

### 10.2 실무 적용 순서

1. **히트맵**: 전체 상관관계 파악 (5분)
2. **타겟 상관**: 중요 features 식별 (5분)
3. **산점도**: 관계 시각적 확인 (10분)
4. **VIF**: 다중공선성 진단 (5분)
5. **Feature Selection**: 최종 변수 선택 (10분)

**총 소요 시간**: 약 35분

### 10.3 다음 단계

- **Feature 중요도 분석**: `09-feature-importance.md` 참고
- **통계 검정**: `11-hypothesis-testing.md` 참고
- **회귀 분석**: statsmodels OLS 활용
- **차원 축소**: `06-multivariate-analysis.md` 참고

---

**작성일**: 2025-01-25  
**버전**: 1.0  
**난이도**: ⭐⭐ (중급)  
**예상 소요 시간**: 2-3시간 (학습 및 실습)
