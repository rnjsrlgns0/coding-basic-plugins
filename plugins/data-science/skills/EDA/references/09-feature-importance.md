# 09. Feature Importance (특성 중요도)

## 1. 개요

### 1.1 목적
Feature Importance는 예측 모델에서 각 변수(feature)가 타겟 예측에 얼마나 기여하는지를 정량화하는 분석 기법입니다. 불필요한 변수를 제거하고, 중요 변수에 집중하여 모델 성능과 해석 가능성을 향상시킵니다.

### 1.2 적용 시기
- 고차원 데이터에서 중요 변수를 선택하고 싶을 때
- 모델 성능에 기여하지 않는 변수를 제거하여 과적합 방지
- 비즈니스 인사이트 도출 (어떤 요인이 결과에 영향을 미치는가?)
- Feature engineering 우선순위 결정
- 모델 해석 및 설명 (Explainable AI)

### 1.3 주요 기법
- **Tree-based Importance**: Random Forest, XGBoost의 불순도/gain 기반
- **Permutation Importance**: 변수 셔플 후 성능 저하 측정
- **SHAP Values**: 게임 이론 기반 기여도 분석
- **Coefficient-based**: 선형 모델의 계수 크기
- **Recursive Feature Elimination (RFE)**: 순차적 제거

---

## 2. 이론적 배경

### 2.1 특성 중요도의 개념

**핵심 질문**: "이 변수가 없다면 예측 성능이 얼마나 떨어질까?"

```
예시: 주택 가격 예측
Feature          Importance    해석
면적             0.45         → 가장 중요 (45% 기여)
위치(지역)       0.25         → 두 번째로 중요
건물 연식        0.15         → 세 번째
방 개수          0.08         → 중간
주차 여부        0.05         → 낮음
페인트 색상      0.02         → 거의 무관
```

### 2.2 특성 중요도 방법론 비교

#### 1. Impurity-based Importance (불순도 기반)
- **원리**: 의사결정트리에서 각 변수가 분할할 때 감소시킨 불순도 평균
- **장점**: 
  - 빠른 계산
  - 학습과 동시에 계산
  - scikit-learn에서 기본 제공
- **단점**: 
  - 고cardinality 변수에 편향
  - 변수 간 상관관계 있으면 불안정
  - Train 데이터에만 의존 (과적합 가능)

#### 2. Permutation Importance (순열 중요도)
- **원리**: 변수를 무작위로 섞은 후 성능 저하 측정
- **장점**: 
  - 모든 모델에 적용 가능
  - Test 데이터로 계산 가능 (일반화)
  - 변수 간 상관관계 영향 적음
- **단점**: 
  - 계산 비용 높음 (N번 반복)
  - 샘플링 의존성

#### 3. SHAP (SHapley Additive exPlanations)
- **원리**: 게임 이론의 Shapley value로 각 변수의 기여도 계산
- **장점**: 
  - 이론적으로 가장 정확
  - 개별 예측 설명 가능
  - 양방향 효과(긍정/부정) 구분
- **단점**: 
  - 계산 비용 매우 높음
  - 복잡한 개념

### 2.3 시나리오

**시나리오 1: 고차원 데이터 축소**
```
상황: 500개 features, 10,000개 샘플
문제: 모델 학습 시간 30분, 과적합

분석:
1. Random Forest로 feature importance 계산
2. Importance > 0.01인 변수만 선택 (50개로 축소)
3. Permutation importance로 검증
4. SHAP로 상위 20개 변수 심층 분석

결과:
- 최종 50개 features 선택
- 학습 시간: 30분 → 3분 (10배 개선)
- 모델 성능: 유지 또는 소폭 향상
```

**시나리오 2: 비즈니스 인사이트 도출**
```
상황: 고객 이탈 예측 모델
목표: 이탈 방지 전략 수립

분석:
1. Feature importance로 이탈 주요 요인 식별
   → 상위 5개: 고객서비스 만족도, 가격, 사용빈도, 경쟁사 프로모션, 계약기간
2. SHAP로 각 요인의 영향 방향 확인
   → 만족도 ↓ → 이탈 ↑
   → 가격 ↑ → 이탈 ↑

액션:
- 고객서비스 품질 개선 (1순위)
- 가격 경쟁력 확보 (2순위)
- 로열티 프로그램 강화 (3순위)
```

**시나리오 3: 모델 디버깅**
```
상황: 모델 성능이 기대보다 낮음
진단:
1. Feature importance 확인
   → 상위 변수들이 예상과 다름
   → '고객ID' 변수가 1위 (data leakage!)
2. Permutation importance로 재확인
   → Train: 높음, Test: 0 (과적합 확인)

해결:
- 'ID', '날짜' 등 leakage 변수 제거
- 재학습 후 정상 성능 확보
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
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 모델
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

# Feature importance 도구
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE, SelectFromModel
import shap

# 시각화
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
%matplotlib inline

# 한글 폰트 (선택)
plt.rcParams['font.family'] = 'AppleGothic'  # Mac
plt.rcParams['axes.unicode_minus'] = False
```

### 3.2 샘플 데이터 생성

```python
def generate_sample_data(n_samples=1000, n_features=20, task='classification'):
    """
    Feature importance 분석용 샘플 데이터 생성
    
    Parameters:
    -----------
    n_samples : int
    n_features : int
    task : str
        'classification' or 'regression'
    
    Returns:
    --------
    X_train, X_test, y_train, y_test, feature_names
    """
    np.random.seed(42)
    
    if task == 'classification':
        # 분류 데이터
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=10,  # 실제 유용한 변수 10개
            n_redundant=5,     # 중복 변수 5개
            n_repeated=0,
            n_classes=2,
            random_state=42
        )
    else:
        # 회귀 데이터
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=10,
            n_targets=1,
            noise=10.0,
            random_state=42
        )
    
    # Feature 이름 생성
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    # DataFrame 변환
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    # Train/Test 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.3, random_state=42
    )
    
    print(f"=" * 70)
    print(f"📊 데이터 생성 완료 ({task.upper()})")
    print(f"=" * 70)
    print(f"Train: {X_train.shape[0]} samples × {X_train.shape[1]} features")
    print(f"Test:  {X_test.shape[0]} samples × {X_test.shape[1]} features")
    print(f"\nFeature 구성:")
    print(f"  - Informative: 10개 (실제 유용)")
    print(f"  - Redundant:   5개 (중복)")
    print(f"  - Random:      5개 (노이즈)")
    
    return X_train, X_test, y_train, y_test, feature_names

# 분류 데이터 생성
X_train, X_test, y_train, y_test, feature_names = generate_sample_data(
    n_samples=1000,
    n_features=20,
    task='classification'
)
```

### 3.3 Random Forest Feature Importance (불순도 기반)

```python
def calculate_tree_importance(X_train, y_train, feature_names, task='classification', top_n=10):
    """
    Tree-based 모델의 feature importance 계산
    
    Parameters:
    -----------
    X_train : DataFrame or array
    y_train : Series or array
    feature_names : list
    task : str
        'classification' or 'regression'
    top_n : int
        상위 몇 개 표시
    
    Returns:
    --------
    importance_df : DataFrame
    model : trained model
    """
    print(f"\n" + "=" * 70)
    print(f"🌲 Random Forest Feature Importance (불순도 기반)")
    print(f"=" * 70)
    
    # 모델 학습
    if task == 'classification':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    
    model.fit(X_train, y_train)
    
    # Feature importance 추출
    importances = model.feature_importances_
    
    # DataFrame 생성
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    # 정규화 (합=1)
    importance_df['importance_pct'] = (
        importance_df['importance'] / importance_df['importance'].sum() * 100
    )
    
    # 누적 중요도
    importance_df['cumulative_pct'] = importance_df['importance_pct'].cumsum()
    
    # 출력
    print(f"\nTop {top_n} Important Features:")
    print(f"-" * 70)
    print(f"{'Rank':<6} {'Feature':<20} {'Importance':<12} {'%':<8} {'Cumul %'}")
    print(f"-" * 70)
    
    for idx, row in importance_df.head(top_n).iterrows():
        print(f"{idx+1:<6} {row['feature']:<20} {row['importance']:<12.4f} "
              f"{row['importance_pct']:<8.2f} {row['cumulative_pct']:.2f}%")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 막대 그래프
    top_features = importance_df.head(top_n)
    axes[0].barh(range(len(top_features)), top_features['importance'], alpha=0.8)
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features['feature'])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Importance', fontsize=11)
    axes[0].set_title(f'Top {top_n} Feature Importance', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # 누적 중요도
    axes[1].plot(
        range(1, len(importance_df)+1),
        importance_df['cumulative_pct'],
        marker='o',
        linewidth=2,
        markersize=6
    )
    axes[1].axhline(y=80, color='r', linestyle='--', label='80% threshold')
    axes[1].axhline(y=95, color='orange', linestyle='--', label='95% threshold')
    axes[1].set_xlabel('Number of Features', fontsize=11)
    axes[1].set_ylabel('Cumulative Importance (%)', fontsize=11)
    axes[1].set_title('Cumulative Feature Importance', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 80% 커버하는 변수 개수
    n_80 = (importance_df['cumulative_pct'] >= 80).idxmax() + 1
    print(f"\n💡 Insight:")
    print(f"  - 상위 {n_80}개 변수로 80% 중요도 커버")
    print(f"  - 나머지 {len(feature_names) - n_80}개 변수는 제거 고려")
    
    return importance_df, model

# Random Forest Importance 계산
rf_importance, rf_model = calculate_tree_importance(
    X_train, y_train, feature_names, task='classification', top_n=15
)
```

### 3.4 Permutation Importance (순열 중요도)

```python
def calculate_permutation_importance(model, X, y, feature_names, n_repeats=10, top_n=10):
    """
    Permutation Importance 계산
    
    Parameters:
    -----------
    model : trained model
    X : DataFrame or array (보통 test set 사용)
    y : Series or array
    feature_names : list
    n_repeats : int
        셔플 반복 횟수
    top_n : int
    
    Returns:
    --------
    perm_importance_df : DataFrame
    """
    print(f"\n" + "=" * 70)
    print(f"🔄 Permutation Importance (순열 중요도)")
    print(f"=" * 70)
    print(f"계산 중... (n_repeats={n_repeats})")
    
    # Permutation importance 계산
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1
    )
    
    # DataFrame 생성
    perm_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False).reset_index(drop=True)
    
    # 출력
    print(f"\nTop {top_n} Important Features:")
    print(f"-" * 70)
    print(f"{'Rank':<6} {'Feature':<20} {'Mean':<12} {'Std':<12}")
    print(f"-" * 70)
    
    for idx, row in perm_importance_df.head(top_n).iterrows():
        print(f"{idx+1:<6} {row['feature']:<20} {row['importance_mean']:<12.4f} "
              f"{row['importance_std']:<12.4f}")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 에러바 포함 막대 그래프
    top_features = perm_importance_df.head(top_n)
    axes[0].barh(
        range(len(top_features)),
        top_features['importance_mean'],
        xerr=top_features['importance_std'],
        alpha=0.8,
        capsize=5
    )
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features['feature'])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Permutation Importance', fontsize=11)
    axes[0].set_title(f'Top {top_n} Permutation Importance', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # 박스플롯 (상위 5개)
    top_5 = perm_importance_df.head(5)['feature'].tolist()
    result_subset = permutation_importance(
        model, X, y,
        n_repeats=30,  # 더 많은 반복으로 분포 확인
        random_state=42
    )
    
    data_for_box = []
    for i, feat in enumerate(top_5):
        feat_idx = feature_names.index(feat)
        data_for_box.append(result_subset.importances[feat_idx])
    
    axes[1].boxplot(data_for_box, labels=top_5)
    axes[1].set_ylabel('Permutation Importance', fontsize=11)
    axes[1].set_title('Top 5 Features Distribution (n=30)', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n💡 Insight:")
    print(f"  - Std가 큰 변수: 불안정 (데이터 의존적)")
    print(f"  - Std가 작은 변수: 안정적 중요도")
    print(f"  - Mean ≈ 0: 예측에 기여 없음 (제거 고려)")
    
    return perm_importance_df

# Permutation Importance 계산 (Test set 사용)
perm_importance = calculate_permutation_importance(
    rf_model, X_test, y_test, feature_names, n_repeats=10, top_n=15
)
```

### 3.5 Tree Importance vs Permutation Importance 비교

```python
def compare_importance_methods(rf_importance, perm_importance, top_n=15):
    """
    두 방법의 feature importance 비교
    """
    print(f"\n" + "=" * 70)
    print(f"⚖️  Feature Importance 방법 비교")
    print(f"=" * 70)
    
    # 데이터 병합
    comparison = rf_importance[['feature', 'importance']].copy()
    comparison = comparison.rename(columns={'importance': 'RF_importance'})
    comparison = comparison.merge(
        perm_importance[['feature', 'importance_mean']],
        on='feature'
    ).rename(columns={'importance_mean': 'Perm_importance'})
    
    # 정규화 (0-1)
    comparison['RF_norm'] = (
        comparison['RF_importance'] / comparison['RF_importance'].max()
    )
    comparison['Perm_norm'] = (
        comparison['Perm_importance'] / comparison['Perm_importance'].max()
    )
    
    # 차이 계산
    comparison['difference'] = abs(
        comparison['RF_norm'] - comparison['Perm_norm']
    )
    
    # 정렬 (RF 기준)
    comparison = comparison.sort_values('RF_importance', ascending=False)
    
    # 출력
    print(f"\nTop {top_n} Features 비교:")
    print(f"-" * 70)
    print(f"{'Feature':<20} {'RF':<10} {'Perm':<10} {'Diff':<10} {'Status'}")
    print(f"-" * 70)
    
    for idx, row in comparison.head(top_n).iterrows():
        diff = row['difference']
        if diff < 0.2:
            status = "✅ 일치"
        elif diff < 0.5:
            status = "⚠️  차이"
        else:
            status = "🚨 불일치"
        
        print(f"{row['feature']:<20} {row['RF_norm']:<10.3f} "
              f"{row['Perm_norm']:<10.3f} {diff:<10.3f} {status}")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 산점도
    axes[0].scatter(
        comparison['RF_norm'],
        comparison['Perm_norm'],
        s=100,
        alpha=0.6
    )
    
    # 대각선 (완벽한 일치)
    max_val = max(comparison['RF_norm'].max(), comparison['Perm_norm'].max())
    axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Agreement')
    
    # 레이블 (상위 5개)
    top_5 = comparison.head(5)
    for idx, row in top_5.iterrows():
        axes[0].annotate(
            row['feature'],
            (row['RF_norm'], row['Perm_norm']),
            fontsize=9,
            alpha=0.7
        )
    
    axes[0].set_xlabel('RF Importance (normalized)', fontsize=11)
    axes[0].set_ylabel('Permutation Importance (normalized)', fontsize=11)
    axes[0].set_title('RF vs Permutation Importance', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 막대 그래프 (상위 10개)
    top_10 = comparison.head(10)
    x = np.arange(len(top_10))
    width = 0.35
    
    axes[1].barh(x - width/2, top_10['RF_norm'], width, label='RF', alpha=0.8)
    axes[1].barh(x + width/2, top_10['Perm_norm'], width, label='Perm', alpha=0.8)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(top_10['feature'])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Normalized Importance', fontsize=11)
    axes[1].set_title('Top 10 Features: RF vs Permutation', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    # 불일치 변수 경고
    high_diff = comparison[comparison['difference'] > 0.5]
    if len(high_diff) > 0:
        print(f"\n⚠️  큰 차이를 보이는 변수 ({len(high_diff)}개):")
        for idx, row in high_diff.iterrows():
            print(f"  - {row['feature']}: RF={row['RF_norm']:.3f}, Perm={row['Perm_norm']:.3f}")
        print(f"\n💡 원인:")
        print(f"  - 변수 간 상관관계 (RF는 중복 변수에 importance 분산)")
        print(f"  - Train/Test 분포 차이")
        print(f"  - 과적합 (RF는 높지만 Perm은 낮음)")
    
    return comparison

# 비교 실행
comparison_df = compare_importance_methods(rf_importance, perm_importance, top_n=15)
```

### 3.6 SHAP Values (게임 이론 기반)

```python
def calculate_shap_importance(model, X_train, X_test, feature_names, max_display=15):
    """
    SHAP (SHapley Additive exPlanations) 값 계산
    
    Parameters:
    -----------
    model : trained model
    X_train : training data (for TreeExplainer background)
    X_test : test data (for explanation)
    feature_names : list
    max_display : int
    
    Returns:
    --------
    shap_values : array
    explainer : SHAP explainer
    """
    print(f"\n" + "=" * 70)
    print(f"🎮 SHAP Values (게임 이론 기반)")
    print(f"=" * 70)
    print(f"계산 중... (시간 소요 가능)")
    
    # SHAP Explainer 생성
    # Tree 모델용 (빠름)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # 분류 모델의 경우 클래스별 shap_values (클래스 1만 사용)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    print(f"✅ SHAP 계산 완료")
    
    # Summary Plot (전체 변수)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )
    plt.title('SHAP Summary Plot (Feature Importance + Direction)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()
    
    # Bar Plot (평균 절댓값)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        plot_type='bar',
        max_display=max_display,
        show=False
    )
    plt.title('SHAP Feature Importance (Mean |SHAP|)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()
    
    # Feature importance (mean absolute SHAP)
    shap_importance = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': shap_importance
    }).sort_values('shap_importance', ascending=False).reset_index(drop=True)
    
    print(f"\nTop {max_display} SHAP Important Features:")
    print(f"-" * 70)
    print(f"{'Rank':<6} {'Feature':<20} {'Mean |SHAP|':<15}")
    print(f"-" * 70)
    
    for idx, row in shap_df.head(max_display).iterrows():
        print(f"{idx+1:<6} {row['feature']:<20} {row['shap_importance']:<15.4f}")
    
    print(f"\n💡 SHAP 해석:")
    print(f"  - 점의 색상: Feature 값 (빨강=높음, 파랑=낮음)")
    print(f"  - X축 위치: SHAP 값 (양수=예측↑, 음수=예측↓)")
    print(f"  - 예시: 빨간 점이 오른쪽 → 값이 높으면 예측↑")
    print(f"  - 예시: 파란 점이 왼쪽 → 값이 낮으면 예측↓")
    
    return shap_values, explainer, shap_df

# SHAP 계산 (샘플 크기 제한으로 속도 향상)
X_test_sample = X_test.sample(min(300, len(X_test)), random_state=42)
shap_values, explainer, shap_df = calculate_shap_importance(
    rf_model,
    X_train,
    X_test_sample,
    feature_names,
    max_display=15
)
```

### 3.7 SHAP 개별 예측 설명

```python
def explain_single_prediction(explainer, X_test, feature_names, sample_idx=0):
    """
    SHAP을 사용한 개별 예측 설명
    
    Parameters:
    -----------
    explainer : SHAP explainer
    X_test : test data
    feature_names : list
    sample_idx : int
        설명할 샘플 인덱스
    """
    print(f"\n" + "=" * 70)
    print(f"🔍 개별 예측 설명 (Sample #{sample_idx})")
    print(f"=" * 70)
    
    # 샘플 선택
    X_sample = X_test.iloc[[sample_idx]]
    
    # SHAP 값 계산
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Waterfall Plot (개별 예측의 기여도)
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=X_sample.values[0],
            feature_names=feature_names
        ),
        max_display=15,
        show=False
    )
    plt.title(f'SHAP Waterfall Plot (Sample #{sample_idx})', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()
    
    # Force Plot (단일 샘플)
    plt.figure(figsize=(16, 3))
    shap.force_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        shap_values[0],
        X_sample.values[0],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.title(f'SHAP Force Plot (Sample #{sample_idx})', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()
    
    # Feature 값 및 기여도 출력
    contributions = pd.DataFrame({
        'feature': feature_names,
        'value': X_sample.values[0],
        'shap_value': shap_values[0]
    }).sort_values('shap_value', key=abs, ascending=False)
    
    print(f"\n샘플 #{sample_idx}의 Feature 기여도:")
    print(f"-" * 70)
    print(f"{'Feature':<20} {'Value':<15} {'SHAP Value':<15} {'Effect'}")
    print(f"-" * 70)
    
    for idx, row in contributions.head(10).iterrows():
        effect = "예측↑" if row['shap_value'] > 0 else "예측↓"
        print(f"{row['feature']:<20} {row['value']:<15.3f} "
              f"{row['shap_value']:<15.4f} {effect}")
    
    print(f"\n💡 해석:")
    print(f"  - Base value: 모든 샘플의 평균 예측값")
    print(f"  - SHAP value > 0: 해당 feature가 예측을 증가시킴")
    print(f"  - SHAP value < 0: 해당 feature가 예측을 감소시킴")
    print(f"  - 최종 예측 = Base + Σ(SHAP values)")

# 개별 예측 설명 (3개 샘플)
for i in [0, 10, 50]:
    explain_single_prediction(explainer, X_test_sample, feature_names, sample_idx=i)
```

### 3.8 Recursive Feature Elimination (RFE)

```python
def perform_rfe(X_train, y_train, feature_names, n_features_to_select=10, task='classification'):
    """
    RFE (Recursive Feature Elimination)로 최적 변수 선택
    
    Parameters:
    -----------
    X_train, y_train : training data
    feature_names : list
    n_features_to_select : int
        선택할 변수 개수
    task : str
    
    Returns:
    --------
    selected_features : list
    rfe : RFE object
    """
    print(f"\n" + "=" * 70)
    print(f"🔄 Recursive Feature Elimination (RFE)")
    print(f"=" * 70)
    print(f"목표: {n_features_to_select}개 변수 선택")
    
    # 모델 선택
    if task == 'classification':
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    else:
        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
    
    # RFE 실행
    rfe = RFE(
        estimator=estimator,
        n_features_to_select=n_features_to_select,
        step=1  # 한 번에 제거할 변수 개수
    )
    
    print(f"RFE 실행 중...")
    rfe.fit(X_train, y_train)
    print(f"✅ 완료")
    
    # 선택된 변수
    selected_mask = rfe.support_
    selected_features = [f for f, selected in zip(feature_names, selected_mask) if selected]
    
    # 순위
    ranking_df = pd.DataFrame({
        'feature': feature_names,
        'ranking': rfe.ranking_,
        'selected': selected_mask
    }).sort_values('ranking')
    
    print(f"\n선택된 변수 ({len(selected_features)}개):")
    print(f"-" * 70)
    for feat in selected_features:
        print(f"  ✅ {feat}")
    
    print(f"\n제거된 변수 ({len(feature_names) - len(selected_features)}개):")
    removed = ranking_df[ranking_df['selected'] == False].head(10)
    for idx, row in removed.iterrows():
        print(f"  ❌ {row['feature']} (rank: {row['ranking']})")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 순위 막대 그래프
    colors = ['green' if sel else 'red' for sel in ranking_df['selected']]
    axes[0].barh(range(len(ranking_df)), ranking_df['ranking'], color=colors, alpha=0.7)
    axes[0].set_yticks(range(len(ranking_df)))
    axes[0].set_yticklabels(ranking_df['feature'])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Ranking (1=Best)', fontsize=11)
    axes[0].set_title('RFE Feature Ranking', fontsize=12)
    axes[0].axvline(x=n_features_to_select, color='blue', linestyle='--', linewidth=2, label='Cutoff')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # 선택/제거 파이 차트
    selected_count = selected_mask.sum()
    removed_count = len(feature_names) - selected_count
    axes[1].pie(
        [selected_count, removed_count],
        labels=['Selected', 'Removed'],
        autopct='%1.1f%%',
        colors=['green', 'red'],
        startangle=90
    )
    axes[1].set_title(f'Feature Selection Result\n({selected_count} / {len(feature_names)})', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return selected_features, rfe, ranking_df

# RFE 실행
selected_features, rfe_model, rfe_ranking = perform_rfe(
    X_train, y_train, feature_names, n_features_to_select=10, task='classification'
)
```

### 3.9 모든 방법 종합 비교

```python
def compare_all_methods(rf_importance, perm_importance, shap_df, rfe_ranking, top_n=15):
    """
    모든 feature importance 방법 종합 비교
    """
    print(f"\n" + "=" * 70)
    print(f"🏆 Feature Importance 종합 비교")
    print(f"=" * 70)
    
    # 데이터 병합
    comparison = rf_importance[['feature', 'importance']].copy()
    comparison = comparison.rename(columns={'importance': 'RF'})
    
    comparison = comparison.merge(
        perm_importance[['feature', 'importance_mean']],
        on='feature'
    ).rename(columns={'importance_mean': 'Permutation'})
    
    comparison = comparison.merge(
        shap_df[['feature', 'shap_importance']],
        on='feature'
    ).rename(columns={'shap_importance': 'SHAP'})
    
    comparison = comparison.merge(
        rfe_ranking[['feature', 'ranking', 'selected']],
        on='feature'
    )
    
    # 정규화 (0-1)
    for col in ['RF', 'Permutation', 'SHAP']:
        comparison[f'{col}_norm'] = comparison[col] / comparison[col].max()
    
    # 평균 순위 계산
    comparison['avg_importance'] = (
        comparison['RF_norm'] + comparison['Permutation_norm'] + comparison['SHAP_norm']
    ) / 3
    
    comparison = comparison.sort_values('avg_importance', ascending=False)
    
    # 출력
    print(f"\nTop {top_n} Features (종합):")
    print(f"-" * 90)
    print(f"{'Rank':<6} {'Feature':<18} {'RF':<8} {'Perm':<8} {'SHAP':<8} {'Avg':<8} {'RFE'}")
    print(f"-" * 90)
    
    for idx, row in comparison.head(top_n).iterrows():
        rfe_status = "✅" if row['selected'] else "❌"
        print(f"{idx+1:<6} {row['feature']:<18} {row['RF_norm']:<8.3f} "
              f"{row['Permutation_norm']:<8.3f} {row['SHAP_norm']:<8.3f} "
              f"{row['avg_importance']:<8.3f} {rfe_status}")
    
    # 시각화: 히트맵
    plt.figure(figsize=(12, 10))
    
    top_features = comparison.head(top_n)
    heatmap_data = top_features[['RF_norm', 'Permutation_norm', 'SHAP_norm']].T
    heatmap_data.columns = top_features['feature']
    
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        linewidths=1,
        cbar_kws={'label': 'Normalized Importance'}
    )
    
    plt.ylabel('Method', fontsize=12)
    plt.xlabel('Feature', fontsize=12)
    plt.title(f'Feature Importance Heatmap (Top {top_n})', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # 일치도 분석
    print(f"\n📊 방법 간 일치도:")
    print(f"-" * 70)
    
    # 상위 10개 변수가 겹치는 정도
    top_10_rf = set(rf_importance.head(10)['feature'])
    top_10_perm = set(perm_importance.head(10)['feature'])
    top_10_shap = set(shap_df.head(10)['feature'])
    top_10_rfe = set(rfe_ranking[rfe_ranking['selected']].head(10)['feature'])
    
    all_methods = top_10_rf & top_10_perm & top_10_shap & top_10_rfe
    print(f"모든 방법이 일치하는 변수 ({len(all_methods)}개):")
    for feat in all_methods:
        print(f"  ✅ {feat}")
    
    return comparison

# 종합 비교
final_comparison = compare_all_methods(
    rf_importance, perm_importance, shap_df, rfe_ranking, top_n=15
)
```

### 3.10 최종 Feature Selection 권장

```python
def recommend_final_features(comparison, min_methods=2, top_n=15):
    """
    최종 feature selection 권장
    
    Parameters:
    -----------
    comparison : DataFrame
        종합 비교 결과
    min_methods : int
        최소 몇 개 방법에서 상위권이어야 하는지
    top_n : int
    """
    print(f"\n" + "=" * 70)
    print(f"✅ 최종 Feature Selection 권장")
    print(f"=" * 70)
    
    # 각 방법에서 상위 top_n에 포함되는지 확인
    comparison['RF_top'] = comparison['RF_norm'].rank(ascending=False) <= top_n
    comparison['Perm_top'] = comparison['Permutation_norm'].rank(ascending=False) <= top_n
    comparison['SHAP_top'] = comparison['SHAP_norm'].rank(ascending=False) <= top_n
    
    # 몇 개 방법에서 상위권인지
    comparison['n_methods_top'] = (
        comparison['RF_top'].astype(int) +
        comparison['Perm_top'].astype(int) +
        comparison['SHAP_top'].astype(int)
    )
    
    # 권장 변수
    recommended = comparison[comparison['n_methods_top'] >= min_methods].sort_values(
        'avg_importance', ascending=False
    )
    
    print(f"\n권장 기준: 최소 {min_methods}개 방법에서 Top {top_n}")
    print(f"권장 변수: {len(recommended)}개")
    print(f"-" * 70)
    print(f"{'Feature':<20} {'Avg':<8} {'Methods':<10} {'RFE'}")
    print(f"-" * 70)
    
    for idx, row in recommended.iterrows():
        rfe_status = "✅" if row['selected'] else "❌"
        methods_str = f"{row['n_methods_top']}/3"
        print(f"{row['feature']:<20} {row['avg_importance']:<8.3f} "
              f"{methods_str:<10} {rfe_status}")
    
    # 제거 권장 변수
    not_recommended = comparison[comparison['n_methods_top'] < min_methods].sort_values(
        'avg_importance', ascending=False
    ).head(10)
    
    print(f"\n제거 고려 변수 (Top 10):")
    print(f"-" * 70)
    for idx, row in not_recommended.iterrows():
        print(f"  ❌ {row['feature']} (평균 중요도: {row['avg_importance']:.3f})")
    
    # 최종 권장
    print(f"\n💡 최종 권장사항:")
    print(f"=" * 70)
    print(f"1. 강력 권장 (모든 방법 일치): {(comparison['n_methods_top'] == 3).sum()}개")
    print(f"2. 권장 (2개 이상 방법): {(comparison['n_methods_top'] >= 2).sum()}개")
    print(f"3. 제거 고려 (1개 이하 방법): {(comparison['n_methods_top'] <= 1).sum()}개")
    
    print(f"\n✅ Feature Selection 완료!")
    print(f"   원본: {len(comparison)}개 → 선택: {len(recommended)}개")
    print(f"   축소율: {(1 - len(recommended)/len(comparison))*100:.1f}%")
    
    return recommended['feature'].tolist()

# 최종 권장 변수 선택
final_features = recommend_final_features(final_comparison, min_methods=2, top_n=10)

print(f"\n🎯 최종 선택 변수 목록:")
for i, feat in enumerate(final_features, 1):
    print(f"{i}. {feat}")
```

---

## 4. 예시

### 4.1 실전 예제: 신용 위험 평가 모델

```python
print("=" * 70)
print("📈 비즈니스 시나리오: 신용 위험 평가 모델")
print("=" * 70)

print("\n목표:")
print("- 고객의 신용 위험 예측")
print("- 100개 features에서 핵심 변수 선택")
print("- 규제 기관에 모델 설명 필요 (Explainable AI)")

print("\n🔄 분석 프로세스:")
print("-" * 70)
print("1단계: Random Forest로 빠른 screening")
print("   → 상위 30개 features 선택")
print("\n2단계: Permutation Importance로 검증")
print("   → Test set에서 실제 기여도 확인")
print("   → 5개 features 추가 제거 (과적합 방지)")
print("\n3단계: SHAP으로 해석")
print("   → 각 변수가 위험도에 미치는 방향 확인")
print("   → '연체 횟수↑ → 위험도↑' 등 비즈니스 로직 검증")
print("\n4단계: RFE로 최종 선택")
print("   → 15개 핵심 features 확정")

print("\n✅ 결과:")
print("-" * 70)
print("최종 15개 Features:")
print("  1. 연체 이력 (SHAP: 0.45)")
print("  2. 신용 점수 (SHAP: 0.38)")
print("  3. 부채 비율 (SHAP: 0.32)")
print("  4. 소득 수준 (SHAP: 0.28)")
print("  5. 고용 기간 (SHAP: 0.21)")
print("  ... (이하 10개)")

print("\n비즈니스 효과:")
print("  - 모델 정확도: 88% (100개 features) → 89% (15개 features)")
print("  - 학습 시간: 45분 → 3분 (15배 개선)")
print("  - 규제 기관 승인 획득 (SHAP 설명 제공)")
```

### 4.2 입출력 예시

```python
# 입력: Feature 목록
print("\n📥 입력: 원본 Features")
print(f"총 {len(feature_names)}개 features")
print(feature_names[:10])

# 출력 1: RF Importance
print("\n📤 출력 1: Random Forest Importance")
print(rf_importance.head(10))

# 출력 2: Permutation Importance
print("\n📤 출력 2: Permutation Importance")
print(perm_importance.head(10))

# 출력 3: SHAP Importance
print("\n📤 출력 3: SHAP Importance")
print(shap_df.head(10))

# 출력 4: 최종 선택 Features
print("\n📤 출력 4: 최종 선택 Features")
print(final_features)
```

---

## 5. 에이전트 매핑

### 5.1 담당 에이전트

| 작업 | Primary Agent | Supporting Agents |
|------|--------------|-------------------|
| Tree-based Importance | `feature-engineering-specialist` | `ml-modeling-specialist` |
| Permutation Importance | `feature-engineering-specialist` | `data-scientist` |
| SHAP 분석 | `feature-engineering-specialist` | `ml-modeling-specialist` |
| RFE 실행 | `feature-engineering-specialist` | - |
| Feature selection 전략 | `feature-engineering-specialist` | `data-scientist` |
| 비즈니스 해석 | `data-scientist` | `feature-engineering-specialist` |

### 5.2 관련 스킬

**Scientific Skills**:
- `scikit-learn` (feature_importances_, permutation_importance, RFE)
- `xgboost` (XGBoost feature importance)
- `shap` (SHAP values)
- `matplotlib`, `seaborn` (시각화)
- `pandas`, `numpy` (데이터 처리)

---

## 6. 필요 라이브러리

### 6.1 필수 라이브러리

```bash
# 머신러닝
pip install scikit-learn==1.4.0
pip install xgboost==2.0.3

# SHAP
pip install shap==0.44.1

# 데이터 처리
pip install pandas==2.2.0
pip install numpy==1.26.3

# 시각화
pip install matplotlib==3.8.2
pip install seaborn==0.13.1
```

### 6.2 라이브러리 버전 확인

```python
import sklearn
import xgboost
import shap
import pandas as pd
import numpy as np

print("라이브러리 버전:")
print(f"scikit-learn: {sklearn.__version__}")
print(f"xgboost: {xgboost.__version__}")
print(f"shap: {shap.__version__}")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
```

---

## 7. 체크포인트

### 7.1 분석 전 체크리스트

- [ ] **데이터 준비**
  - [ ] Train/Test split 완료
  - [ ] 결측값 처리
  - [ ] 범주형 변수 인코딩

- [ ] **모델 선택**
  - [ ] Tree 모델 (RF, XGBoost): 빠른 importance
  - [ ] 선형 모델: coefficient 기반

### 7.2 분석 중 체크리스트

- [ ] **여러 방법 사용**
  - [ ] Tree importance (빠른 탐색)
  - [ ] Permutation (검증)
  - [ ] SHAP (해석)

- [ ] **일관성 확인**
  - [ ] 방법 간 상위 변수 일치하는가?
  - [ ] 불일치 시 원인 분석

### 7.3 분석 후 체크리스트

- [ ] **Feature Selection**
  - [ ] 최종 변수 목록 확정
  - [ ] 제거된 변수 문서화
  - [ ] 비즈니스 로직 검증

- [ ] **모델 재학습**
  - [ ] 선택된 변수로 재학습
  - [ ] 성능 비교 (Before/After)

---

## 8. 트러블슈팅

### 8.1 일반적 오류

**문제 1: SHAP 계산이 너무 느림**

```python
# 해결 1: 샘플 크기 줄이기
X_sample = X_test.sample(min(500, len(X_test)), random_state=42)

# 해결 2: TreeExplainer 사용 (Tree 모델)
explainer = shap.TreeExplainer(model)  # 빠름
# explainer = shap.KernelExplainer(model.predict, X_sample)  # 느림

# 해결 3: GPU 사용 (XGBoost + GPU)
model = XGBClassifier(tree_method='gpu_hist')
```

**문제 2: Permutation Importance가 음수**

```python
# 원인: 무작위 셔플 후 우연히 성능이 향상됨
# 해결: n_repeats 증가 (평균으로 안정화)
result = permutation_importance(
    model, X, y,
    n_repeats=30,  # 10 → 30
    random_state=42
)
```

**문제 3: Tree Importance와 Permutation이 크게 다름**

```python
# 원인 1: 변수 간 상관관계
# → Tree는 상관변수에 importance 분산
# → Permutation은 각 변수 독립 평가

# 원인 2: 과적합
# → Tree는 Train에서 높지만, Perm은 Test에서 낮음

# 해결: 두 방법 모두 참고하여 종합 판단
```

### 8.2 해석 관련

**Q1: Feature Importance가 낮다고 무조건 제거해야 하나요?**

```
A: 아닙니다.
- 비즈니스 중요성 고려 (법적 요구사항 등)
- 해석 가능성 (모델 설명에 필요)
- 상호작용 효과 (단독으로는 약하지만 조합 시 강함)

권장:
1. Importance < 0.01 and 비즈니스 중요도 낮음 → 제거
2. Importance < 0.01 but 비즈니스 중요도 높음 → 유지
3. 애매한 경우 → A/B 테스트 (제거 전후 성능 비교)
```

**Q2: SHAP 값이 양수/음수는 무엇을 의미하나요?**

```
A: 예측값에 대한 기여 방향

양수 SHAP:
- 해당 feature가 예측을 증가시킴
- 분류: 클래스 1 확률 증가
- 회귀: 타겟 값 증가

음수 SHAP:
- 해당 feature가 예측을 감소시킴
- 분류: 클래스 0 확률 증가
- 회귀: 타겟 값 감소

예시:
Feature='신용점수', SHAP=+0.3
→ 신용점수가 승인 확률을 30% 포인트 증가시킴
```

**Q3: Feature Importance 높다고 인과관계가 있나요?**

```
A: 아닙니다. (상관관계 ≠ 인과관계)

Feature Importance는:
- 예측에 유용한 정도
- 상관관계 강도

인과관계 입증 필요:
- 실험 (A/B 테스트)
- 시간 순서 (원인이 결과보다 먼저)
- 도메인 지식 (메커니즘 설명)
```

### 8.3 성능 최적화

```python
# 대용량 데이터 처리

# 1. 샘플링
X_sample = X.sample(min(10000, len(X)), random_state=42)

# 2. 병렬 처리
result = permutation_importance(
    model, X, y,
    n_repeats=10,
    n_jobs=-1  # 모든 CPU 사용
)

# 3. Tree 모델 속도 향상
model = RandomForestClassifier(
    n_estimators=50,  # 100 → 50
    max_depth=10,     # 깊이 제한
    n_jobs=-1
)

# 4. SHAP 근사
explainer = shap.TreeExplainer(
    model,
    feature_perturbation='interventional'  # 빠른 근사
)
```

---

## 9. 참고 자료

### 9.1 공식 문서

- **Scikit-learn Feature Selection**: https://scikit-learn.org/stable/modules/feature_selection.html
- **Permutation Importance**: https://scikit-learn.org/stable/modules/permutation_importance.html
- **SHAP**: https://shap.readthedocs.io/en/latest/
- **XGBoost Feature Importance**: https://xgboost.readthedocs.io/en/latest/python/python_api.html

### 9.2 베스트 프랙티스

1. **Feature Importance 파이프라인**
   ```
   1. RF Importance: 빠른 탐색 (5분)
   2. Permutation: Test set에서 검증 (10분)
   3. SHAP: 상위 20개 변수 심층 분석 (20분)
   4. RFE: 최종 변수 개수 결정 (15분)
   5. 비즈니스 검증: 도메인 전문가 확인
   ```

2. **Feature Selection 기준**
   ```
   강력 권장 제거:
   - 모든 방법에서 하위 20%
   - Permutation ≈ 0
   - 비즈니스 중요도 낮음
   
   제거 고려:
   - 2개 이상 방법에서 하위 30%
   - 상관관계 높은 변수 중 하나
   
   유지:
   - 2개 이상 방법에서 상위 30%
   - SHAP로 해석 가능한 패턴
   - 비즈니스 중요도 높음
   ```

3. **해석 및 설명**
   ```
   내부 설명 (팀):
   - RF Importance로 충분
   - Permutation으로 검증
   
   외부 설명 (규제, 고객):
   - SHAP 필수
   - 개별 예측 설명 (Waterfall plot)
   - 비즈니스 용어로 번역
   ```

### 9.3 추가 학습 자료

- **SHAP 직관적 이해**: https://christophm.github.io/interpretable-ml-book/shap.html
- **Feature Importance 비교**: https://explained.ai/rf-importance/
- **Permutation Importance 논문**: https://arxiv.org/abs/1801.01489
- **SHAP 논문**: https://arxiv.org/abs/1705.07874

---

## 10. 요약

### 10.1 핵심 메시지

Feature Importance는 모델 성능 향상과 해석 가능성을 동시에 확보하는 핵심 기법입니다. Tree-based, Permutation, SHAP 등 다양한 방법을 조합하여 robust한 feature selection을 수행하고, 비즈니스 인사이트를 도출할 수 있습니다.

### 10.2 방법 선택 가이드

| 목적 | 추천 방법 | 소요 시간 |
|------|----------|----------|
| 빠른 탐색 | Random Forest | 5분 |
| 일반화 검증 | Permutation (Test) | 10분 |
| 해석 및 설명 | SHAP | 20분 |
| 최적 변수 개수 | RFE | 15분 |

### 10.3 다음 단계

- **Feature Engineering**: 선택된 변수로 새로운 변수 생성
- **모델 최적화**: Hyperparameter tuning
- **모델 해석**: SHAP 심화 분석
- **A/B 테스트**: 실제 환경에서 검증

---

**작성일**: 2025-01-25  
**버전**: 1.0  
**난이도**: ⭐⭐⭐ (고급)  
**예상 소요 시간**: 3-4시간 (학습 및 실습)
