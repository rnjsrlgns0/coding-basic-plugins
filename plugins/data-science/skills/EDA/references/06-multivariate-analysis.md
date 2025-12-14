# 06. Multivariate Analysis (다변량 분석)

## 1. 개요

### 1.1 목적
다변량 분석(Multivariate Analysis)은 3개 이상의 변수 간 복합적인 관계와 패턴을 동시에 파악하는 분석 기법입니다. 이변량 분석이 놓치는 고차원 상호작용과 숨겨진 구조를 발견할 수 있습니다.

### 1.2 적용 시기
- 변수 간 복잡한 상호작용을 이해하고자 할 때
- 고차원 데이터의 패턴을 2D/3D로 시각화하고 싶을 때
- 변수 간 숨겨진 구조(cluster, group)를 탐색할 때
- Feature engineering 전 변수 간 관계를 종합적으로 파악할 때
- 차원 축소가 필요한지 판단하고자 할 때

### 1.3 주요 기법
- **Pairplot**: 모든 변수 쌍의 관계를 한 번에 시각화
- **3D Scatter Plot**: 3개 변수의 공간적 관계 표현
- **차원 축소**: PCA, t-SNE, UMAP
- **Parallel Coordinates**: 다변량 패턴 시각화
- **Andrews Curves**: 고차원 데이터의 클러스터 탐지

---

## 2. 이론적 배경

### 2.1 다변량 분석의 필요성

**문제 상황**:
```
고차원 데이터(features > 3)에서:
- 변수 A와 B는 상관성이 낮지만, 변수 C를 고려하면 강한 관계 발견
- 개별 변수는 타겟과 약한 상관이지만, 조합하면 강력한 예측력
- 데이터에 숨겨진 클러스터 존재 (비지도 학습 가능성)
```

**해결 방법**:
- **동시 시각화**: 여러 변수를 동시에 보는 시각화 기법
- **차원 축소**: 고차원을 2D/3D로 변환하여 패턴 파악
- **군집 분석**: 숨겨진 그룹 구조 탐지

### 2.2 시나리오

**시나리오 1: 다중 변수 상호작용**
```
상황: 고객 이탈 예측 모델 개발
- 개별 변수(age, tenure, usage)는 이탈과 약한 상관
- 그러나 (young + low_tenure + high_usage) 조합은 이탈 확률 높음
→ 다변량 분석으로 상호작용 패턴 발견
```

**시나리오 2: 고차원 데이터 클러스터 탐지**
```
상황: 100개 features를 가진 유전자 발현 데이터
- 개별 변수로는 패턴 파악 불가
- t-SNE/UMAP으로 2D 변환 → 명확한 3개 클러스터 발견
→ 서브타입 존재 가능성 발견
```

**시나리오 3: 차원의 저주(Curse of Dimensionality)**
```
상황: 50개 features, 1000개 샘플
- 모델 성능 저하 (overfitting)
- PCA 적용 → 95% 분산을 10개 주성분으로 설명 가능
→ 차원 축소로 모델 효율성 향상
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
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# 차원 축소 라이브러리
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

# 전처리
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 시각화 설정
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
%matplotlib inline

# 한글 폰트 설정 (선택)
plt.rcParams['font.family'] = 'AppleGothic'  # Mac
# plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False
```

### 3.2 샘플 데이터 생성

```python
# 실습용 다변량 데이터 생성
np.random.seed(42)

def generate_multivariate_data(n_samples=500):
    """
    3개 클러스터를 가진 고차원 데이터 생성
    """
    # 클러스터 1: 젊은 고소득층
    cluster1 = pd.DataFrame({
        'age': np.random.normal(30, 5, n_samples // 3),
        'income': np.random.normal(80000, 10000, n_samples // 3),
        'spending': np.random.normal(3000, 500, n_samples // 3),
        'education': np.random.normal(16, 2, n_samples // 3),
        'family_size': np.random.poisson(2, n_samples // 3),
        'cluster': 0
    })
    
    # 클러스터 2: 중년 중소득층
    cluster2 = pd.DataFrame({
        'age': np.random.normal(45, 7, n_samples // 3),
        'income': np.random.normal(55000, 8000, n_samples // 3),
        'spending': np.random.normal(2000, 400, n_samples // 3),
        'education': np.random.normal(14, 2, n_samples // 3),
        'family_size': np.random.poisson(3, n_samples // 3),
        'cluster': 1
    })
    
    # 클러스터 3: 고령 저소득층
    cluster3 = pd.DataFrame({
        'age': np.random.normal(65, 8, n_samples // 3 + n_samples % 3),
        'income': np.random.normal(35000, 5000, n_samples // 3 + n_samples % 3),
        'spending': np.random.normal(1200, 300, n_samples // 3 + n_samples % 3),
        'education': np.random.normal(12, 2, n_samples // 3 + n_samples % 3),
        'family_size': np.random.poisson(1, n_samples // 3 + n_samples % 3),
        'cluster': 2
    })
    
    # 데이터 결합
    df = pd.concat([cluster1, cluster2, cluster3], ignore_index=True)
    
    # 추가 변수 생성 (상호작용)
    df['income_age_ratio'] = df['income'] / df['age']
    df['spending_ratio'] = df['spending'] / df['income']
    df['education_income'] = df['education'] * df['income'] / 1000
    
    return df

# 데이터 생성
df = generate_multivariate_data(600)
print(f"데이터 크기: {df.shape}")
print(f"\n기본 정보:")
print(df.info())
print(f"\n기술 통계:")
print(df.describe())
```

### 3.3 Pairplot: 모든 변수 쌍 관계 시각화

```python
def create_pairplot(df, hue_col='cluster', vars_to_plot=None):
    """
    모든 변수 쌍의 관계를 시각화
    
    Parameters:
    -----------
    df : DataFrame
        분석할 데이터
    hue_col : str
        색상 구분 변수 (범주형)
    vars_to_plot : list
        플롯할 변수 목록 (None이면 모든 수치형 변수)
    """
    if vars_to_plot is None:
        vars_to_plot = df.select_dtypes(include=[np.number]).columns.tolist()
        if hue_col in vars_to_plot:
            vars_to_plot.remove(hue_col)
    
    # Pairplot 생성
    g = sns.pairplot(
        df,
        vars=vars_to_plot[:6],  # 최대 6개 변수 (너무 많으면 가독성 저하)
        hue=hue_col,
        diag_kind='kde',        # 대각선: KDE plot
        plot_kws={'alpha': 0.6, 's': 50},
        diag_kws={'alpha': 0.7},
        height=2.5
    )
    
    g.fig.suptitle('Pairplot: 모든 변수 쌍 관계', y=1.02, fontsize=16)
    plt.tight_layout()
    
    return g

# 주요 변수만 선택하여 Pairplot
selected_vars = ['age', 'income', 'spending', 'education', 'family_size']
pairplot_fig = create_pairplot(df, hue_col='cluster', vars_to_plot=selected_vars)
plt.show()

# 해석 가이드 출력
print("📊 Pairplot 해석 가이드:")
print("=" * 60)
print("1. 대각선 (KDE): 각 변수의 분포 확인")
print("   - 클러스터별 분포 차이가 클수록 변수가 그룹을 잘 구분")
print("\n2. 비대각선 (Scatter): 변수 쌍의 관계")
print("   - 선형 패턴: 양/음의 상관관계")
print("   - 클러스터 분리: 명확한 그룹 구조 존재")
print("   - 겹침: 변수 쌍으로는 그룹 구분 어려움")
print("\n3. 색상 분리도")
print("   - 색상이 명확히 분리: 변수들이 그룹을 잘 설명")
print("   - 색상이 섞임: 추가 변수 또는 비선형 분석 필요")
```

### 3.4 3D Scatter Plot: 3개 변수의 공간적 관계

```python
def create_3d_scatter(df, x_col, y_col, z_col, color_col=None, title="3D Scatter Plot"):
    """
    3개 변수의 3차원 산점도
    
    Parameters:
    -----------
    df : DataFrame
    x_col, y_col, z_col : str
        각 축에 표시할 변수
    color_col : str
        색상 구분 변수 (범주형 또는 수치형)
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if color_col is not None:
        if df[color_col].dtype in ['object', 'category'] or df[color_col].nunique() < 10:
            # 범주형: 그룹별 색상
            for label in df[color_col].unique():
                mask = df[color_col] == label
                ax.scatter(
                    df.loc[mask, x_col],
                    df.loc[mask, y_col],
                    df.loc[mask, z_col],
                    label=f'{color_col}={label}',
                    s=60,
                    alpha=0.6
                )
            ax.legend()
        else:
            # 수치형: 연속 색상
            scatter = ax.scatter(
                df[x_col], df[y_col], df[z_col],
                c=df[color_col],
                cmap='viridis',
                s=60,
                alpha=0.6
            )
            fig.colorbar(scatter, ax=ax, label=color_col, shrink=0.5)
    else:
        ax.scatter(df[x_col], df[y_col], df[z_col], s=60, alpha=0.6)
    
    # 축 레이블
    ax.set_xlabel(x_col, fontsize=12, labelpad=10)
    ax.set_ylabel(y_col, fontsize=12, labelpad=10)
    ax.set_zlabel(z_col, fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, pad=20)
    
    # 회전 애니메이션 효과 (선택)
    # ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    return fig, ax

# 3D Scatter Plot 생성
fig1, ax1 = create_3d_scatter(
    df, 
    x_col='age', 
    y_col='income', 
    z_col='spending',
    color_col='cluster',
    title='고객 세그먼트 3D 시각화'
)
plt.show()

# 다른 변수 조합
fig2, ax2 = create_3d_scatter(
    df, 
    x_col='education', 
    y_col='income', 
    z_col='spending_ratio',
    color_col='cluster',
    title='교육-소득-소비비율 관계'
)
plt.show()

print("\n💡 3D Scatter Plot 활용 팁:")
print("=" * 60)
print("1. 마우스로 드래그하여 다양한 각도에서 관찰")
print("2. 명확한 3개 그룹 분리 → 클러스터링/분류 모델 효과적")
print("3. 그룹이 겹침 → 추가 변수 필요 또는 비선형 모델 고려")
print("4. 특이점(outlier) 쉽게 발견 가능")
```

### 3.5 차원 축소: PCA (Principal Component Analysis)

```python
def apply_pca(df, n_components=2, target_col=None):
    """
    PCA를 적용하여 고차원 데이터를 2D/3D로 축소
    
    Parameters:
    -----------
    df : DataFrame
    n_components : int
        축소할 차원 수 (2 또는 3)
    target_col : str
        타겟 변수 (시각화 색상용, PCA에서 제외)
    
    Returns:
    --------
    df_pca : DataFrame
        PCA 변환된 데이터
    pca : PCA object
        학습된 PCA 객체
    """
    # 수치형 변수만 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    X = df[numeric_cols].copy()
    
    # 결측값 처리 (있는 경우)
    if X.isnull().any().any():
        X = X.fillna(X.mean())
    
    # 표준화 (PCA는 스케일에 민감)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA 적용
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # 결과 DataFrame 생성
    pca_cols = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
    
    if target_col:
        df_pca[target_col] = df[target_col].values
    
    # PCA 정보 출력
    print("=" * 70)
    print("📊 PCA 결과 요약")
    print("=" * 70)
    print(f"원본 변수 개수: {len(numeric_cols)}")
    print(f"축소 차원: {n_components}")
    print(f"\n설명된 분산 비율:")
    for i, var_ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var_ratio:.2%}")
    print(f"  누적: {pca.explained_variance_ratio_.sum():.2%}")
    
    print(f"\n주성분 로딩 (각 변수의 기여도):")
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=pca_cols,
        index=numeric_cols
    )
    print(loadings.round(3))
    
    # 가장 중요한 변수 (PC1 기준)
    top_features = loadings['PC1'].abs().sort_values(ascending=False).head(5)
    print(f"\nPC1에 가장 큰 영향을 주는 변수:")
    for var, loading in top_features.items():
        print(f"  {var}: {loading:.3f}")
    
    return df_pca, pca, scaler

# PCA 적용 (2D)
df_pca_2d, pca_2d, scaler = apply_pca(df, n_components=2, target_col='cluster')

# PCA 결과 시각화 (2D)
plt.figure(figsize=(12, 8))
for cluster in df_pca_2d['cluster'].unique():
    mask = df_pca_2d['cluster'] == cluster
    plt.scatter(
        df_pca_2d.loc[mask, 'PC1'],
        df_pca_2d.loc[mask, 'PC2'],
        label=f'Cluster {cluster}',
        s=60,
        alpha=0.6
    )

plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} 분산)', fontsize=12)
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} 분산)', fontsize=12)
plt.title('PCA 2D 시각화', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Scree Plot (설명된 분산)
pca_full = PCA()
pca_full.fit(scaler.transform(df.select_dtypes(include=[np.number]).drop('cluster', axis=1)))

plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(pca_full.explained_variance_ratio_) + 1),
    np.cumsum(pca_full.explained_variance_ratio_),
    'bo-',
    linewidth=2,
    markersize=8
)
plt.xlabel('주성분 개수', fontsize=12)
plt.ylabel('누적 설명 분산 비율', fontsize=12)
plt.title('Scree Plot: 필요한 주성분 개수 결정', fontsize=14)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% 분산')
plt.axhline(y=0.90, color='orange', linestyle='--', label='90% 분산')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n💡 PCA 해석 가이드:")
print("=" * 60)
print("1. 설명된 분산 비율:")
print("   - PC1+PC2 > 80%: 2D로 충분히 설명 가능")
print("   - PC1+PC2 < 60%: 추가 차원 필요 (3D 고려)")
print("\n2. Scree Plot:")
print("   - Elbow 지점: 최적 주성분 개수")
print("   - 95% 선 도달 지점: 권장 주성분 개수")
print("\n3. 주성분 로딩:")
print("   - |loading| > 0.5: 해당 변수가 주성분에 강하게 기여")
print("   - PC1: 보통 전체 스케일/크기 반영")
print("   - PC2: 보통 두 번째로 중요한 패턴")
```

### 3.6 차원 축소: t-SNE (비선형 패턴)

```python
def apply_tsne(df, target_col=None, perplexity=30, random_state=42):
    """
    t-SNE를 적용하여 비선형 관계 시각화
    
    Parameters:
    -----------
    df : DataFrame
    target_col : str
        타겟 변수
    perplexity : int (5-50)
        이웃 개수 파라미터 (작을수록 로컬 구조 강조)
    
    Returns:
    --------
    df_tsne : DataFrame
    """
    # 수치형 변수 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # t-SNE 적용 (경고: 시간 소요 가능)
    print("⏳ t-SNE 실행 중... (1-2분 소요 가능)")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
        verbose=0
    )
    X_tsne = tsne.fit_transform(X_scaled)
    
    df_tsne = pd.DataFrame(
        X_tsne,
        columns=['t-SNE1', 't-SNE2'],
        index=df.index
    )
    
    if target_col:
        df_tsne[target_col] = df[target_col].values
    
    print("✅ t-SNE 완료!")
    return df_tsne

# t-SNE 적용
df_tsne = apply_tsne(df, target_col='cluster', perplexity=30)

# t-SNE 시각화
plt.figure(figsize=(12, 8))
for cluster in df_tsne['cluster'].unique():
    mask = df_tsne['cluster'] == cluster
    plt.scatter(
        df_tsne.loc[mask, 't-SNE1'],
        df_tsne.loc[mask, 't-SNE2'],
        label=f'Cluster {cluster}',
        s=60,
        alpha=0.6
    )

plt.xlabel('t-SNE Component 1', fontsize=12)
plt.ylabel('t-SNE Component 2', fontsize=12)
plt.title('t-SNE 2D 시각화 (비선형 패턴)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n💡 t-SNE vs PCA 비교:")
print("=" * 60)
print("| 특성          | PCA                | t-SNE              |")
print("|---------------|--------------------|--------------------|")
print("| 관계 탐지     | 선형               | 비선형             |")
print("| 글로벌 구조   | 보존 ✅            | 일부 손실          |")
print("| 로컬 구조     | 일부 손실          | 보존 ✅            |")
print("| 계산 속도     | 빠름 ✅            | 느림               |")
print("| 재현성        | 항상 동일 ✅       | 랜덤 시드 필요     |")
print("| 해석 가능성   | 높음 ✅            | 낮음 (시각화 전용) |")
print("| 추천 상황     | 빠른 탐색, 전처리  | 복잡한 패턴 탐지   |")
```

### 3.7 차원 축소: UMAP (빠르고 정확한 비선형)

```python
def apply_umap(df, target_col=None, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    UMAP을 적용하여 고차원 데이터 시각화
    
    Parameters:
    -----------
    df : DataFrame
    target_col : str
    n_neighbors : int (2-100)
        이웃 개수 (크면 글로벌, 작으면 로컬 구조 강조)
    min_dist : float (0.0-0.99)
        포인트 간 최소 거리 (작으면 조밀, 크면 분산)
    
    Returns:
    --------
    df_umap : DataFrame
    """
    # 수치형 변수 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # UMAP 적용
    print("⏳ UMAP 실행 중... (t-SNE보다 빠름)")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        verbose=False
    )
    X_umap = reducer.fit_transform(X_scaled)
    
    df_umap = pd.DataFrame(
        X_umap,
        columns=['UMAP1', 'UMAP2'],
        index=df.index
    )
    
    if target_col:
        df_umap[target_col] = df[target_col].values
    
    print("✅ UMAP 완료!")
    return df_umap

# UMAP 적용
df_umap = apply_umap(df, target_col='cluster', n_neighbors=15, min_dist=0.1)

# UMAP 시각화
plt.figure(figsize=(12, 8))
for cluster in df_umap['cluster'].unique():
    mask = df_umap['cluster'] == cluster
    plt.scatter(
        df_umap.loc[mask, 'UMAP1'],
        df_umap.loc[mask, 'UMAP2'],
        label=f'Cluster {cluster}',
        s=60,
        alpha=0.6
    )

plt.xlabel('UMAP Component 1', fontsize=12)
plt.ylabel('UMAP Component 2', fontsize=12)
plt.title('UMAP 2D 시각화 (빠른 비선형)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 파라미터 영향 비교
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

params = [
    {'n_neighbors': 5, 'min_dist': 0.1, 'title': 'n_neighbors=5, min_dist=0.1'},
    {'n_neighbors': 50, 'min_dist': 0.1, 'title': 'n_neighbors=50, min_dist=0.1'},
    {'n_neighbors': 15, 'min_dist': 0.01, 'title': 'n_neighbors=15, min_dist=0.01'},
    {'n_neighbors': 15, 'min_dist': 0.5, 'title': 'n_neighbors=15, min_dist=0.5'},
]

for ax, param in zip(axes.flat, params):
    df_temp = apply_umap(
        df, 
        target_col='cluster',
        n_neighbors=param['n_neighbors'],
        min_dist=param['min_dist'],
        random_state=42
    )
    
    for cluster in df_temp['cluster'].unique():
        mask = df_temp['cluster'] == cluster
        ax.scatter(
            df_temp.loc[mask, 'UMAP1'],
            df_temp.loc[mask, 'UMAP2'],
            label=f'Cluster {cluster}',
            s=40,
            alpha=0.6
        )
    
    ax.set_xlabel('UMAP1', fontsize=10)
    ax.set_ylabel('UMAP2', fontsize=10)
    ax.set_title(param['title'], fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n💡 UMAP 파라미터 가이드:")
print("=" * 60)
print("n_neighbors (이웃 개수):")
print("  - 작은 값 (5-10): 로컬 구조 강조, 세밀한 클러스터")
print("  - 큰 값 (30-50): 글로벌 구조 보존, 큰 그림")
print("\nmin_dist (최소 거리):")
print("  - 작은 값 (0.0-0.1): 포인트가 조밀하게 모임")
print("  - 큰 값 (0.3-0.99): 포인트가 넓게 분산")
print("\n추천 시작값:")
print("  - 일반적: n_neighbors=15, min_dist=0.1")
print("  - 세밀한 클러스터: n_neighbors=5, min_dist=0.01")
print("  - 전체 구조: n_neighbors=50, min_dist=0.3")
```

### 3.8 Parallel Coordinates Plot

```python
def create_parallel_coordinates(df, class_column, features=None, alpha=0.6):
    """
    Parallel Coordinates Plot: 다변량 패턴 시각화
    
    Parameters:
    -----------
    df : DataFrame
    class_column : str
        색상 구분 변수
    features : list
        표시할 변수 목록 (None이면 모든 수치형)
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        if class_column in features:
            features.remove(class_column)
    
    # 너무 많은 변수 제한 (가독성)
    if len(features) > 8:
        print(f"⚠️  변수가 {len(features)}개로 너무 많습니다. 상위 8개만 표시합니다.")
        features = features[:8]
    
    # 데이터 정규화 (0-1 스케일)
    df_plot = df[features + [class_column]].copy()
    for col in features:
        min_val = df_plot[col].min()
        max_val = df_plot[col].max()
        df_plot[col] = (df_plot[col] - min_val) / (max_val - min_val)
    
    # Parallel Coordinates Plot
    from pandas.plotting import parallel_coordinates
    
    plt.figure(figsize=(14, 8))
    parallel_coordinates(
        df_plot,
        class_column=class_column,
        cols=features,
        alpha=alpha,
        linewidth=1.5
    )
    
    plt.title('Parallel Coordinates Plot', fontsize=14)
    plt.xlabel('변수', fontsize=12)
    plt.ylabel('정규화된 값 (0-1)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title=class_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    print("\n💡 Parallel Coordinates 해석:")
    print("=" * 60)
    print("1. 선의 패턴:")
    print("   - 평행한 선들: 해당 클래스의 일관된 특성")
    print("   - 교차하는 선들: 변수 간 상호작용 존재")
    print("\n2. 색상 분리:")
    print("   - 특정 축에서 색상 분리: 해당 변수가 클래스를 잘 구분")
    print("   - 모든 색상 혼재: 변수가 클래스 구분에 비효과적")
    print("\n3. 활용:")
    print("   - Feature selection: 분리도 높은 변수 선택")
    print("   - 패턴 발견: 클래스별 특징적 프로파일 파악")

# Parallel Coordinates Plot 생성
selected_features = ['age', 'income', 'spending', 'education', 'family_size']
create_parallel_coordinates(df, class_column='cluster', features=selected_features)
```

### 3.9 Andrews Curves

```python
def create_andrews_curves(df, class_column, features=None, alpha=0.6):
    """
    Andrews Curves: 고차원 데이터의 클러스터 시각화
    
    각 관측치를 푸리에 함수로 변환하여 곡선으로 표현
    비슷한 관측치는 비슷한 곡선 형태
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        if class_column in features:
            features.remove(class_column)
    
    # 너무 많은 변수 제한
    if len(features) > 10:
        features = features[:10]
    
    df_plot = df[features + [class_column]].copy()
    
    # Andrews Curves
    from pandas.plotting import andrews_curves
    
    plt.figure(figsize=(14, 8))
    andrews_curves(
        df_plot,
        class_column=class_column,
        alpha=alpha,
        linewidth=1.2
    )
    
    plt.title('Andrews Curves', fontsize=14)
    plt.xlabel('t', fontsize=12)
    plt.ylabel('f(t)', fontsize=12)
    plt.legend(title=class_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n💡 Andrews Curves 해석:")
    print("=" * 60)
    print("1. 곡선의 형태:")
    print("   - 비슷한 관측치 → 비슷한 곡선 형태")
    print("   - 다른 클래스 → 다른 곡선 패턴")
    print("\n2. 클러스터 분리:")
    print("   - 색상별로 곡선이 분리: 명확한 클러스터 구조")
    print("   - 곡선이 혼재: 클러스터 경계 불명확")
    print("\n3. 이상치 탐지:")
    print("   - 다른 곡선들과 동떨어진 곡선: 잠재적 이상치")

# Andrews Curves 생성
create_andrews_curves(df, class_column='cluster', features=selected_features)
```

### 3.10 차원 축소 비교 종합

```python
def compare_dimensionality_reduction(df, target_col='cluster', sample_size=None):
    """
    PCA, t-SNE, UMAP을 한 번에 비교
    
    Parameters:
    -----------
    df : DataFrame
    target_col : str
    sample_size : int
        샘플 크기 (None이면 전체, t-SNE 속도 고려)
    """
    # 샘플링 (t-SNE 속도 고려)
    if sample_size and len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
        print(f"⚠️  t-SNE 속도를 위해 {sample_size}개 샘플만 사용")
    else:
        df_sample = df.copy()
    
    # 3가지 방법 적용
    df_pca, _, _ = apply_pca(df_sample, n_components=2, target_col=target_col)
    df_tsne = apply_tsne(df_sample, target_col=target_col, perplexity=30)
    df_umap = apply_umap(df_sample, target_col=target_col, n_neighbors=15)
    
    # 시각화 비교
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    results = [
        (df_pca, 'PC1', 'PC2', 'PCA', axes[0]),
        (df_tsne, 't-SNE1', 't-SNE2', 't-SNE', axes[1]),
        (df_umap, 'UMAP1', 'UMAP2', 'UMAP', axes[2])
    ]
    
    for df_result, x_col, y_col, method_name, ax in results:
        for cluster in df_result[target_col].unique():
            mask = df_result[target_col] == cluster
            ax.scatter(
                df_result.loc[mask, x_col],
                df_result.loc[mask, y_col],
                label=f'Cluster {cluster}',
                s=50,
                alpha=0.6
            )
        
        ax.set_xlabel(f'{method_name} Component 1', fontsize=11)
        ax.set_ylabel(f'{method_name} Component 2', fontsize=11)
        ax.set_title(f'{method_name} 시각화', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 70)
    print("📊 차원 축소 방법 선택 가이드")
    print("=" * 70)
    print("\n1. PCA 추천:")
    print("   ✅ 빠른 탐색이 필요할 때")
    print("   ✅ 선형 관계가 주된 데이터")
    print("   ✅ 설명 가능한 변수 축소 (feature engineering)")
    print("   ✅ 전처리로 차원 축소 (모델 입력)")
    
    print("\n2. t-SNE 추천:")
    print("   ✅ 복잡한 비선형 관계")
    print("   ✅ 로컬 클러스터 구조 탐지")
    print("   ✅ 시각화 목적 (논문, 보고서)")
    print("   ⚠️  대용량 데이터는 느림 (샘플링 권장)")
    
    print("\n3. UMAP 추천:")
    print("   ✅ t-SNE의 장점 + 빠른 속도")
    print("   ✅ 글로벌 + 로컬 구조 모두 보존")
    print("   ✅ 대용량 데이터")
    print("   ✅ 범용적 선택 (시각화 + 전처리)")
    
    print("\n4. 실무 전략:")
    print("   1단계: PCA로 빠른 탐색")
    print("   2단계: UMAP으로 정밀 분석")
    print("   3단계: 필요시 t-SNE로 최종 검증")

# 비교 실행
compare_dimensionality_reduction(df, target_col='cluster', sample_size=500)
```

---

## 4. 예시

### 4.1 실전 예제: 고객 세그먼트 분석

```python
# 실제 비즈니스 시나리오
print("=" * 70)
print("📈 비즈니스 시나리오: 온라인 쇼핑몰 고객 세그먼트 분석")
print("=" * 70)
print("\n목표:")
print("- 10개 features를 가진 고객 데이터에서 숨겨진 세그먼트 발견")
print("- 세그먼트별 특성 파악 및 마케팅 전략 수립")
print("\n데이터:")
print("- 고객 5,000명")
print("- Features: 나이, 소득, 구매빈도, 평균구매액, 체류시간 등 10개")

# 워크플로우
print("\n🔄 분석 워크플로우:")
print("-" * 70)
print("1. Pairplot으로 전체 변수 관계 파악")
print("   → 발견: age-income, spending-frequency 강한 상관")
print("\n2. 3D Scatter로 주요 3개 변수 공간적 관계 확인")
print("   → 발견: 3개 그룹이 명확히 분리됨")
print("\n3. PCA로 차원 축소 및 중요 변수 식별")
print("   → 발견: PC1(소비력), PC2(활동성) 2개로 80% 설명")
print("\n4. UMAP으로 비선형 패턴 탐지")
print("   → 발견: 4번째 소규모 세그먼트 발견 (VIP 고객)")
print("\n5. Parallel Coordinates로 세그먼트별 프로파일 작성")
print("   → 발견: 각 세그먼트의 특징적 패턴")

# 결과
print("\n✅ 분석 결과:")
print("-" * 70)
print("세그먼트 1: 젊은 고소득층 (30대, 고소득, 고빈도 구매)")
print("  → 전략: 프리미엄 상품 추천, 멤버십 프로그램")
print("\n세그먼트 2: 중년 가족층 (40대, 중소득, 대량 구매)")
print("  → 전략: 가족 패키지, 할인 쿠폰")
print("\n세그먼트 3: 시니어 절약층 (60대, 저소득, 가격 민감)")
print("  → 전략: 시니어 할인, 필수품 중심")
print("\n세그먼트 4: VIP (소수, 초고소득, 프리미엄)")
print("  → 전략: 1:1 맞춤 서비스, 전용 이벤트")
```

### 4.2 입출력 예시

```python
# 입력 데이터 샘플
print("\n📥 입력 데이터 샘플:")
print(df.head(10))

# PCA 변환 결과
print("\n📤 PCA 변환 결과:")
print(df_pca_2d.head(10))

# UMAP 변환 결과
print("\n📤 UMAP 변환 결과:")
print(df_umap.head(10))

# 변수 중요도 (PCA 로딩)
print("\n📊 변수 중요도 (PCA 로딩):")
numeric_cols = df.select_dtypes(include=[np.number]).drop('cluster', axis=1).columns
loadings_df = pd.DataFrame(
    pca_2d.components_.T,
    columns=['PC1', 'PC2'],
    index=numeric_cols
)
loadings_df['Importance'] = np.sqrt(loadings_df['PC1']**2 + loadings_df['PC2']**2)
print(loadings_df.sort_values('Importance', ascending=False))
```

---

## 5. 에이전트 매핑

### 5.1 담당 에이전트

| 작업 | Primary Agent | Supporting Agents |
|------|--------------|-------------------|
| Pairplot, 3D Scatter | `data-visualization-specialist` | `data-scientist` |
| PCA 분석 및 해석 | `data-scientist` | `feature-engineering-specialist` |
| t-SNE, UMAP 실행 | `data-scientist` | - |
| 차원 축소 결과 해석 | `feature-engineering-specialist` | `data-scientist` |
| 비즈니스 인사이트 도출 | `data-scientist` | - |

### 5.2 관련 스킬

**Scientific Skills**:
- `matplotlib` (3D plotting)
- `seaborn` (pairplot, 고급 시각화)
- `scikit-learn` (PCA, StandardScaler)
- `umap-learn` (UMAP 차원 축소)
- `pandas` (데이터 조작)

**추가 도구**:
- `plotly` (인터랙티브 3D 시각화, 선택)
- `prince` (다중 대응 분석, MCA, 선택)

---

## 6. 필요 라이브러리

### 6.1 필수 라이브러리

```bash
# 데이터 처리
pip install pandas==2.2.0
pip install numpy==1.26.3

# 시각화
pip install matplotlib==3.8.2
pip install seaborn==0.13.1

# 머신러닝 및 차원 축소
pip install scikit-learn==1.4.0
pip install umap-learn==0.5.5

# 선택 (고급 시각화)
pip install plotly==5.18.0
```

### 6.2 라이브러리 버전 확인

```python
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import sklearn
import umap

print("라이브러리 버전:")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"matplotlib: {matplotlib.__version__}")
print(f"seaborn: {sns.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"umap-learn: {umap.__version__}")
```

---

## 7. 체크포인트

### 7.1 분석 전 체크리스트

- [ ] **데이터 준비**
  - [ ] 결측값 처리 완료
  - [ ] 이상치 확인 완료
  - [ ] 변수 타입 확인 (수치형만 분석 가능)

- [ ] **변수 선택**
  - [ ] 분석 목적에 맞는 변수 선정
  - [ ] 너무 많은 변수는 제외 (pairplot: 6개 이하)
  - [ ] 타겟 변수 명확히 정의 (있는 경우)

- [ ] **스케일링 필요성**
  - [ ] PCA, t-SNE, UMAP은 스케일링 필수
  - [ ] StandardScaler 적용 확인

### 7.2 분석 중 체크리스트

- [ ] **Pairplot**
  - [ ] 클러스터가 명확히 분리되는가?
  - [ ] 강한 상관관계 변수 쌍은?
  - [ ] 비선형 관계가 있는가?

- [ ] **차원 축소 (PCA)**
  - [ ] 설명된 분산 비율 > 80%?
  - [ ] Scree plot에서 elbow 지점은?
  - [ ] 주요 변수(high loading)는?

- [ ] **차원 축소 (t-SNE/UMAP)**
  - [ ] 클러스터가 명확히 보이는가?
  - [ ] 파라미터 조정이 필요한가?
  - [ ] PCA와 다른 패턴이 보이는가?

### 7.3 분석 후 체크리스트

- [ ] **인사이트 도출**
  - [ ] 명확한 클러스터/그룹이 있는가?
  - [ ] 중요 변수를 식별했는가?
  - [ ] 비선형 관계를 발견했는가?

- [ ] **액션 아이템**
  - [ ] Feature selection 필요 여부
  - [ ] 차원 축소 적용 여부 (모델 입력)
  - [ ] 추가 분석 필요 영역

---

## 8. 트러블슈팅

### 8.1 일반적 오류

**문제 1: PCA 적용 시 `ValueError: could not convert string to float`**

```python
# 원인: 비수치형 변수 포함
# 해결:
numeric_cols = df.select_dtypes(include=[np.number]).columns
X = df[numeric_cols]
```

**문제 2: t-SNE가 너무 느림**

```python
# 원인: 대용량 데이터 (n > 10,000)
# 해결: 샘플링
df_sample = df.sample(5000, random_state=42)
df_tsne = apply_tsne(df_sample, target_col='cluster')
```

**문제 3: UMAP 설치 오류 (`libumap.so not found`)**

```bash
# 해결 (Mac):
brew install llvm libomp
pip install umap-learn

# 해결 (Linux):
sudo apt-get install libomp-dev
pip install umap-learn
```

**문제 4: Pairplot이 너무 복잡함**

```python
# 원인: 변수가 너무 많음 (> 8개)
# 해결: 주요 변수만 선택
important_vars = ['age', 'income', 'spending', 'education']
sns.pairplot(df, vars=important_vars, hue='cluster')
```

### 8.2 해석 관련

**Q1: PCA의 PC1, PC2는 무엇을 의미하나요?**

```
A: 주성분(Principal Component)은 원본 변수들의 선형 조합입니다.
- PC1: 데이터의 가장 큰 분산 방향 (보통 전체 크기/스케일)
- PC2: PC1과 직교하는 두 번째 분산 방향

예시:
PC1 = 0.5*income + 0.4*spending + 0.3*education + ...
→ "경제력 축"으로 해석 가능

PC2 = 0.6*age - 0.5*family_size + ...
→ "생애 주기 축"으로 해석 가능
```

**Q2: t-SNE와 UMAP 중 어떤 것을 선택해야 하나요?**

```
A: 일반적으로 UMAP을 추천합니다.
- UMAP: 더 빠르고, 글로벌 구조도 어느 정도 보존
- t-SNE: 로컬 구조에 더 집중, 느림

선택 기준:
- 빠른 탐색: UMAP
- 논문 품질 시각화: t-SNE (좀 더 알려짐)
- 대용량 데이터: UMAP
```

**Q3: 차원 축소 결과를 모델 입력으로 사용해도 되나요?**

```
A: 경우에 따라 다릅니다.
- PCA: ✅ 모델 입력 적합 (선형 변환, 해석 가능)
- t-SNE: ❌ 시각화 전용 (비결정적, 새 데이터 변환 불가)
- UMAP: ⚠️  가능하지만 주의 (transform 메서드 사용)

추천:
- Feature engineering: PCA
- 시각화: t-SNE, UMAP
```

### 8.3 성능 최적화

**대용량 데이터 처리**:

```python
# 1. 샘플링
df_sample = df.sample(min(10000, len(df)), random_state=42)

# 2. PCA로 사전 축소 후 t-SNE/UMAP
pca = PCA(n_components=50)  # 50차원으로 먼저 축소
X_pca = pca.fit_transform(X_scaled)
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X_pca)

# 3. UMAP 병렬 처리 (n_jobs)
reducer = umap.UMAP(n_components=2, n_jobs=-1)  # 모든 CPU 사용
```

---

## 9. 참고 자료

### 9.1 공식 문서

- **Scikit-learn PCA**: https://scikit-learn.org/stable/modules/decomposition.html#pca
- **Scikit-learn t-SNE**: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
- **UMAP**: https://umap-learn.readthedocs.io/en/latest/
- **Seaborn Pairplot**: https://seaborn.pydata.org/generated/seaborn.pairplot.html

### 9.2 베스트 프랙티스

1. **차원 축소 파이프라인**
   - 항상 StandardScaler 적용
   - PCA → t-SNE/UMAP 2단계 접근 (고차원일 때)
   - 재현성을 위해 random_state 고정

2. **시각화 전략**
   - Pairplot: 최대 6개 변수
   - 3D Plot: 마우스 인터랙션 활용
   - Parallel Coordinates: 클래스별 프로파일 비교

3. **파라미터 튜닝**
   - t-SNE perplexity: 5-50 (데이터 크기에 비례)
   - UMAP n_neighbors: 5-50 (로컬↔글로벌 균형)
   - PCA n_components: Scree plot으로 결정

### 9.3 추가 학습 자료

- **PCA 직관적 이해**: http://setosa.io/ev/principal-component-analysis/
- **t-SNE 설명**: https://distill.pub/2016/misread-tsne/
- **UMAP vs t-SNE 비교**: https://pair-code.github.io/understanding-umap/
- **차원 축소 종합 가이드**: https://scikit-learn.org/stable/modules/manifold.html

---

## 10. 요약

### 10.1 핵심 메시지

다변량 분석은 고차원 데이터의 숨겨진 패턴을 발견하는 강력한 도구입니다. Pairplot으로 전체 관계를 파악하고, 차원 축소(PCA, t-SNE, UMAP)로 복잡한 구조를 2D/3D로 시각화하여 클러스터, 이상치, 변수 중요도를 직관적으로 이해할 수 있습니다.

### 10.2 실무 적용 순서

1. **Pairplot**: 모든 변수 쌍 관계 빠른 탐색 (5분)
2. **3D Scatter**: 주요 3개 변수의 공간적 관계 확인 (5분)
3. **PCA**: 선형 차원 축소 및 중요 변수 식별 (10분)
4. **UMAP**: 비선형 패턴 탐지 및 클러스터 발견 (10분)
5. **Parallel Coordinates**: 세그먼트별 프로파일 작성 (5분)

**총 소요 시간**: 약 35분

### 10.3 다음 단계

- **클러스터 발견 시**: `14-advanced-segmentation.md` 참고
- **중요 변수 식별 시**: `09-feature-importance.md` 참고
- **시각화 심화**: `07-visualization-patterns.md` 참고
- **통계 검정**: `11-hypothesis-testing.md` 참고

---

**작성일**: 2025-01-25  
**버전**: 1.0  
**난이도**: ⭐⭐⭐ (고급)  
**예상 소요 시간**: 2-3시간 (학습 및 실습)
