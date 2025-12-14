# 13. Data Lineage (데이터 리니지)

**생성일**: 2025-01-26  
**버전**: 1.0  
**카테고리**: Data Governance & Traceability

---

## 1. 개요 (Overview)

### 1.1 목적 (Purpose)

데이터 리니지(Data Lineage)는 데이터의 출처, 이동 경로, 변환 과정을 추적하고 문서화하는 프로세스입니다. 데이터 클렌징 작업의 투명성, 재현성, 감사 가능성을 보장하며, 데이터 거버넌스의 핵심 요소입니다.

### 1.2 적용 시기 (When to Apply)

**필수 적용 시점**:
- ✅ 모든 데이터 변환 작업 시
- ✅ 규제 준수가 필요한 프로젝트 (금융, 의료 등)
- ✅ 프로덕션 환경 데이터 처리
- ✅ 감사(audit) 요구사항이 있는 경우

**권장 적용**:
- 🔹 복잡한 데이터 파이프라인
- 🔹 여러 팀이 협업하는 프로젝트
- 🔹 장기 운영 데이터 시스템
- 🔹 데이터 품질 이슈 디버깅

### 1.3 리니지 레벨 (Lineage Levels)

```
Level 1: Technical Lineage (기술적 리니지)
└── 코드 수준의 변환 기록 (함수, 파라미터)

Level 2: Operational Lineage (운영 리니지)
└── 실행 이력 (시간, 사용자, 결과)

Level 3: Business Lineage (비즈니스 리니지)
└── 비즈니스 컨텍스트 (목적, 영향, 승인)

Level 4: Data Lineage (데이터 리니지)
└── 데이터 흐름 및 의존성 (소스, 타겟, 관계)
```

---

## 2. 이론적 배경 (Theoretical Background)

### 2.1 데이터 리니지의 중요성

**1. 투명성 (Transparency)**
- 데이터가 어떻게 변환되었는지 명확히 파악
- 이해관계자에게 신뢰 제공

**2. 재현성 (Reproducibility)**
- 동일한 입력에 대해 동일한 출력 보장
- 과학적 방법론 준수

**3. 디버깅 (Debugging)**
- 데이터 품질 이슈 발생 시 원인 추적
- 문제 발생 지점 정확히 식별

**4. 규제 준수 (Compliance)**
- GDPR, HIPAA 등 규제 요구사항 충족
- 감사 추적(audit trail) 제공

**5. 영향 분석 (Impact Analysis)**
- 변경 사항의 다운스트림 영향 평가
- 의존성 관리

### 2.2 리니지 구성 요소

**메타데이터 (Metadata)**:
```python
{
    "operation_id": "unique_identifier",
    "timestamp": "2024-01-26 10:30:00",
    "operation_type": "impute_missing_values",
    "user": "data_engineer_1",
    "parameters": {
        "method": "knn",
        "n_neighbors": 5
    },
    "input_data": {
        "shape": (10000, 15),
        "hash": "abc123...",
        "source": "raw_data.csv"
    },
    "output_data": {
        "shape": (10000, 15),
        "hash": "def456...",
        "target": "imputed_data.csv"
    },
    "metrics": {
        "rows_affected": 1500,
        "execution_time": 2.5
    },
    "status": "success"
}
```

### 2.3 리니지 추적 패턴

**Pattern 1: Linear Lineage (선형 리니지)**
```
Raw Data → Cleaning → Transformation → Output
```

**Pattern 2: Branching Lineage (분기 리니지)**
```
                  → Branch A → Output A
Raw Data → Split → Branch B → Output B
                  → Branch C → Output C
```

**Pattern 3: Merging Lineage (병합 리니지)**
```
Source A → Cleaning A ↘
                      → Join → Output
Source B → Cleaning B ↗
```

---

## 3. 구현 (Implementation)

### 3.1 DataLineage 클래스

```python
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib
import json
import pickle

class DataLineage:
    """
    데이터 변환 이력 추적 클래스
    Track data transformation history
    """
    
    def __init__(
        self,
        dataset_name: str,
        initial_df: pd.DataFrame,
        project_name: str = "Data Cleansing Project"
    ):
        """
        Parameters:
        -----------
        dataset_name : str
            데이터셋 이름
        initial_df : pd.DataFrame
            초기 데이터프레임
        project_name : str
            프로젝트 이름
        """
        self.dataset_name = dataset_name
        self.project_name = project_name
        self.history = []
        self.current_df = initial_df.copy()
        
        # 초기 상태 기록
        self._log_initial_state(initial_df)
    
    def _log_initial_state(self, df: pd.DataFrame):
        """
        초기 데이터 상태 기록
        Log initial data state
        """
        initial_entry = {
            'operation_id': self._generate_id(),
            'timestamp': datetime.now().isoformat(),
            'operation_type': 'initial_load',
            'operation_name': 'Load Raw Data',
            'parameters': {},
            'data_before': self._capture_data_snapshot(df),
            'data_after': self._capture_data_snapshot(df),
            'changes': {
                'rows_added': 0,
                'rows_removed': 0,
                'columns_added': 0,
                'columns_removed': 0
            },
            'execution_time_seconds': 0.0,
            'status': 'success',
            'error': None,
            'user': 'system',
            'description': f'Initial load of {self.dataset_name}'
        }
        
        self.history.append(initial_entry)
    
    def _generate_id(self) -> str:
        """
        고유 ID 생성
        Generate unique ID
        """
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _capture_data_snapshot(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        데이터 스냅샷 캡처
        Capture data snapshot
        """
        snapshot = {
            'shape': df.shape,
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'memory_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            'hash': self._calculate_dataframe_hash(df),
            'missing_values': int(df.isnull().sum().sum()),
            'missing_rate': round(100 * df.isnull().sum().sum() / df.size, 2)
        }
        
        return snapshot
    
    def _calculate_dataframe_hash(self, df: pd.DataFrame) -> str:
        """
        데이터프레임 해시 계산 (데이터 무결성 검증용)
        Calculate dataframe hash for integrity verification
        """
        try:
            # 데이터프레임을 바이너리로 변환하여 해시 계산
            df_bytes = pickle.dumps(df)
            hash_obj = hashlib.md5(df_bytes)
            return hash_obj.hexdigest()[:16]
        except Exception:
            return "hash_error"
    
    def log_operation(
        self,
        operation_type: str,
        operation_name: str,
        df_after: pd.DataFrame,
        parameters: Dict[str, Any] = None,
        description: str = None,
        user: str = "system",
        execution_time: float = 0.0
    ):
        """
        데이터 변환 작업 기록
        Log a data transformation operation
        
        Parameters:
        -----------
        operation_type : str
            작업 유형 (예: 'impute_missing', 'remove_outliers')
        operation_name : str
            작업 이름 (사람이 읽기 쉬운 이름)
        df_after : pd.DataFrame
            변환 후 데이터프레임
        parameters : dict, optional
            작업 파라미터
        description : str, optional
            작업 설명
        user : str, optional
            작업 수행자
        execution_time : float, optional
            실행 시간 (초)
            
        Example:
        --------
        >>> lineage.log_operation(
        ...     operation_type='impute_missing',
        ...     operation_name='KNN Imputation',
        ...     df_after=df_imputed,
        ...     parameters={'method': 'knn', 'n_neighbors': 5},
        ...     description='Imputed missing values using KNN'
        ... )
        """
        # Before 스냅샷 (현재 상태)
        data_before = self._capture_data_snapshot(self.current_df)
        
        # After 스냅샷
        data_after = self._capture_data_snapshot(df_after)
        
        # 변경 사항 계산
        changes = self._calculate_changes(self.current_df, df_after)
        
        # 리니지 엔트리 생성
        entry = {
            'operation_id': self._generate_id(),
            'timestamp': datetime.now().isoformat(),
            'operation_type': operation_type,
            'operation_name': operation_name,
            'parameters': parameters or {},
            'data_before': data_before,
            'data_after': data_after,
            'changes': changes,
            'execution_time_seconds': execution_time,
            'status': 'success',
            'error': None,
            'user': user,
            'description': description or operation_name
        }
        
        self.history.append(entry)
        
        # 현재 상태 업데이트
        self.current_df = df_after.copy()
        
        print(f"✅ Logged operation: {operation_name} (ID: {entry['operation_id']})")
    
    def _calculate_changes(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        변경 사항 계산
        Calculate changes between before and after
        """
        # 행 변화
        rows_added = max(0, len(df_after) - len(df_before))
        rows_removed = max(0, len(df_before) - len(df_after))
        
        # 컬럼 변화
        cols_before = set(df_before.columns)
        cols_after = set(df_after.columns)
        
        columns_added = list(cols_after - cols_before)
        columns_removed = list(cols_before - cols_after)
        
        # 공통 컬럼의 변경된 셀 개수
        common_cols = cols_before & cols_after
        cells_modified = 0
        
        for col in common_cols:
            if col in df_before.columns and col in df_after.columns:
                # 인덱스 정렬 후 비교
                min_len = min(len(df_before), len(df_after))
                if min_len > 0:
                    try:
                        diff = df_before[col].iloc[:min_len] != df_after[col].iloc[:min_len]
                        cells_modified += diff.sum()
                    except:
                        pass  # 비교 불가능한 타입은 스킵
        
        changes = {
            'rows_added': int(rows_added),
            'rows_removed': int(rows_removed),
            'columns_added': columns_added,
            'columns_removed': columns_removed,
            'cells_modified': int(cells_modified)
        }
        
        return changes
    
    def get_lineage_report(self) -> pd.DataFrame:
        """
        리니지 리포트 생성
        Generate lineage report
        
        Returns:
        --------
        report_df : pd.DataFrame
            리니지 리포트 테이블
        """
        if not self.history:
            return pd.DataFrame()
        
        # 주요 필드만 추출
        report_data = []
        for entry in self.history:
            report_data.append({
                'operation_id': entry['operation_id'],
                'timestamp': entry['timestamp'],
                'operation_name': entry['operation_name'],
                'operation_type': entry['operation_type'],
                'rows_before': entry['data_before']['rows'],
                'rows_after': entry['data_after']['rows'],
                'cols_before': entry['data_before']['columns'],
                'cols_after': entry['data_after']['columns'],
                'rows_changed': entry['changes']['rows_added'] - entry['changes']['rows_removed'],
                'execution_time': entry['execution_time_seconds'],
                'status': entry['status'],
                'user': entry['user']
            })
        
        report_df = pd.DataFrame(report_data)
        
        return report_df
    
    def get_operation_details(self, operation_id: str) -> Dict[str, Any]:
        """
        특정 작업의 상세 정보 조회
        Get detailed information about a specific operation
        """
        for entry in self.history:
            if entry['operation_id'] == operation_id:
                return entry
        
        return None
    
    def export_lineage(self, filepath: str):
        """
        리니지 전체 내역을 파일로 저장
        Export full lineage to file
        
        Parameters:
        -----------
        filepath : str
            저장 경로 (JSON 형식)
        """
        export_data = {
            'project_name': self.project_name,
            'dataset_name': self.dataset_name,
            'created_at': self.history[0]['timestamp'] if self.history else None,
            'last_updated': self.history[-1]['timestamp'] if self.history else None,
            'total_operations': len(self.history),
            'history': self.history
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Lineage exported to {filepath}")
    
    def import_lineage(self, filepath: str):
        """
        리니지 내역 불러오기
        Import lineage from file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.project_name = data['project_name']
        self.dataset_name = data['dataset_name']
        self.history = data['history']
        
        print(f"✅ Lineage imported from {filepath}")
        print(f"   Total operations: {len(self.history)}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        리니지 요약 정보
        Get lineage summary
        """
        if not self.history:
            return {}
        
        first_entry = self.history[0]
        last_entry = self.history[-1]
        
        summary = {
            'project_name': self.project_name,
            'dataset_name': self.dataset_name,
            'total_operations': len(self.history),
            'start_time': first_entry['timestamp'],
            'end_time': last_entry['timestamp'],
            'initial_rows': first_entry['data_before']['rows'],
            'final_rows': last_entry['data_after']['rows'],
            'rows_changed': last_entry['data_after']['rows'] - first_entry['data_before']['rows'],
            'initial_columns': first_entry['data_before']['columns'],
            'final_columns': last_entry['data_after']['columns'],
            'total_execution_time': sum(e['execution_time_seconds'] for e in self.history),
            'operations_list': [e['operation_name'] for e in self.history]
        }
        
        return summary


# 사용 예시
def demo_lineage_tracking():
    """
    데이터 리니지 추적 데모
    """
    # 샘플 데이터
    np.random.seed(42)
    df = pd.DataFrame({
        'id': range(1, 1001),
        'value': np.random.randn(1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    # 결측값 삽입
    df.loc[df.sample(frac=0.1).index, 'value'] = np.nan
    
    # 리니지 추적 시작
    lineage = DataLineage(
        dataset_name='sample_data',
        initial_df=df,
        project_name='Data Cleansing Demo'
    )
    
    # 작업 1: 결측값 대체
    df_imputed = df.copy()
    df_imputed['value'].fillna(df_imputed['value'].mean(), inplace=True)
    
    lineage.log_operation(
        operation_type='impute_missing',
        operation_name='Mean Imputation',
        df_after=df_imputed,
        parameters={'method': 'mean', 'column': 'value'},
        description='Filled missing values with mean',
        execution_time=0.5
    )
    
    # 작업 2: 이상치 제거
    Q1 = df_imputed['value'].quantile(0.25)
    Q3 = df_imputed['value'].quantile(0.75)
    IQR = Q3 - Q1
    df_no_outliers = df_imputed[
        (df_imputed['value'] >= Q1 - 1.5 * IQR) &
        (df_imputed['value'] <= Q3 + 1.5 * IQR)
    ]
    
    lineage.log_operation(
        operation_type='remove_outliers',
        operation_name='IQR Outlier Removal',
        df_after=df_no_outliers,
        parameters={'method': 'IQR', 'multiplier': 1.5},
        description='Removed outliers using IQR method',
        execution_time=0.3
    )
    
    # 작업 3: 새 컬럼 추가
    df_final = df_no_outliers.copy()
    df_final['value_squared'] = df_final['value'] ** 2
    
    lineage.log_operation(
        operation_type='feature_engineering',
        operation_name='Add Squared Feature',
        df_after=df_final,
        parameters={'new_column': 'value_squared', 'formula': 'value ** 2'},
        description='Added squared value feature',
        execution_time=0.1
    )
    
    # 리니지 리포트 확인
    print("\n" + "="*80)
    print("LINEAGE REPORT")
    print("="*80)
    report = lineage.get_lineage_report()
    print(report)
    
    # 요약 정보
    print("\n" + "="*80)
    print("LINEAGE SUMMARY")
    print("="*80)
    summary = lineage.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # 리니지 저장
    lineage.export_lineage('lineage_history.json')
    
    return lineage
```

### 3.2 시각화 도구

```python
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch

class LineageVisualizer:
    """
    데이터 리니지 시각화 클래스
    Visualize data lineage
    """
    
    def __init__(self, lineage: DataLineage):
        self.lineage = lineage
        self.history = lineage.history
    
    def plot_data_flow(self, save_path: str = None):
        """
        데이터 흐름 다이어그램
        Plot data flow diagram
        """
        if not self.history:
            print("No lineage history to visualize")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 노드 생성 (각 작업)
        n_operations = len(self.history)
        node_positions = {}
        
        for i, entry in enumerate(self.history):
            x = i
            y = 0
            node_positions[i] = (x, y)
            
            # 노드 그리기
            if entry['operation_type'] == 'initial_load':
                color = '#E8F5E9'
                edge_color = '#4CAF50'
            elif entry['status'] == 'success':
                color = '#E3F2FD'
                edge_color = '#2196F3'
            else:
                color = '#FFEBEE'
                edge_color = '#F44336'
            
            # 박스 그리기
            box = FancyBboxPatch(
                (x - 0.4, y - 0.3),
                0.8,
                0.6,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor=edge_color,
                linewidth=2
            )
            ax.add_patch(box)
            
            # 작업 이름
            ax.text(x, y + 0.1, entry['operation_name'],
                   ha='center', va='center', fontsize=9, fontweight='bold',
                   wrap=True)
            
            # 데이터 크기
            rows_after = entry['data_after']['rows']
            cols_after = entry['data_after']['columns']
            ax.text(x, y - 0.15, f"{rows_after} × {cols_after}",
                   ha='center', va='center', fontsize=8, color='#666')
            
            # 화살표 (다음 작업으로)
            if i < n_operations - 1:
                ax.annotate('',
                          xy=(x + 0.6, y),
                          xytext=(x + 0.4, y),
                          arrowprops=dict(arrowstyle='->', lw=2, color='#333'))
        
        # 축 설정
        ax.set_xlim(-0.5, n_operations - 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.axis('off')
        
        plt.title('Data Transformation Flow', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Flow diagram saved to {save_path}")
        
        plt.show()
    
    def plot_metrics_timeline(self, save_path: str = None):
        """
        메트릭 타임라인 차트
        Plot metrics timeline
        """
        if not self.history:
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # 데이터 추출
        timestamps = [i for i in range(len(self.history))]
        operation_names = [e['operation_name'] for e in self.history]
        rows = [e['data_after']['rows'] for e in self.history]
        columns = [e['data_after']['columns'] for e in self.history]
        missing_rates = [e['data_after']['missing_rate'] for e in self.history]
        
        # 1. 행 개수 변화
        ax = axes[0]
        ax.plot(timestamps, rows, marker='o', linewidth=2, markersize=8, color='#2196F3')
        ax.fill_between(timestamps, rows, alpha=0.3, color='#2196F3')
        ax.set_ylabel('Number of Rows', fontsize=12, fontweight='bold')
        ax.set_title('Data Volume Over Operations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(timestamps)
        ax.set_xticklabels([])
        
        # 2. 컬럼 개수 변화
        ax = axes[1]
        ax.plot(timestamps, columns, marker='s', linewidth=2, markersize=8, color='#4CAF50')
        ax.fill_between(timestamps, columns, alpha=0.3, color='#4CAF50')
        ax.set_ylabel('Number of Columns', fontsize=12, fontweight='bold')
        ax.set_title('Feature Count Over Operations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(timestamps)
        ax.set_xticklabels([])
        
        # 3. 결측률 변화
        ax = axes[2]
        ax.plot(timestamps, missing_rates, marker='^', linewidth=2, markersize=8, color='#FF9800')
        ax.fill_between(timestamps, missing_rates, alpha=0.3, color='#FF9800')
        ax.set_ylabel('Missing Rate (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Operations', fontsize=12, fontweight='bold')
        ax.set_title('Data Completeness Over Operations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(timestamps)
        ax.set_xticklabels(operation_names, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Timeline chart saved to {save_path}")
        
        plt.show()
    
    def plot_lineage_graph(self, save_path: str = None):
        """
        NetworkX를 사용한 리니지 그래프
        Plot lineage graph using NetworkX
        """
        if not self.history:
            return
        
        # 그래프 생성
        G = nx.DiGraph()
        
        # 노드 및 엣지 추가
        for i, entry in enumerate(self.history):
            node_id = f"op_{i}"
            G.add_node(
                node_id,
                label=entry['operation_name'],
                rows=entry['data_after']['rows'],
                cols=entry['data_after']['columns']
            )
            
            # 이전 작업과 연결
            if i > 0:
                prev_node_id = f"op_{i-1}"
                G.add_edge(prev_node_id, node_id)
        
        # 레이아웃
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # 그리기
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # 노드
        nx.draw_networkx_nodes(
            G, pos, node_size=3000, node_color='#E3F2FD',
            edgecolors='#2196F3', linewidths=2, ax=ax
        )
        
        # 엣지
        nx.draw_networkx_edges(
            G, pos, edge_color='#666', arrows=True,
            arrowsize=20, arrowstyle='->', width=2, ax=ax
        )
        
        # 레이블
        labels = {node: G.nodes[node]['label'] for node in G.nodes()}
        nx.draw_networkx_labels(
            G, pos, labels, font_size=10, font_weight='bold', ax=ax
        )
        
        # 데이터 크기 표시
        for node in G.nodes():
            x, y = pos[node]
            rows = G.nodes[node]['rows']
            cols = G.nodes[node]['cols']
            ax.text(x, y - 0.15, f"{rows} × {cols}",
                   ha='center', fontsize=8, color='#666')
        
        ax.axis('off')
        plt.title('Data Lineage Dependency Graph', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Lineage graph saved to {save_path}")
        
        plt.show()


# 사용 예시
def visualize_lineage_demo(lineage):
    """
    리니지 시각화 데모
    """
    visualizer = LineageVisualizer(lineage)
    
    print("Creating lineage visualizations...")
    
    # 1. 데이터 흐름 다이어그램
    print("\n1. Data flow diagram...")
    visualizer.plot_data_flow(save_path='lineage_flow.png')
    
    # 2. 메트릭 타임라인
    print("\n2. Metrics timeline...")
    visualizer.plot_metrics_timeline(save_path='lineage_timeline.png')
    
    # 3. 리니지 그래프
    print("\n3. Lineage dependency graph...")
    visualizer.plot_lineage_graph(save_path='lineage_graph.png')
    
    print("\n✅ All visualizations created successfully!")
```

### 3.3 재현 가능 스크립트 생성

```python
class ReproducibleScriptGenerator:
    """
    재현 가능한 Python 스크립트 생성
    Generate reproducible Python scripts from lineage
    """
    
    def __init__(self, lineage: DataLineage):
        self.lineage = lineage
        self.history = lineage.history
    
    def generate_script(
        self,
        output_filepath: str = 'reproduce_cleansing.py',
        include_comments: bool = True
    ) -> str:
        """
        재현 스크립트 생성
        Generate reproduction script
        
        Parameters:
        -----------
        output_filepath : str
            출력 파일 경로
        include_comments : bool
            주석 포함 여부
            
        Returns:
        --------
        script : str
            생성된 스크립트 내용
        """
        script_lines = []
        
        # 헤더
        script_lines.append("#!/usr/bin/env python3")
        script_lines.append('"""')
        script_lines.append(f"Data Cleansing Reproduction Script")
        script_lines.append(f"Project: {self.lineage.project_name}")
        script_lines.append(f"Dataset: {self.lineage.dataset_name}")
        script_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        script_lines.append('"""')
        script_lines.append("")
        
        # Imports
        script_lines.append("import pandas as pd")
        script_lines.append("import numpy as np")
        script_lines.append("from sklearn.impute import SimpleImputer, KNNImputer")
        script_lines.append("")
        
        # Main function
        script_lines.append("def reproduce_cleansing(input_filepath):")
        script_lines.append('    """Reproduce data cleansing process"""')
        script_lines.append("")
        
        # 각 작업을 코드로 변환
        for i, entry in enumerate(self.history):
            if entry['operation_type'] == 'initial_load':
                continue  # 초기 로드는 스킵
            
            if include_comments:
                script_lines.append(f"    # Operation {i}: {entry['operation_name']}")
                script_lines.append(f"    # {entry['description']}")
            
            # 작업 타입별 코드 생성
            code = self._generate_code_for_operation(entry)
            script_lines.extend(["    " + line for line in code.split("\n")])
            script_lines.append("")
        
        script_lines.append("    return df")
        script_lines.append("")
        
        # Main block
        script_lines.append("if __name__ == '__main__':")
        script_lines.append("    import sys")
        script_lines.append("    ")
        script_lines.append("    if len(sys.argv) < 2:")
        script_lines.append("        print('Usage: python reproduce_cleansing.py <input_file>')")
        script_lines.append("        sys.exit(1)")
        script_lines.append("    ")
        script_lines.append("    input_file = sys.argv[1]")
        script_lines.append("    df = pd.read_csv(input_file)")
        script_lines.append("    ")
        script_lines.append("    df_cleaned = reproduce_cleansing(df)")
        script_lines.append("    ")
        script_lines.append("    df_cleaned.to_csv('cleaned_output.csv', index=False)")
        script_lines.append("    print('✅ Data cleansing completed!')")
        
        # 스크립트 조합
        script = "\n".join(script_lines)
        
        # 파일 저장
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(script)
        
        print(f"✅ Reproducible script generated: {output_filepath}")
        
        return script
    
    def _generate_code_for_operation(self, entry: Dict[str, Any]) -> str:
        """
        작업 타입별 코드 생성
        Generate code for specific operation type
        """
        op_type = entry['operation_type']
        params = entry['parameters']
        
        if op_type == 'impute_missing':
            method = params.get('method', 'mean')
            column = params.get('column', 'all')
            
            if method == 'mean':
                code = f"df['{column}'].fillna(df['{column}'].mean(), inplace=True)"
            elif method == 'median':
                code = f"df['{column}'].fillna(df['{column}'].median(), inplace=True)"
            elif method == 'mode':
                code = f"df['{column}'].fillna(df['{column}'].mode()[0], inplace=True)"
            elif method == 'knn':
                n_neighbors = params.get('n_neighbors', 5)
                code = f"""imputer = KNNImputer(n_neighbors={n_neighbors})
df[['{column}']] = imputer.fit_transform(df[['{column}']]) """
            else:
                code = f"# Custom imputation method: {method}"
        
        elif op_type == 'remove_outliers':
            method = params.get('method', 'IQR')
            
            if method == 'IQR':
                multiplier = params.get('multiplier', 1.5)
                code = f"""Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - {multiplier} * IQR)) | (df > (Q3 + {multiplier} * IQR))).any(axis=1)]"""
            else:
                code = f"# Custom outlier removal method: {method}"
        
        elif op_type == 'remove_duplicates':
            subset = params.get('subset')
            keep = params.get('keep', 'first')
            
            if subset:
                code = f"df.drop_duplicates(subset={subset}, keep='{keep}', inplace=True)"
            else:
                code = f"df.drop_duplicates(keep='{keep}', inplace=True)"
        
        elif op_type == 'feature_engineering':
            new_column = params.get('new_column')
            formula = params.get('formula')
            
            if new_column and formula:
                code = f"df['{new_column}'] = {formula}"
            else:
                code = "# Feature engineering operation"
        
        else:
            code = f"# Operation: {op_type}\n# Parameters: {params}"
        
        return code


# 사용 예시
def generate_reproduction_script_demo(lineage):
    """
    재현 스크립트 생성 데모
    """
    generator = ReproducibleScriptGenerator(lineage)
    
    script = generator.generate_script(
        output_filepath='reproduce_cleansing.py',
        include_comments=True
    )
    
    print("\n" + "="*80)
    print("GENERATED SCRIPT PREVIEW")
    print("="*80)
    print(script[:1000] + "\n...\n")
    
    return script
```

---

## 4. 예시 (Examples)

### 4.1 전체 리니지 추적 예시

```python
# 실제 사용 예시: E-commerce 주문 데이터 클렌징

import pandas as pd
import numpy as np
import time

def complete_lineage_example():
    """
    완전한 리니지 추적 예시
    """
    print("="*80)
    print("COMPLETE DATA LINEAGE TRACKING EXAMPLE")
    print("="*80)
    
    # 1. 원시 데이터 생성
    print("\n1. Loading raw data...")
    np.random.seed(42)
    n = 5000
    
    df_raw = pd.DataFrame({
        'order_id': range(1, n + 1),
        'customer_id': np.random.randint(1, 1001, n),
        'product_id': np.random.randint(1, 201, n),
        'quantity': np.random.randint(1, 10, n),
        'unit_price': np.random.uniform(10, 500, n).round(2),
        'order_date': pd.date_range('2024-01-01', periods=n, freq='H')
    })
    
    # 품질 이슈 삽입
    df_raw.loc[df_raw.sample(frac=0.15).index, 'quantity'] = np.nan
    df_raw.loc[df_raw.sample(frac=0.10).index, 'unit_price'] = np.nan
    df_raw = pd.concat([df_raw, df_raw.sample(frac=0.05)], ignore_index=True)
    
    print(f"   Loaded {len(df_raw)} rows with quality issues")
    
    # 2. 리니지 추적 시작
    print("\n2. Initializing lineage tracking...")
    lineage = DataLineage(
        dataset_name='ecommerce_orders',
        initial_df=df_raw,
        project_name='E-commerce Data Cleansing'
    )
    
    # 3. 결측값 대체
    print("\n3. Imputing missing values...")
    start_time = time.time()
    df_step1 = df_raw.copy()
    df_step1['quantity'].fillna(df_step1['quantity'].median(), inplace=True)
    df_step1['unit_price'].fillna(df_step1['unit_price'].mean(), inplace=True)
    execution_time = time.time() - start_time
    
    lineage.log_operation(
        operation_type='impute_missing',
        operation_name='Median/Mean Imputation',
        df_after=df_step1,
        parameters={
            'quantity_method': 'median',
            'unit_price_method': 'mean'
        },
        description='Imputed missing quantity with median and unit_price with mean',
        user='data_engineer',
        execution_time=execution_time
    )
    
    # 4. 이상치 제거
    print("\n4. Removing outliers...")
    start_time = time.time()
    Q1 = df_step1['unit_price'].quantile(0.25)
    Q3 = df_step1['unit_price'].quantile(0.75)
    IQR = Q3 - Q1
    df_step2 = df_step1[
        (df_step1['unit_price'] >= Q1 - 1.5 * IQR) &
        (df_step1['unit_price'] <= Q3 + 1.5 * IQR)
    ]
    execution_time = time.time() - start_time
    
    lineage.log_operation(
        operation_type='remove_outliers',
        operation_name='IQR Outlier Removal',
        df_after=df_step2,
        parameters={
            'method': 'IQR',
            'column': 'unit_price',
            'multiplier': 1.5
        },
        description='Removed outliers in unit_price using IQR method',
        user='data_engineer',
        execution_time=execution_time
    )
    
    # 5. 중복 제거
    print("\n5. Removing duplicates...")
    start_time = time.time()
    df_step3 = df_step2.drop_duplicates()
    execution_time = time.time() - start_time
    
    lineage.log_operation(
        operation_type='remove_duplicates',
        operation_name='Remove Duplicate Rows',
        df_after=df_step3,
        parameters={'keep': 'first'},
        description='Removed exact duplicate rows',
        user='data_engineer',
        execution_time=execution_time
    )
    
    # 6. 피처 엔지니어링
    print("\n6. Adding new features...")
    start_time = time.time()
    df_step4 = df_step3.copy()
    df_step4['total_amount'] = (df_step4['quantity'] * df_step4['unit_price']).round(2)
    df_step4['order_month'] = df_step4['order_date'].dt.month
    execution_time = time.time() - start_time
    
    lineage.log_operation(
        operation_type='feature_engineering',
        operation_name='Add Calculated Features',
        df_after=df_step4,
        parameters={
            'new_features': ['total_amount', 'order_month']
        },
        description='Added total_amount and order_month features',
        user='data_engineer',
        execution_time=execution_time
    )
    
    # 7. 리니지 리포트
    print("\n" + "="*80)
    print("LINEAGE REPORT")
    print("="*80)
    report = lineage.get_lineage_report()
    print(report.to_string())
    
    # 8. 요약 정보
    print("\n" + "="*80)
    print("LINEAGE SUMMARY")
    print("="*80)
    summary = lineage.get_summary()
    for key, value in summary.items():
        if key != 'operations_list':
            print(f"{key}: {value}")
    
    # 9. 리니지 내보내기
    print("\n" + "="*80)
    print("EXPORTING LINEAGE")
    print("="*80)
    lineage.export_lineage('ecommerce_lineage.json')
    
    # 10. 시각화 생성
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    visualizer = LineageVisualizer(lineage)
    visualizer.plot_data_flow(save_path='ecommerce_flow.png')
    visualizer.plot_metrics_timeline(save_path='ecommerce_timeline.png')
    visualizer.plot_lineage_graph(save_path='ecommerce_graph.png')
    
    # 11. 재현 스크립트 생성
    print("\n" + "="*80)
    print("GENERATING REPRODUCTION SCRIPT")
    print("="*80)
    generator = ReproducibleScriptGenerator(lineage)
    script = generator.generate_script(
        output_filepath='reproduce_ecommerce_cleansing.py',
        include_comments=True
    )
    
    print("\n✅ Complete lineage tracking example finished!")
    print(f"📁 Generated files:")
    print("   - ecommerce_lineage.json")
    print("   - ecommerce_flow.png")
    print("   - ecommerce_timeline.png")
    print("   - ecommerce_graph.png")
    print("   - reproduce_ecommerce_cleansing.py")
    
    return lineage


# 실행
if __name__ == "__main__":
    lineage = complete_lineage_example()
```

---

## 5. 에이전트 매핑 (Agent Mapping)

### 5.1 Primary Agent

**`data-cleaning-specialist`**
- 역할: 데이터 리니지 추적 및 관리
- 책임:
  - 모든 변환 작업 기록
  - 리니지 데이터 저장
  - 재현 스크립트 생성

### 5.2 Supporting Agents

**`technical-documentation-writer`**
- 역할: 리니지 문서화
- 책임:
  - 리니지 리포트 작성
  - 변환 이력 설명
  - 사용자 가이드 생성

**`data-visualization-specialist`**
- 역할: 리니지 시각화
- 책임:
  - 데이터 흐름 다이어그램
  - 의존성 그래프
  - 타임라인 차트

---

## 6. 필요 라이브러리 (Required Libraries)

```bash
# 필수 라이브러리
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install matplotlib>=3.7.0
pip install networkx>=3.0

# 선택 라이브러리
pip install graphviz>=0.20.0
pip install pydot>=1.4.0
```

---

## 7. 체크포인트 (Checkpoints)

### 7.1 리니지 추적 체크리스트

- [ ] 모든 변환 작업 기록
- [ ] 파라미터 문서화
- [ ] 실행 시간 측정
- [ ] 데이터 해시 계산
- [ ] 에러 처리 포함

---

## 8. 트러블슈팅 (Troubleshooting)

**문제: 대용량 데이터 해시 계산 느림**
```python
# 해결: 샘플링 사용
def quick_hash(df, sample_size=1000):
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df
    return hashlib.md5(pickle.dumps(df_sample)).hexdigest()
```

---

## 9. 참고 자료 (References)

- NetworkX: https://networkx.org/
- Data Lineage Best Practices: https://www.dataversity.net/data-lineage-best-practices/
- Related: `11-data-validation.md`, `12-quality-reporting.md`

---

**작성자**: Claude Code  
**최종 수정일**: 2025-01-26  
**버전**: 1.0
