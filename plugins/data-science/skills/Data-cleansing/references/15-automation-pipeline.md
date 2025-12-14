# 15. Automation Pipeline (자동화 파이프라인)

**생성일**: 2025-01-26  
**버전**: 1.0  
**카테고리**: Data Pipeline & Automation

---

## 1. 개요 (Overview)

### 1.1 목적 (Purpose)

데이터 클렌징 자동화 파이프라인은 반복적인 데이터 품질 작업을 자동화하고, 에이전트 오케스트레이션을 통해 효율적인 워크플로우를 구축합니다. 이 레퍼런스는 완전 자동화된 데이터 클렌징 파이프라인 설계 및 구현 방법을 제공합니다.

### 1.2 적용 시기 (When to Apply)

**필수 적용**:
- ✅ 반복적인 데이터 클렌징 작업
- ✅ 프로덕션 데이터 파이프라인
- ✅ 대용량 데이터 처리
- ✅ 실시간 데이터 품질 모니터링

**권장 적용**:
- 🔹 정기적인 데이터 처리 (일일, 주간, 월간)
- 🔹 여러 데이터 소스 통합
- 🔹 팀 협업 프로젝트
- 🔹 CI/CD 통합

### 1.3 파이프라인 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Cleansing Pipeline                     │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Data Sources │ ───> │   Ingestion  │ ───> │  Validation  │
│              │      │   (Extract)  │      │   (Quality)  │
│ - Files      │      │              │      │              │
│ - Databases  │      │ • Read data  │      │ • Profile    │
│ - APIs       │      │ • Parse      │      │ • Validate   │
└──────────────┘      └──────────────┘      └──────────────┘
                                                     │
                                                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Storage    │ <─── │  Transform   │ <─── │   Cleanse    │
│   (Load)     │      │  (Features)  │      │   (Fix)      │
│              │      │              │      │              │
│ • Save       │      │ • Normalize  │      │ • Missing    │
│ • Index      │      │ • Encode     │      │ • Outliers   │
│ • Version    │      │ • Aggregate  │      │ • Duplicates │
└──────────────┘      └──────────────┘      └──────────────┘
       │
       ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Monitoring  │      │   Alerting   │      │   Reporting  │
│              │      │              │      │              │
│ • Metrics    │      │ • Email      │      │ • Dashboard  │
│ • Logs       │      │ • Slack      │      │ • Reports    │
│ • Health     │      │ • PagerDuty  │      │ • Lineage    │
└──────────────┘      └──────────────┘      └──────────────┘
```

---

## 2. 이론적 배경 (Theoretical Background)

### 2.1 파이프라인 설계 원칙

**1. Modularity (모듈성)**
- 각 단계를 독립적인 모듈로 설계
- 재사용 가능한 컴포넌트
- 쉬운 유지보수

**2. Scalability (확장성)**
- 병렬 처리 지원
- 대용량 데이터 처리
- 수평적 확장 가능

**3. Reliability (안정성)**
- 에러 핸들링
- 재시도 메커니즘
- 롤백 기능

**4. Observability (관찰 가능성)**
- 로깅 및 모니터링
- 메트릭 수집
- 알람 시스템

**5. Reproducibility (재현성)**
- 버전 관리
- 리니지 추적
- 감사 로그

### 2.2 워크플로우 오케스트레이션

**오케스트레이션 도구**:
- **Airflow**: Apache의 워크플로우 관리 플랫폼
- **Prefect**: 현대적인 워크플로우 오케스트레이션
- **Luigi**: Spotify의 파이프라인 빌더
- **Dagster**: 데이터 오케스트레이터

---

## 3. 구현 (Implementation)

### 3.1 CleansingPipeline 클래스

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable, Optional
from datetime import datetime
import logging
import time
import traceback
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class CleansingPipeline:
    """
    데이터 클렌징 자동화 파이프라인
    Automated data cleansing pipeline
    """
    
    def __init__(
        self,
        pipeline_name: str,
        config: Dict[str, Any] = None
    ):
        """
        Parameters:
        -----------
        pipeline_name : str
            파이프라인 이름
        config : dict, optional
            파이프라인 설정
        """
        self.pipeline_name = pipeline_name
        self.config = config or {}
        self.logger = logging.getLogger(pipeline_name)
        
        # 실행 상태
        self.execution_id = None
        self.start_time = None
        self.end_time = None
        self.status = 'initialized'
        
        # 파이프라인 단계
        self.stages = []
        
        # 메트릭
        self.metrics = {}
        
        # 에러
        self.errors = []
    
    def add_stage(
        self,
        stage_name: str,
        stage_func: Callable,
        enabled: bool = True,
        retry_count: int = 0,
        retry_delay: int = 5
    ):
        """
        파이프라인 단계 추가
        Add a pipeline stage
        
        Parameters:
        -----------
        stage_name : str
            단계 이름
        stage_func : Callable
            실행할 함수 (df를 받아서 df 반환)
        enabled : bool
            단계 활성화 여부
        retry_count : int
            재시도 횟수
        retry_delay : int
            재시도 간 대기 시간 (초)
        """
        stage = {
            'name': stage_name,
            'func': stage_func,
            'enabled': enabled,
            'retry_count': retry_count,
            'retry_delay': retry_delay,
            'status': 'pending',
            'execution_time': 0.0,
            'error': None
        }
        
        self.stages.append(stage)
        self.logger.info(f"Added stage: {stage_name}")
    
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        파이프라인 실행
        Run the pipeline
        
        Parameters:
        -----------
        df : pd.DataFrame
            입력 데이터
            
        Returns:
        --------
        df_result : pd.DataFrame
            클렌징된 데이터
        """
        self.execution_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.start_time = time.time()
        self.status = 'running'
        
        self.logger.info("="*80)
        self.logger.info(f"Pipeline: {self.pipeline_name}")
        self.logger.info(f"Execution ID: {self.execution_id}")
        self.logger.info(f"Input shape: {df.shape}")
        self.logger.info("="*80)
        
        df_current = df.copy()
        
        # 각 단계 실행
        for i, stage in enumerate(self.stages, 1):
            if not stage['enabled']:
                self.logger.info(f"\n[{i}/{len(self.stages)}] {stage['name']} - SKIPPED")
                stage['status'] = 'skipped'
                continue
            
            self.logger.info(f"\n[{i}/{len(self.stages)}] {stage['name']} - STARTING")
            
            # 단계 실행 (재시도 포함)
            success = False
            attempt = 0
            max_attempts = stage['retry_count'] + 1
            
            while attempt < max_attempts and not success:
                try:
                    attempt += 1
                    if attempt > 1:
                        self.logger.warning(f"Retry attempt {attempt-1}/{stage['retry_count']}")
                        time.sleep(stage['retry_delay'])
                    
                    stage_start = time.time()
                    df_current = stage['func'](df_current)
                    stage_time = time.time() - stage_start
                    
                    stage['execution_time'] = stage_time
                    stage['status'] = 'success'
                    success = True
                    
                    self.logger.info(f"✅ {stage['name']} - COMPLETED ({stage_time:.2f}s)")
                    self.logger.info(f"   Output shape: {df_current.shape}")
                    
                except Exception as e:
                    error_msg = f"Error in {stage['name']}: {str(e)}"
                    self.logger.error(error_msg)
                    self.logger.error(traceback.format_exc())
                    
                    stage['error'] = error_msg
                    self.errors.append({
                        'stage': stage['name'],
                        'attempt': attempt,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    })
                    
                    if attempt >= max_attempts:
                        stage['status'] = 'failed'
                        
                        # 실패 시 처리 방법
                        if self.config.get('fail_fast', True):
                            self.status = 'failed'
                            raise RuntimeError(f"Pipeline failed at stage: {stage['name']}")
                        else:
                            self.logger.warning(f"Continuing despite failure in {stage['name']}")
        
        # 완료
        self.end_time = time.time()
        self.status = 'completed'
        
        total_time = self.end_time - self.start_time
        
        self.logger.info("\n" + "="*80)
        self.logger.info("PIPELINE COMPLETED")
        self.logger.info("="*80)
        self.logger.info(f"Total time: {total_time:.2f}s")
        self.logger.info(f"Final shape: {df_current.shape}")
        
        # 메트릭 저장
        self.metrics = {
            'execution_id': self.execution_id,
            'total_time': total_time,
            'input_rows': len(df),
            'output_rows': len(df_current),
            'input_columns': len(df.columns),
            'output_columns': len(df_current.columns),
            'rows_removed': len(df) - len(df_current),
            'stages_total': len(self.stages),
            'stages_success': sum(1 for s in self.stages if s['status'] == 'success'),
            'stages_failed': sum(1 for s in self.stages if s['status'] == 'failed'),
            'stages_skipped': sum(1 for s in self.stages if s['status'] == 'skipped')
        }
        
        return df_current
    
    def get_execution_report(self) -> Dict[str, Any]:
        """
        실행 리포트 생성
        Generate execution report
        """
        report = {
            'pipeline_name': self.pipeline_name,
            'execution_id': self.execution_id,
            'status': self.status,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            'end_time': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            'metrics': self.metrics,
            'stages': []
        }
        
        for stage in self.stages:
            report['stages'].append({
                'name': stage['name'],
                'status': stage['status'],
                'execution_time': stage['execution_time'],
                'error': stage['error']
            })
        
        return report
    
    def save_execution_report(self, filepath: str):
        """
        실행 리포트 저장
        Save execution report
        """
        import json
        
        report = self.get_execution_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Execution report saved to {filepath}")


# 사용 예시: 기본 파이프라인 생성
def create_basic_cleansing_pipeline():
    """
    기본 데이터 클렌징 파이프라인 생성
    Create basic data cleansing pipeline
    """
    pipeline = CleansingPipeline(
        pipeline_name='basic_cleansing',
        config={'fail_fast': False}
    )
    
    # Stage 1: 데이터 프로파일링
    def stage_profiling(df):
        print(f"\n📊 Data Profiling:")
        print(f"   Shape: {df.shape}")
        print(f"   Missing: {df.isnull().sum().sum()}")
        print(f"   Duplicates: {df.duplicated().sum()}")
        return df
    
    pipeline.add_stage('data_profiling', stage_profiling, enabled=True)
    
    # Stage 2: 결측값 대체
    def stage_imputation(df):
        df_clean = df.copy()
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        return df_clean
    
    pipeline.add_stage('missing_imputation', stage_imputation, enabled=True)
    
    # Stage 3: 이상치 제거
    def stage_outliers(df):
        df_clean = df.copy()
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        return df_clean
    
    pipeline.add_stage('outlier_removal', stage_outliers, enabled=True, retry_count=2)
    
    # Stage 4: 중복 제거
    def stage_deduplication(df):
        return df.drop_duplicates()
    
    pipeline.add_stage('deduplication', stage_deduplication, enabled=True)
    
    return pipeline
```

### 3.2 에이전트 오케스트레이터

```python
from enum import Enum
from dataclasses import dataclass
from typing import List

class AgentType(Enum):
    """에이전트 유형"""
    DATA_CLEANING_SPECIALIST = "data-cleaning-specialist"
    DATA_SCIENTIST = "data-scientist"
    DATA_VISUALIZATION_SPECIALIST = "data-visualization-specialist"
    FEATURE_ENGINEERING_SPECIALIST = "feature-engineering-specialist"
    TECHNICAL_DOCUMENTATION_WRITER = "technical-documentation-writer"


@dataclass
class Task:
    """작업 정의"""
    task_id: str
    task_type: str
    agent: AgentType
    parameters: Dict[str, Any]
    dependencies: List[str] = None
    status: str = 'pending'
    result: Any = None


class AgentOrchestrator:
    """
    에이전트 오케스트레이터
    Orchestrate multiple agents for data cleansing
    """
    
    def __init__(self):
        self.tasks = []
        self.completed_tasks = set()
        self.logger = logging.getLogger('AgentOrchestrator')
    
    def add_task(
        self,
        task_id: str,
        task_type: str,
        agent: AgentType,
        parameters: Dict[str, Any],
        dependencies: List[str] = None
    ):
        """
        작업 추가
        Add a task
        """
        task = Task(
            task_id=task_id,
            task_type=task_type,
            agent=agent,
            parameters=parameters,
            dependencies=dependencies or []
        )
        
        self.tasks.append(task)
        self.logger.info(f"Added task: {task_id} (Agent: {agent.value})")
    
    def can_execute(self, task: Task) -> bool:
        """
        작업 실행 가능 여부 확인
        Check if task can be executed
        """
        if not task.dependencies:
            return True
        
        return all(dep in self.completed_tasks for dep in task.dependencies)
    
    def execute_task(self, task: Task, df: pd.DataFrame) -> pd.DataFrame:
        """
        작업 실행
        Execute a task
        """
        self.logger.info(f"\n🤖 Executing task: {task.task_id}")
        self.logger.info(f"   Agent: {task.agent.value}")
        self.logger.info(f"   Type: {task.task_type}")
        
        # 에이전트별 작업 실행 (시뮬레이션)
        if task.agent == AgentType.DATA_CLEANING_SPECIALIST:
            df_result = self._execute_cleaning_task(df, task)
        elif task.agent == AgentType.DATA_SCIENTIST:
            df_result = self._execute_analysis_task(df, task)
        elif task.agent == AgentType.FEATURE_ENGINEERING_SPECIALIST:
            df_result = self._execute_feature_task(df, task)
        else:
            df_result = df  # 기본: 데이터 그대로 반환
        
        task.status = 'completed'
        task.result = {'success': True}
        self.completed_tasks.add(task.task_id)
        
        self.logger.info(f"✅ Task completed: {task.task_id}")
        
        return df_result
    
    def _execute_cleaning_task(self, df: pd.DataFrame, task: Task) -> pd.DataFrame:
        """
        데이터 클렌징 작업 실행
        Execute data cleaning task
        """
        task_type = task.task_type
        params = task.parameters
        
        if task_type == 'impute_missing':
            method = params.get('method', 'mean')
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].isnull().any():
                    if method == 'mean':
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif method == 'median':
                        df[col].fillna(df[col].median(), inplace=True)
        
        elif task_type == 'remove_outliers':
            method = params.get('method', 'IQR')
            if method == 'IQR':
                for col in df.select_dtypes(include=[np.number]).columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
        
        elif task_type == 'remove_duplicates':
            df = df.drop_duplicates()
        
        return df
    
    def _execute_analysis_task(self, df: pd.DataFrame, task: Task) -> pd.DataFrame:
        """
        데이터 분석 작업 실행
        Execute data analysis task
        """
        # 분석만 수행, 데이터는 변경 없음
        self.logger.info(f"   Analysis: {task.task_type}")
        return df
    
    def _execute_feature_task(self, df: pd.DataFrame, task: Task) -> pd.DataFrame:
        """
        피처 엔지니어링 작업 실행
        Execute feature engineering task
        """
        params = task.parameters
        
        if task.task_type == 'create_feature':
            new_col = params.get('column_name')
            formula = params.get('formula')
            if new_col and formula:
                df[new_col] = eval(formula, {'df': df})
        
        return df
    
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 작업 실행 (의존성 순서 고려)
        Run all tasks respecting dependencies
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("AGENT ORCHESTRATION STARTING")
        self.logger.info("="*80)
        
        df_current = df.copy()
        pending_tasks = [t for t in self.tasks if t.status == 'pending']
        
        while pending_tasks:
            executed_any = False
            
            for task in pending_tasks:
                if self.can_execute(task):
                    df_current = self.execute_task(task, df_current)
                    executed_any = True
            
            if not executed_any:
                # 순환 의존성 또는 미해결 의존성
                remaining = [t.task_id for t in pending_tasks]
                self.logger.error(f"Cannot execute remaining tasks: {remaining}")
                break
            
            pending_tasks = [t for t in self.tasks if t.status == 'pending']
        
        self.logger.info("\n" + "="*80)
        self.logger.info("AGENT ORCHESTRATION COMPLETED")
        self.logger.info("="*80)
        self.logger.info(f"Total tasks: {len(self.tasks)}")
        self.logger.info(f"Completed: {len(self.completed_tasks)}")
        
        return df_current


# 사용 예시: 에이전트 오케스트레이션
def demo_agent_orchestration():
    """
    에이전트 오케스트레이션 데모
    """
    orchestrator = AgentOrchestrator()
    
    # Task 1: 데이터 프로파일링 (의존성 없음)
    orchestrator.add_task(
        task_id='task_001',
        task_type='profile_data',
        agent=AgentType.DATA_SCIENTIST,
        parameters={}
    )
    
    # Task 2: 결측값 대체 (Task 1 완료 후)
    orchestrator.add_task(
        task_id='task_002',
        task_type='impute_missing',
        agent=AgentType.DATA_CLEANING_SPECIALIST,
        parameters={'method': 'median'},
        dependencies=['task_001']
    )
    
    # Task 3: 이상치 제거 (Task 2 완료 후)
    orchestrator.add_task(
        task_id='task_003',
        task_type='remove_outliers',
        agent=AgentType.DATA_CLEANING_SPECIALIST,
        parameters={'method': 'IQR'},
        dependencies=['task_002']
    )
    
    # Task 4: 중복 제거 (Task 3 완료 후)
    orchestrator.add_task(
        task_id='task_004',
        task_type='remove_duplicates',
        agent=AgentType.DATA_CLEANING_SPECIALIST,
        parameters={},
        dependencies=['task_003']
    )
    
    # Task 5: 피처 생성 (Task 4 완료 후)
    orchestrator.add_task(
        task_id='task_005',
        task_type='create_feature',
        agent=AgentType.FEATURE_ENGINEERING_SPECIALIST,
        parameters={
            'column_name': 'total_amount',
            'formula': "df['quantity'] * df['unit_price']"
        },
        dependencies=['task_004']
    )
    
    return orchestrator
```

### 3.3 CLI 커맨드 인터페이스

```python
import click
import json

@click.group()
def cli():
    """Data Cleansing Pipeline CLI"""
    pass


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', default='cleaned_output.csv', help='Output file path')
@click.option('--report', '-r', is_flag=True, help='Generate execution report')
def clean_full(input_file, output, report):
    """
    완전 자동 클렌징 파이프라인
    Full automatic cleansing pipeline
    
    Usage: python cleansing_cli.py clean-full data.csv -o cleaned.csv -r
    """
    click.echo(f"🚀 Starting full cleansing pipeline...")
    click.echo(f"   Input: {input_file}")
    click.echo(f"   Output: {output}")
    
    # 데이터 로드
    df = pd.read_csv(input_file)
    click.echo(f"   Loaded: {len(df)} rows")
    
    # 파이프라인 생성 및 실행
    pipeline = create_basic_cleansing_pipeline()
    df_clean = pipeline.run(df)
    
    # 저장
    df_clean.to_csv(output, index=False)
    click.echo(f"✅ Cleaned data saved to {output}")
    
    # 리포트 생성
    if report:
        report_file = output.replace('.csv', '_report.json')
        pipeline.save_execution_report(report_file)
        click.echo(f"📄 Report saved to {report_file}")


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--method', '-m', default='median', help='Imputation method (mean/median/mode)')
@click.option('--output', '-o', default='imputed_output.csv', help='Output file path')
def clean_missing(input_file, method, output):
    """
    결측값 집중 처리
    Focus on missing value imputation
    
    Usage: python cleansing_cli.py clean-missing data.csv -m median -o imputed.csv
    """
    click.echo(f"🔧 Imputing missing values using {method} method...")
    
    df = pd.read_csv(input_file)
    
    # 결측값 대체
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            if method == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif method == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif method == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    df.to_csv(output, index=False)
    click.echo(f"✅ Imputed data saved to {output}")


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--method', '-m', default='IQR', help='Outlier detection method (IQR/zscore)')
@click.option('--output', '-o', default='no_outliers_output.csv', help='Output file path')
def clean_outliers(input_file, method, output):
    """
    이상치 탐지 및 처리
    Detect and handle outliers
    
    Usage: python cleansing_cli.py clean-outliers data.csv -m IQR -o cleaned.csv
    """
    click.echo(f"🎯 Removing outliers using {method} method...")
    
    df = pd.read_csv(input_file)
    
    if method == 'IQR':
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
    
    df.to_csv(output, index=False)
    click.echo(f"✅ Data without outliers saved to {output}")


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', default='custom_output.csv', help='Output file path')
def clean_custom(config_file, input_file, output):
    """
    커스텀 설정 기반 클렌징
    Custom configuration-based cleansing
    
    Usage: python cleansing_cli.py clean-custom config.json data.csv -o cleaned.csv
    """
    click.echo(f"⚙️ Running custom cleansing with config: {config_file}")
    
    # 설정 로드
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    df = pd.read_csv(input_file)
    
    # 설정에 따라 파이프라인 구성
    pipeline = CleansingPipeline('custom_pipeline', config=config)
    
    # 동적으로 스테이지 추가
    # (실제로는 config에서 읽어서 추가)
    
    df_clean = pipeline.run(df)
    df_clean.to_csv(output, index=False)
    
    click.echo(f"✅ Custom cleaned data saved to {output}")


if __name__ == '__main__':
    cli()
```

---

## 4. 예시 (Examples)

### 4.1 완전한 자동화 파이프라인 예시

```python
def complete_automation_example():
    """
    완전한 자동화 파이프라인 예시
    Complete automation pipeline example
    """
    print("="*80)
    print("COMPLETE AUTOMATION PIPELINE EXAMPLE")
    print("="*80)
    
    # 1. 샘플 데이터 생성
    print("\n1. Creating sample data...")
    np.random.seed(42)
    n = 5000
    
    df = pd.DataFrame({
        'order_id': range(1, n + 1),
        'customer_id': np.random.randint(1, 1001, n),
        'quantity': np.random.randint(1, 10, n),
        'unit_price': np.random.uniform(10, 500, n).round(2),
        'order_date': pd.date_range('2024-01-01', periods=n, freq='H')
    })
    
    # 품질 이슈 삽입
    df.loc[df.sample(frac=0.15).index, 'quantity'] = np.nan
    df.loc[df.sample(frac=0.10).index, 'unit_price'] = np.nan
    df = pd.concat([df, df.sample(frac=0.05)], ignore_index=True)
    
    print(f"   Created {len(df)} rows with quality issues")
    
    # 2. 파이프라인 생성
    print("\n2. Creating cleansing pipeline...")
    pipeline = create_basic_cleansing_pipeline()
    
    # 3. 파이프라인 실행
    print("\n3. Running pipeline...")
    df_clean = pipeline.run(df)
    
    # 4. 결과 저장
    print("\n4. Saving results...")
    output_dir = Path('pipeline_output')
    output_dir.mkdir(exist_ok=True)
    
    df_clean.to_csv(output_dir / 'cleaned_data.csv', index=False)
    pipeline.save_execution_report(str(output_dir / 'execution_report.json'))
    
    # 5. 메트릭 출력
    print("\n5. Pipeline metrics:")
    for key, value in pipeline.metrics.items():
        print(f"   {key}: {value}")
    
    print(f"\n✅ Pipeline completed successfully!")
    print(f"📁 Output directory: {output_dir}")
    
    return pipeline


# 실행
if __name__ == "__main__":
    pipeline = complete_automation_example()
```

---

## 5. 에이전트 매핑 (Agent Mapping)

### 5.1 Primary Agent

**`data-cleaning-specialist`**
- 파이프라인 설계 및 구현
- 에이전트 오케스트레이션
- 자동화 스크립트 작성

### 5.2 Supporting Agents

**All agents** (작업 수행)
- 각 에이전트는 할당된 작업 실행
- 결과를 다음 단계로 전달

---

## 6. 필요 라이브러리 (Required Libraries)

```bash
# 필수
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install click>=8.1.0

# 워크플로우 오케스트레이션 (선택 1개)
pip install apache-airflow>=2.7.0
pip install prefect>=2.14.0
pip install luigi>=3.4.0

# 병렬 처리
pip install dask>=2023.10.0
pip install joblib>=1.3.0
```

---

## 7. 체크포인트 (Checkpoints)

### 7.1 파이프라인 설계

- [ ] 단계별 작업 정의
- [ ] 의존성 관계 명확화
- [ ] 에러 핸들링 전략
- [ ] 재시도 메커니즘

### 7.2 운영

- [ ] 모니터링 설정
- [ ] 알람 설정
- [ ] 로깅 구성
- [ ] 성능 최적화

---

## 8. 트러블슈팅 (Troubleshooting)

**문제: 파이프라인 실행 중 중단**
```python
# 해결: 체크포인트 추가
def save_checkpoint(df, stage_name):
    df.to_parquet(f'checkpoint_{stage_name}.parquet')

# 재시작 시 체크포인트에서 로드
if os.path.exists(checkpoint_file):
    df = pd.read_parquet(checkpoint_file)
```

---

## 9. 참고 자료 (References)

- Apache Airflow: https://airflow.apache.org/
- Prefect: https://www.prefect.io/
- Related: All previous Data-cleansing references

---

**작성자**: Claude Code  
**최종 수정일**: 2025-01-26  
**버전**: 1.0
