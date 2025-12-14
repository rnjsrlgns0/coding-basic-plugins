# 11. Data Validation (데이터 검증)

**생성일**: 2025-01-26  
**버전**: 1.0  
**카테고리**: Data Quality & Validation

---

## 1. 개요 (Overview)

### 1.1 목적 (Purpose)

데이터 검증(Data Validation)은 데이터 클렌징 프로세스의 최종 단계로, 데이터의 일관성(consistency), 정확성(accuracy), 무결성(integrity)을 보장하는 필수 프로세스입니다. 이 레퍼런스는 교차 필드 검증, 참조 무결성 검증, 비즈니스 로직 검증을 자동화하는 방법을 제공합니다.

### 1.2 적용 시기 (When to Apply)

**필수 적용 시점**:
- ✅ 데이터 클렌징 작업 완료 후 최종 검증 단계
- ✅ 프로덕션 환경으로 데이터 배포 전
- ✅ 데이터 통합(integration) 후 일관성 확인
- ✅ ML 모델 학습 전 데이터 품질 확보

**상황별 적용**:
- 🔹 여러 소스에서 데이터 병합 시
- 🔹 외부 데이터를 받았을 때
- 🔹 시계열 데이터의 논리적 순서 확인
- 🔹 비즈니스 규칙이 복잡한 도메인 (금융, 의료 등)

### 1.3 검증 레벨 (Validation Levels)

```
Level 1: Field-Level Validation (개별 필드)
└── Data type, format, range, null check

Level 2: Cross-Field Validation (교차 필드)
└── start_date < end_date, age vs birth_date

Level 3: Record-Level Validation (레코드)
└── Business rules, calculation verification

Level 4: Referential Integrity (참조 무결성)
└── Foreign key relationships, orphan records

Level 5: Dataset-Level Validation (데이터셋)
└── Aggregation checks, distribution validation
```

---

## 2. 이론적 배경 (Theoretical Background)

### 2.1 데이터 검증의 중요성

**데이터 품질 차원 (Data Quality Dimensions)**:
1. **Completeness** (완전성): 결측값이 없는가?
2. **Consistency** (일관성): 데이터 간 모순이 없는가?
3. **Accuracy** (정확성): 데이터가 실제와 일치하는가?
4. **Validity** (유효성): 비즈니스 규칙을 준수하는가?
5. **Timeliness** (적시성): 데이터가 최신인가?
6. **Uniqueness** (유일성): 중복이 없는가?

### 2.2 검증 메커니즘

**Type 1: Constraint-Based Validation**
- 정의된 제약 조건 기반 (범위, 형식, NOT NULL 등)
- 명시적 규칙 (Explicit rules)
- 예: `age BETWEEN 0 AND 120`

**Type 2: Relationship-Based Validation**
- 필드 간 관계 검증
- 계산식 검증
- 예: `total = quantity × unit_price`

**Type 3: Reference-Based Validation**
- 외래 키(Foreign Key) 검증
- 참조 테이블과의 일치성
- 예: `customer_id IN customers.id`

**Type 4: Statistical Validation**
- 통계적 패턴 검증
- 이상 분포 탐지
- 예: `mean(sales) > 0 AND std(sales) < 1000`

### 2.3 검증 시나리오

**시나리오 1: E-commerce 주문 데이터**
```python
# 필수 검증 항목
- order_date <= shipped_date <= delivered_date
- quantity > 0 AND unit_price > 0
- total_amount = quantity × unit_price × (1 - discount_rate)
- customer_id exists in customers table
- product_id exists in products table
- payment_method in ['credit_card', 'debit_card', 'paypal', 'cash']
```

**시나리오 2: 의료 환자 데이터**
```python
# 필수 검증 항목
- admission_date <= discharge_date
- age = (today - birth_date) / 365.25
- blood_pressure_systolic > blood_pressure_diastolic
- patient_id is unique
- diagnosis_code follows ICD-10 format
- medication_dose within safe range
```

**시나리오 3: 금융 거래 데이터**
```python
# 필수 검증 항목
- transaction_amount != 0
- account_balance_before + transaction_amount = account_balance_after
- transaction_date >= account_open_date
- currency_code in ISO_4217_codes
- transaction_type in ['deposit', 'withdrawal', 'transfer']
```

---

## 3. 구현 (Implementation)

### 3.1 교차 필드 검증 (Cross-Field Validation)

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

class CrossFieldValidator:
    """
    교차 필드 일관성 검증 클래스
    Cross-field consistency validation
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Parameters:
        -----------
        df : pd.DataFrame
            검증할 데이터프레임
        """
        self.df = df.copy()
        self.violations = []
        
    def validate_date_sequence(
        self, 
        date_columns: List[str],
        allow_same_date: bool = False
    ) -> pd.DataFrame:
        """
        날짜 컬럼들이 올바른 순서인지 검증
        Validate chronological order of date columns
        
        Parameters:
        -----------
        date_columns : List[str]
            순서대로 정렬되어야 하는 날짜 컬럼 리스트
            예: ['start_date', 'end_date', 'completed_date']
        allow_same_date : bool
            동일한 날짜를 허용할지 여부
            
        Returns:
        --------
        violations_df : pd.DataFrame
            규칙 위반 레코드
            
        Example:
        --------
        >>> validator = CrossFieldValidator(df)
        >>> violations = validator.validate_date_sequence(
        ...     ['order_date', 'shipped_date', 'delivered_date']
        ... )
        """
        # 날짜 컬럼을 datetime으로 변환
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # 순서 검증
        mask = pd.Series([False] * len(self.df), index=self.df.index)
        
        for i in range(len(date_columns) - 1):
            col1, col2 = date_columns[i], date_columns[i + 1]
            
            if col1 not in self.df.columns or col2 not in self.df.columns:
                continue
                
            if allow_same_date:
                # col1 <= col2
                invalid = self.df[col1] > self.df[col2]
            else:
                # col1 < col2
                invalid = self.df[col1] >= self.df[col2]
            
            mask = mask | invalid.fillna(False)
        
        violations = self.df[mask].copy()
        
        if len(violations) > 0:
            self.violations.append({
                'rule': f'Date sequence: {" < ".join(date_columns)}',
                'violations_count': len(violations),
                'severity': 'HIGH',
                'columns': date_columns
            })
            
        return violations
    
    def validate_calculation(
        self,
        result_column: str,
        formula: str,
        tolerance: float = 0.01
    ) -> pd.DataFrame:
        """
        계산식 검증 (예: total = quantity × price)
        Validate calculation formulas
        
        Parameters:
        -----------
        result_column : str
            결과 컬럼명
        formula : str
            계산 공식 (pandas eval 형식)
            예: 'quantity * unit_price'
        tolerance : float
            허용 오차 (부동소수점 비교용)
            
        Returns:
        --------
        violations_df : pd.DataFrame
            계산 불일치 레코드
            
        Example:
        --------
        >>> violations = validator.validate_calculation(
        ...     result_column='total_amount',
        ...     formula='quantity * unit_price * (1 - discount_rate)',
        ...     tolerance=0.01
        ... )
        """
        try:
            # 공식 계산
            calculated = self.df.eval(formula)
            actual = self.df[result_column]
            
            # 차이 계산 (절대값)
            difference = np.abs(calculated - actual)
            
            # 허용 오차를 초과하는 레코드
            mask = difference > tolerance
            violations = self.df[mask].copy()
            
            # 계산 결과 추가
            violations['calculated_value'] = calculated[mask]
            violations['actual_value'] = actual[mask]
            violations['difference'] = difference[mask]
            
            if len(violations) > 0:
                self.violations.append({
                    'rule': f'{result_column} = {formula}',
                    'violations_count': len(violations),
                    'severity': 'HIGH',
                    'columns': [result_column]
                })
                
            return violations
            
        except Exception as e:
            print(f"Error in calculation validation: {e}")
            return pd.DataFrame()
    
    def validate_age_birthdate(
        self,
        age_column: str = 'age',
        birthdate_column: str = 'birth_date',
        tolerance_years: float = 1.0
    ) -> pd.DataFrame:
        """
        나이와 생년월일 일치성 검증
        Validate consistency between age and birth date
        
        Parameters:
        -----------
        age_column : str
            나이 컬럼명
        birthdate_column : str
            생년월일 컬럼명
        tolerance_years : float
            허용 오차 (년)
            
        Returns:
        --------
        violations_df : pd.DataFrame
            나이 불일치 레코드
        """
        if age_column not in self.df.columns or birthdate_column not in self.df.columns:
            return pd.DataFrame()
        
        # 생년월일을 datetime으로 변환
        birth_dates = pd.to_datetime(self.df[birthdate_column], errors='coerce')
        
        # 현재 날짜 기준 나이 계산
        today = pd.Timestamp.now()
        calculated_age = (today - birth_dates).dt.days / 365.25
        
        # 나이 차이 계산
        age_difference = np.abs(self.df[age_column] - calculated_age)
        
        # 허용 오차를 초과하는 레코드
        mask = age_difference > tolerance_years
        violations = self.df[mask].copy()
        
        violations['calculated_age'] = calculated_age[mask]
        violations['actual_age'] = self.df[age_column][mask]
        violations['age_difference'] = age_difference[mask]
        
        if len(violations) > 0:
            self.violations.append({
                'rule': f'{age_column} must match {birthdate_column}',
                'violations_count': len(violations),
                'severity': 'MEDIUM',
                'columns': [age_column, birthdate_column]
            })
            
        return violations
    
    def validate_conditional_logic(
        self,
        condition: str,
        required_fields: List[str],
        rule_description: str
    ) -> pd.DataFrame:
        """
        조건부 로직 검증
        Validate conditional business logic
        
        Parameters:
        -----------
        condition : str
            조건식 (pandas query 형식)
            예: "status == 'active' and end_date.isnull()"
        required_fields : List[str]
            조건에 사용된 필드 목록
        rule_description : str
            규칙 설명
            
        Returns:
        --------
        violations_df : pd.DataFrame
            조건 위반 레코드
            
        Example:
        --------
        >>> # 활성 상태인 경우 종료일이 없어야 함
        >>> violations = validator.validate_conditional_logic(
        ...     condition="status == 'active' and end_date.notna()",
        ...     required_fields=['status', 'end_date'],
        ...     rule_description='Active records should not have end_date'
        ... )
        """
        try:
            # 조건에 맞는 레코드 찾기 (이것이 위반)
            violations = self.df.query(condition).copy()
            
            if len(violations) > 0:
                self.violations.append({
                    'rule': rule_description,
                    'violations_count': len(violations),
                    'severity': 'MEDIUM',
                    'columns': required_fields
                })
                
            return violations
            
        except Exception as e:
            print(f"Error in conditional logic validation: {e}")
            return pd.DataFrame()
    
    def get_validation_summary(self) -> pd.DataFrame:
        """
        검증 결과 요약
        Get validation summary
        
        Returns:
        --------
        summary_df : pd.DataFrame
            검증 결과 요약 테이블
        """
        if not self.violations:
            print("✅ No validation violations found!")
            return pd.DataFrame()
        
        summary = pd.DataFrame(self.violations)
        summary = summary.sort_values('violations_count', ascending=False)
        
        print(f"\n⚠️ Found {len(summary)} validation rule violations")
        print(f"Total violation records: {summary['violations_count'].sum()}")
        
        return summary


# 사용 예시 1: E-commerce 주문 데이터 검증
def validate_ecommerce_orders(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    E-commerce 주문 데이터 종합 검증
    Comprehensive validation for e-commerce orders
    """
    validator = CrossFieldValidator(df)
    results = {}
    
    # 1. 날짜 순서 검증
    print("1. Validating date sequence...")
    results['date_violations'] = validator.validate_date_sequence(
        ['order_date', 'shipped_date', 'delivered_date']
    )
    
    # 2. 금액 계산 검증
    print("2. Validating amount calculation...")
    results['amount_violations'] = validator.validate_calculation(
        result_column='total_amount',
        formula='quantity * unit_price',
        tolerance=0.01
    )
    
    # 3. 수량 양수 검증
    print("3. Validating positive quantities...")
    results['quantity_violations'] = validator.validate_conditional_logic(
        condition='quantity <= 0',
        required_fields=['quantity'],
        rule_description='Quantity must be positive'
    )
    
    # 4. 할인율 범위 검증
    print("4. Validating discount rate...")
    if 'discount_rate' in df.columns:
        results['discount_violations'] = validator.validate_conditional_logic(
            condition='discount_rate < 0 or discount_rate > 1',
            required_fields=['discount_rate'],
            rule_description='Discount rate must be between 0 and 1'
        )
    
    # 검증 요약
    print("\n" + "="*60)
    summary = validator.get_validation_summary()
    print("\nValidation Summary:")
    print(summary)
    
    return results


# 사용 예시 2: 의료 환자 데이터 검증
def validate_patient_records(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    의료 환자 데이터 종합 검증
    Comprehensive validation for patient records
    """
    validator = CrossFieldValidator(df)
    results = {}
    
    # 1. 입원-퇴원 날짜 검증
    print("1. Validating admission/discharge dates...")
    results['date_violations'] = validator.validate_date_sequence(
        ['admission_date', 'discharge_date']
    )
    
    # 2. 나이-생년월일 검증
    print("2. Validating age vs birth date...")
    results['age_violations'] = validator.validate_age_birthdate(
        age_column='age',
        birthdate_column='birth_date',
        tolerance_years=1.0
    )
    
    # 3. 혈압 검증 (수축기 > 이완기)
    print("3. Validating blood pressure...")
    if 'bp_systolic' in df.columns and 'bp_diastolic' in df.columns:
        results['bp_violations'] = validator.validate_conditional_logic(
            condition='bp_systolic <= bp_diastolic',
            required_fields=['bp_systolic', 'bp_diastolic'],
            rule_description='Systolic BP must be greater than Diastolic BP'
        )
    
    # 4. 체질량지수(BMI) 계산 검증
    print("4. Validating BMI calculation...")
    if all(col in df.columns for col in ['bmi', 'weight_kg', 'height_m']):
        results['bmi_violations'] = validator.validate_calculation(
            result_column='bmi',
            formula='weight_kg / (height_m ** 2)',
            tolerance=0.1
        )
    
    # 검증 요약
    print("\n" + "="*60)
    summary = validator.get_validation_summary()
    print("\nValidation Summary:")
    print(summary)
    
    return results
```

### 3.2 참조 무결성 검증 (Referential Integrity Validation)

```python
class ReferentialIntegrityValidator:
    """
    참조 무결성 검증 클래스
    Foreign key and referential integrity validation
    """
    
    def __init__(self):
        self.violations = []
    
    def validate_foreign_key(
        self,
        df: pd.DataFrame,
        fk_column: str,
        reference_df: pd.DataFrame,
        pk_column: str,
        relationship_name: str = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        외래 키(Foreign Key) 검증
        Validate foreign key relationships
        
        Parameters:
        -----------
        df : pd.DataFrame
            검증할 주 데이터프레임 (외래 키 포함)
        fk_column : str
            외래 키 컬럼명
        reference_df : pd.DataFrame
            참조 데이터프레임 (기본 키 포함)
        pk_column : str
            기본 키 컬럼명
        relationship_name : str, optional
            관계 이름 (예: 'orders -> customers')
            
        Returns:
        --------
        orphan_records : pd.DataFrame
            고아 레코드 (orphan records)
        stats : dict
            검증 통계
            
        Example:
        --------
        >>> validator = ReferentialIntegrityValidator()
        >>> orphans, stats = validator.validate_foreign_key(
        ...     df=orders_df,
        ...     fk_column='customer_id',
        ...     reference_df=customers_df,
        ...     pk_column='customer_id',
        ...     relationship_name='orders -> customers'
        ... )
        """
        # NULL 값 제외 (NULL은 외래 키에서 허용될 수 있음)
        non_null_fk = df[df[fk_column].notna()].copy()
        
        # 참조 테이블의 유니크 키
        reference_keys = set(reference_df[pk_column].dropna().unique())
        
        # 고아 레코드 찾기
        orphan_mask = ~non_null_fk[fk_column].isin(reference_keys)
        orphan_records = non_null_fk[orphan_mask].copy()
        
        # 통계 계산
        stats = {
            'relationship': relationship_name or f'{fk_column} -> {pk_column}',
            'total_records': len(df),
            'non_null_fk': len(non_null_fk),
            'null_fk': len(df) - len(non_null_fk),
            'orphan_records': len(orphan_records),
            'orphan_percentage': 100 * len(orphan_records) / len(non_null_fk) if len(non_null_fk) > 0 else 0,
            'unique_orphan_keys': orphan_records[fk_column].nunique(),
            'reference_table_size': len(reference_df),
            'reference_unique_keys': len(reference_keys)
        }
        
        # 위반 기록
        if len(orphan_records) > 0:
            self.violations.append({
                'relationship': stats['relationship'],
                'orphan_count': stats['orphan_records'],
                'orphan_pct': stats['orphan_percentage'],
                'severity': 'HIGH' if stats['orphan_percentage'] > 5 else 'MEDIUM'
            })
        
        return orphan_records, stats
    
    def validate_multiple_relationships(
        self,
        relationships: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        여러 외래 키 관계를 한번에 검증
        Validate multiple foreign key relationships
        
        Parameters:
        -----------
        relationships : List[Dict]
            검증할 관계 목록
            [{
                'df': main_dataframe,
                'fk_column': 'foreign_key_column',
                'reference_df': reference_dataframe,
                'pk_column': 'primary_key_column',
                'name': 'relationship_name'
            }, ...]
            
        Returns:
        --------
        summary_df : pd.DataFrame
            검증 결과 요약
            
        Example:
        --------
        >>> relationships = [
        ...     {
        ...         'df': orders_df,
        ...         'fk_column': 'customer_id',
        ...         'reference_df': customers_df,
        ...         'pk_column': 'customer_id',
        ...         'name': 'orders -> customers'
        ...     },
        ...     {
        ...         'df': orders_df,
        ...         'fk_column': 'product_id',
        ...         'reference_df': products_df,
        ...         'pk_column': 'product_id',
        ...         'name': 'orders -> products'
        ...     }
        ... ]
        >>> summary = validator.validate_multiple_relationships(relationships)
        """
        results = []
        
        for rel in relationships:
            print(f"\nValidating: {rel.get('name', 'Unknown relationship')}")
            orphans, stats = self.validate_foreign_key(
                df=rel['df'],
                fk_column=rel['fk_column'],
                reference_df=rel['reference_df'],
                pk_column=rel['pk_column'],
                relationship_name=rel.get('name')
            )
            
            results.append(stats)
            
            # 고아 레코드 통계 출력
            if stats['orphan_records'] > 0:
                print(f"  ⚠️ Found {stats['orphan_records']} orphan records "
                      f"({stats['orphan_percentage']:.2f}%)")
                print(f"  Unique orphan keys: {stats['unique_orphan_keys']}")
            else:
                print(f"  ✅ No orphan records found")
        
        summary_df = pd.DataFrame(results)
        return summary_df
    
    def validate_bidirectional_relationship(
        self,
        df1: pd.DataFrame,
        col1: str,
        df2: pd.DataFrame,
        col2: str,
        relationship_name: str = None
    ) -> Dict[str, Any]:
        """
        양방향 관계 검증 (Many-to-Many)
        Validate bidirectional relationships
        
        Parameters:
        -----------
        df1 : pd.DataFrame
            첫 번째 데이터프레임
        col1 : str
            첫 번째 데이터프레임의 키 컬럼
        df2 : pd.DataFrame
            두 번째 데이터프레임
        col2 : str
            두 번째 데이터프레임의 키 컬럼
        relationship_name : str, optional
            관계 이름
            
        Returns:
        --------
        results : dict
            양방향 검증 결과
        """
        # df1 -> df2 검증
        keys1 = set(df1[col1].dropna().unique())
        keys2 = set(df2[col2].dropna().unique())
        
        # 각 방향의 고아 키
        orphans_in_df1 = keys1 - keys2  # df1에만 있고 df2에 없는 키
        orphans_in_df2 = keys2 - keys1  # df2에만 있고 df1에 없는 키
        
        results = {
            'relationship': relationship_name or f'{col1} <-> {col2}',
            'keys_in_df1': len(keys1),
            'keys_in_df2': len(keys2),
            'common_keys': len(keys1 & keys2),
            'orphans_in_df1': len(orphans_in_df1),
            'orphans_in_df2': len(orphans_in_df2),
            'orphan_keys_df1': list(orphans_in_df1)[:10],  # 샘플
            'orphan_keys_df2': list(orphans_in_df2)[:10]   # 샘플
        }
        
        return results
    
    def get_integrity_report(self) -> pd.DataFrame:
        """
        참조 무결성 검증 리포트
        Get referential integrity report
        """
        if not self.violations:
            print("✅ No referential integrity violations found!")
            return pd.DataFrame()
        
        report = pd.DataFrame(self.violations)
        report = report.sort_values('orphan_count', ascending=False)
        
        print(f"\n⚠️ Found {len(report)} referential integrity violations")
        print(f"Total orphan records: {report['orphan_count'].sum()}")
        
        return report


# 사용 예시: E-commerce 데이터베이스 무결성 검증
def validate_ecommerce_integrity(
    orders_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    products_df: pd.DataFrame,
    payments_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    E-commerce 데이터베이스의 참조 무결성 종합 검증
    """
    validator = ReferentialIntegrityValidator()
    
    # 여러 관계 정의
    relationships = [
        {
            'df': orders_df,
            'fk_column': 'customer_id',
            'reference_df': customers_df,
            'pk_column': 'customer_id',
            'name': 'orders -> customers'
        },
        {
            'df': orders_df,
            'fk_column': 'product_id',
            'reference_df': products_df,
            'pk_column': 'product_id',
            'name': 'orders -> products'
        },
        {
            'df': payments_df,
            'fk_column': 'order_id',
            'reference_df': orders_df,
            'pk_column': 'order_id',
            'name': 'payments -> orders'
        }
    ]
    
    # 일괄 검증
    summary = validator.validate_multiple_relationships(relationships)
    
    # 리포트 생성
    integrity_report = validator.get_integrity_report()
    
    return {
        'summary': summary,
        'report': integrity_report
    }
```

### 3.3 비즈니스 로직 검증 (Business Logic Validation)

```python
from typing import Callable

class BusinessRuleValidator:
    """
    비즈니스 규칙 검증 클래스
    Business rule validation engine
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.rules = []
        self.results = []
    
    def add_rule(
        self,
        rule_name: str,
        rule_func: Callable,
        severity: str = 'MEDIUM',
        description: str = None
    ):
        """
        비즈니스 규칙 추가
        Add a business rule
        
        Parameters:
        -----------
        rule_name : str
            규칙 이름
        rule_func : Callable
            검증 함수 (DataFrame을 받아 위반 레코드 반환)
        severity : str
            심각도 ('HIGH', 'MEDIUM', 'LOW')
        description : str
            규칙 설명
        """
        self.rules.append({
            'name': rule_name,
            'func': rule_func,
            'severity': severity,
            'description': description or rule_name
        })
    
    def validate_all(self) -> pd.DataFrame:
        """
        모든 규칙 검증
        Validate all business rules
        
        Returns:
        --------
        summary_df : pd.DataFrame
            검증 결과 요약
        """
        print("Starting business rule validation...")
        print(f"Total rules to validate: {len(self.rules)}\n")
        
        for i, rule in enumerate(self.rules, 1):
            print(f"[{i}/{len(self.rules)}] Validating: {rule['name']}")
            
            try:
                violations = rule['func'](self.df)
                
                result = {
                    'rule_name': rule['name'],
                    'description': rule['description'],
                    'severity': rule['severity'],
                    'violations_count': len(violations),
                    'violations_pct': 100 * len(violations) / len(self.df),
                    'status': 'PASS' if len(violations) == 0 else 'FAIL'
                }
                
                self.results.append(result)
                
                if len(violations) > 0:
                    print(f"  ⚠️ Found {len(violations)} violations ({result['violations_pct']:.2f}%)")
                else:
                    print(f"  ✅ Passed")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
                self.results.append({
                    'rule_name': rule['name'],
                    'description': rule['description'],
                    'severity': rule['severity'],
                    'violations_count': -1,
                    'violations_pct': -1,
                    'status': 'ERROR',
                    'error': str(e)
                })
        
        summary_df = pd.DataFrame(self.results)
        return summary_df
    
    def get_violation_details(
        self,
        rule_name: str,
        max_records: int = 100
    ) -> pd.DataFrame:
        """
        특정 규칙의 위반 상세 조회
        Get detailed violations for a specific rule
        """
        rule = next((r for r in self.rules if r['name'] == rule_name), None)
        
        if rule is None:
            print(f"Rule '{rule_name}' not found")
            return pd.DataFrame()
        
        violations = rule['func'](self.df)
        
        if len(violations) > max_records:
            print(f"Showing first {max_records} of {len(violations)} violations")
            return violations.head(max_records)
        
        return violations


# 도메인별 비즈니스 규칙 예시

def create_financial_rules(df: pd.DataFrame) -> BusinessRuleValidator:
    """
    금융 도메인 비즈니스 규칙
    Financial domain business rules
    """
    validator = BusinessRuleValidator(df)
    
    # 규칙 1: 거래 금액은 0이 아니어야 함
    validator.add_rule(
        rule_name='NonZeroTransaction',
        rule_func=lambda df: df[df['transaction_amount'] == 0],
        severity='HIGH',
        description='Transaction amount must be non-zero'
    )
    
    # 규칙 2: 잔액 정합성 (이전 잔액 + 거래 금액 = 현재 잔액)
    validator.add_rule(
        rule_name='BalanceConsistency',
        rule_func=lambda df: df[
            np.abs(
                (df['balance_before'] + df['transaction_amount']) - df['balance_after']
            ) > 0.01
        ],
        severity='HIGH',
        description='Balance must be consistent: balance_before + transaction = balance_after'
    )
    
    # 규칙 3: 마이너스 잔액 확인 (당좌 계좌 제외)
    validator.add_rule(
        rule_name='NegativeBalance',
        rule_func=lambda df: df[
            (df['balance_after'] < 0) & (df['account_type'] != 'overdraft')
        ],
        severity='HIGH',
        description='Non-overdraft accounts cannot have negative balance'
    )
    
    # 규칙 4: 일일 거래 한도 초과
    validator.add_rule(
        rule_name='DailyTransactionLimit',
        rule_func=lambda df: df[
            df.groupby('account_id')['transaction_amount'].transform('sum') > 10000
        ],
        severity='MEDIUM',
        description='Daily transaction limit exceeded (>10,000)'
    )
    
    return validator


def create_healthcare_rules(df: pd.DataFrame) -> BusinessRuleValidator:
    """
    의료 도메인 비즈니스 규칙
    Healthcare domain business rules
    """
    validator = BusinessRuleValidator(df)
    
    # 규칙 1: 환자 나이 범위 (0-120)
    validator.add_rule(
        rule_name='ValidAge',
        rule_func=lambda df: df[(df['age'] < 0) | (df['age'] > 120)],
        severity='HIGH',
        description='Patient age must be between 0 and 120'
    )
    
    # 규칙 2: 혈압 정상 범위 (수축기: 70-200, 이완기: 40-130)
    validator.add_rule(
        rule_name='ValidBloodPressure',
        rule_func=lambda df: df[
            (df['bp_systolic'] < 70) | (df['bp_systolic'] > 200) |
            (df['bp_diastolic'] < 40) | (df['bp_diastolic'] > 130)
        ],
        severity='MEDIUM',
        description='Blood pressure must be within normal range'
    )
    
    # 규칙 3: 체온 정상 범위 (35-42°C)
    validator.add_rule(
        rule_name='ValidTemperature',
        rule_func=lambda df: df[
            (df['temperature'] < 35) | (df['temperature'] > 42)
        ],
        severity='HIGH',
        description='Body temperature must be between 35 and 42°C'
    )
    
    # 규칙 4: 약물 투여량 안전 범위
    validator.add_rule(
        rule_name='SafeDosage',
        rule_func=lambda df: df[
            df['dosage_mg'] > df['max_safe_dosage_mg']
        ],
        severity='HIGH',
        description='Dosage must not exceed safe maximum'
    )
    
    # 규칙 5: 재입원 간격 (퇴원 후 최소 1일 경과)
    if 'previous_discharge_date' in df.columns:
        validator.add_rule(
            rule_name='ReadmissionInterval',
            rule_func=lambda df: df[
                (pd.to_datetime(df['admission_date']) - 
                 pd.to_datetime(df['previous_discharge_date'])).dt.days < 1
            ],
            severity='MEDIUM',
            description='Readmission must be at least 1 day after discharge'
        )
    
    return validator


def create_ecommerce_rules(df: pd.DataFrame) -> BusinessRuleValidator:
    """
    E-commerce 도메인 비즈니스 규칙
    E-commerce domain business rules
    """
    validator = BusinessRuleValidator(df)
    
    # 규칙 1: 주문 수량은 양수
    validator.add_rule(
        rule_name='PositiveQuantity',
        rule_func=lambda df: df[df['quantity'] <= 0],
        severity='HIGH',
        description='Order quantity must be positive'
    )
    
    # 규칙 2: 할인율 범위 (0-100%)
    validator.add_rule(
        rule_name='ValidDiscountRate',
        rule_func=lambda df: df[
            (df['discount_rate'] < 0) | (df['discount_rate'] > 1)
        ],
        severity='MEDIUM',
        description='Discount rate must be between 0 and 1'
    )
    
    # 규칙 3: 배송비는 주문 금액에 따라 적정해야 함
    validator.add_rule(
        rule_name='ShippingCost',
        rule_func=lambda df: df[
            (df['order_amount'] > 50) & (df['shipping_cost'] > 0)
        ],
        severity='LOW',
        description='Free shipping for orders over $50'
    )
    
    # 규칙 4: 환불 금액은 원 주문 금액 이하
    if 'refund_amount' in df.columns:
        validator.add_rule(
            rule_name='ValidRefundAmount',
            rule_func=lambda df: df[
                df['refund_amount'] > df['order_amount']
            ],
            severity='HIGH',
            description='Refund amount cannot exceed original order amount'
        )
    
    # 규칙 5: 취소된 주문은 배송되지 않아야 함
    validator.add_rule(
        rule_name='CancelledOrderNotShipped',
        rule_func=lambda df: df[
            (df['order_status'] == 'cancelled') & (df['shipped_date'].notna())
        ],
        severity='HIGH',
        description='Cancelled orders should not be shipped'
    )
    
    return validator
```

### 3.4 Great Expectations 통합

```python
try:
    import great_expectations as gx
    from great_expectations.core.batch import RuntimeBatchRequest
    HAS_GX = True
except ImportError:
    HAS_GX = False
    print("Great Expectations not installed. Install with: pip install great-expectations")


class GreatExpectationsValidator:
    """
    Great Expectations 기반 자동 검증
    Automated validation using Great Expectations
    """
    
    def __init__(self, context_root_dir: str = None):
        """
        Parameters:
        -----------
        context_root_dir : str, optional
            Great Expectations 프로젝트 루트 디렉토리
        """
        if not HAS_GX:
            raise ImportError("Great Expectations is required for this validator")
        
        self.context = gx.get_context(context_root_dir=context_root_dir)
    
    def create_expectation_suite(
        self,
        suite_name: str,
        df: pd.DataFrame
    ) -> None:
        """
        Expectation Suite 생성
        Create an expectation suite
        """
        # Data Source 생성
        datasource_name = f"datasource_{suite_name}"
        
        # Pandas DataFrame을 데이터 소스로 추가
        datasource = self.context.sources.add_or_update_pandas(datasource_name)
        
        # Data Asset 추가
        data_asset = datasource.add_dataframe_asset(name=f"asset_{suite_name}")
        
        # Batch Request 생성
        batch_request = data_asset.build_batch_request(dataframe=df)
        
        # Expectation Suite 생성
        self.context.add_or_update_expectation_suite(suite_name)
        
        # Validator 생성
        validator = self.context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=suite_name
        )
        
        return validator
    
    def add_common_expectations(
        self,
        validator,
        column_config: Dict[str, Dict[str, Any]]
    ):
        """
        일반적인 Expectations 추가
        Add common expectations based on column configuration
        
        Parameters:
        -----------
        validator : Validator
            Great Expectations Validator
        column_config : Dict
            컬럼별 설정
            {
                'column_name': {
                    'type': 'numeric|categorical|datetime',
                    'nullable': True|False,
                    'unique': True|False,
                    'min': value,
                    'max': value,
                    'values': [allowed_values]
                }
            }
        """
        df = validator.active_batch_definition
        
        for column, config in column_config.items():
            if column not in validator.active_batch.data.columns:
                continue
            
            # 1. 컬럼 존재 확인
            validator.expect_column_to_exist(column)
            
            # 2. Null 값 체크
            if not config.get('nullable', True):
                validator.expect_column_values_to_not_be_null(column)
            
            # 3. 유니크 체크
            if config.get('unique', False):
                validator.expect_column_values_to_be_unique(column)
            
            # 4. 데이터 타입별 검증
            if config.get('type') == 'numeric':
                # 수치형 범위 검증
                if 'min' in config:
                    validator.expect_column_values_to_be_between(
                        column,
                        min_value=config['min'],
                        max_value=config.get('max')
                    )
            
            elif config.get('type') == 'categorical':
                # 범주형 값 검증
                if 'values' in config:
                    validator.expect_column_values_to_be_in_set(
                        column,
                        value_set=config['values']
                    )
            
            elif config.get('type') == 'datetime':
                # 날짜 형식 검증
                validator.expect_column_values_to_be_of_type(
                    column,
                    type_='datetime64[ns]'
                )
        
        # Suite 저장
        validator.save_expectation_suite(discard_failed_expectations=False)
        
        return validator
    
    def run_validation(
        self,
        df: pd.DataFrame,
        suite_name: str
    ) -> Dict[str, Any]:
        """
        검증 실행
        Run validation
        """
        # Checkpoint 생성
        checkpoint_name = f"checkpoint_{suite_name}"
        
        checkpoint = self.context.add_or_update_checkpoint(
            name=checkpoint_name,
            validations=[
                {
                    "batch_request": {
                        "datasource_name": f"datasource_{suite_name}",
                        "data_asset_name": f"asset_{suite_name}",
                        "options": {}
                    },
                    "expectation_suite_name": suite_name
                }
            ]
        )
        
        # 검증 실행
        results = checkpoint.run()
        
        return results


# 사용 예시
def validate_with_great_expectations(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Great Expectations를 사용한 포괄적 검증
    """
    if not HAS_GX:
        print("Great Expectations not available. Skipping...")
        return {}
    
    validator_gx = GreatExpectationsValidator()
    
    # Expectation Suite 생성
    validator = validator_gx.create_expectation_suite(
        suite_name="data_quality_suite",
        df=df
    )
    
    # 컬럼 설정
    column_config = {
        'age': {
            'type': 'numeric',
            'nullable': False,
            'min': 0,
            'max': 120
        },
        'gender': {
            'type': 'categorical',
            'nullable': False,
            'values': ['Male', 'Female', 'Other']
        },
        'email': {
            'type': 'string',
            'nullable': False,
            'unique': True
        },
        'signup_date': {
            'type': 'datetime',
            'nullable': False
        }
    }
    
    # Expectations 추가
    validator_gx.add_common_expectations(validator, column_config)
    
    # 검증 실행
    results = validator_gx.run_validation(df, "data_quality_suite")
    
    return results
```

### 3.5 Pandera 스키마 기반 검증

```python
try:
    import pandera as pa
    from pandera import Column, DataFrameSchema, Check
    HAS_PANDERA = True
except ImportError:
    HAS_PANDERA = False
    print("Pandera not installed. Install with: pip install pandera")


class PanderaValidator:
    """
    Pandera 스키마 기반 검증
    Schema-based validation using Pandera
    """
    
    @staticmethod
    def create_ecommerce_schema() -> DataFrameSchema:
        """
        E-commerce 데이터 스키마 정의
        """
        if not HAS_PANDERA:
            return None
        
        schema = DataFrameSchema({
            "order_id": Column(
                dtype="int64",
                checks=[
                    Check.greater_than(0),
                    Check(lambda s: s.is_unique, error="order_id must be unique")
                ],
                nullable=False
            ),
            "customer_id": Column(
                dtype="int64",
                checks=Check.greater_than(0),
                nullable=False
            ),
            "order_date": Column(
                dtype="datetime64[ns]",
                checks=Check.less_than_or_equal_to(pd.Timestamp.now()),
                nullable=False
            ),
            "quantity": Column(
                dtype="int64",
                checks=Check.in_range(1, 1000),
                nullable=False
            ),
            "unit_price": Column(
                dtype="float64",
                checks=Check.greater_than(0),
                nullable=False
            ),
            "discount_rate": Column(
                dtype="float64",
                checks=Check.in_range(0, 1),
                nullable=True
            ),
            "status": Column(
                dtype="object",
                checks=Check.isin(['pending', 'processing', 'shipped', 'delivered', 'cancelled']),
                nullable=False
            )
        })
        
        return schema
    
    @staticmethod
    def create_patient_schema() -> DataFrameSchema:
        """
        환자 데이터 스키마 정의
        """
        if not HAS_PANDERA:
            return None
        
        schema = DataFrameSchema({
            "patient_id": Column(
                dtype="object",
                checks=Check(lambda s: s.is_unique),
                nullable=False
            ),
            "age": Column(
                dtype="int64",
                checks=Check.in_range(0, 120),
                nullable=False
            ),
            "birth_date": Column(
                dtype="datetime64[ns]",
                checks=Check.less_than(pd.Timestamp.now()),
                nullable=False
            ),
            "bp_systolic": Column(
                dtype="int64",
                checks=Check.in_range(70, 200),
                nullable=True
            ),
            "bp_diastolic": Column(
                dtype="int64",
                checks=Check.in_range(40, 130),
                nullable=True
            ),
            "temperature": Column(
                dtype="float64",
                checks=Check.in_range(35.0, 42.0),
                nullable=True
            ),
            "admission_date": Column(
                dtype="datetime64[ns]",
                nullable=True
            ),
            "discharge_date": Column(
                dtype="datetime64[ns]",
                nullable=True
            )
        },
        checks=[
            # 교차 필드 검증: 수축기 혈압 > 이완기 혈압
            Check(
                lambda df: (df['bp_systolic'] > df['bp_diastolic']).all(),
                error="bp_systolic must be greater than bp_diastolic"
            ),
            # 교차 필드 검증: 입원일 <= 퇴원일
            Check(
                lambda df: (df['admission_date'] <= df['discharge_date']).all(),
                error="admission_date must be before or equal to discharge_date"
            )
        ])
        
        return schema
    
    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        schema: DataFrameSchema
    ) -> Tuple[bool, pd.DataFrame]:
        """
        데이터프레임 검증
        
        Returns:
        --------
        is_valid : bool
            검증 통과 여부
        validated_df : pd.DataFrame
            검증된 데이터프레임
        """
        if not HAS_PANDERA or schema is None:
            return True, df
        
        try:
            validated_df = schema.validate(df, lazy=True)
            print("✅ Schema validation passed!")
            return True, validated_df
            
        except pa.errors.SchemaErrors as err:
            print("⚠️ Schema validation failed!")
            print(f"\nFailure cases: {len(err.failure_cases)}")
            print("\nError details:")
            print(err.failure_cases)
            
            return False, df


# 사용 예시
def validate_with_pandera(df: pd.DataFrame, domain: str = 'ecommerce'):
    """
    Pandera를 사용한 스키마 기반 검증
    """
    if not HAS_PANDERA:
        print("Pandera not available. Skipping...")
        return df
    
    validator = PanderaValidator()
    
    # 도메인별 스키마 선택
    if domain == 'ecommerce':
        schema = validator.create_ecommerce_schema()
    elif domain == 'healthcare':
        schema = validator.create_patient_schema()
    else:
        print(f"Unknown domain: {domain}")
        return df
    
    # 검증 실행
    is_valid, validated_df = validator.validate_dataframe(df, schema)
    
    return validated_df
```

---

## 4. 예시 (Examples)

### 4.1 전체 검증 파이프라인 예시

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 샘플 데이터 생성
def create_sample_ecommerce_data():
    """
    E-commerce 샘플 데이터 생성 (일부 오류 포함)
    """
    np.random.seed(42)
    n_orders = 1000
    
    df = pd.DataFrame({
        'order_id': range(1, n_orders + 1),
        'customer_id': np.random.randint(1, 201, n_orders),
        'product_id': np.random.randint(1, 51, n_orders),
        'order_date': pd.date_range('2024-01-01', periods=n_orders, freq='H'),
        'quantity': np.random.randint(1, 10, n_orders),
        'unit_price': np.random.uniform(10, 500, n_orders).round(2),
        'discount_rate': np.random.uniform(0, 0.3, n_orders).round(2),
        'total_amount': 0.0,  # 나중에 계산
        'status': np.random.choice(
            ['pending', 'processing', 'shipped', 'delivered', 'cancelled'],
            n_orders
        )
    })
    
    # total_amount 계산
    df['total_amount'] = (df['quantity'] * df['unit_price'] * (1 - df['discount_rate'])).round(2)
    
    # shipped_date, delivered_date 추가
    df['shipped_date'] = df['order_date'] + pd.to_timedelta(
        np.random.randint(1, 5, n_orders), unit='D'
    )
    df['delivered_date'] = df['shipped_date'] + pd.to_timedelta(
        np.random.randint(2, 10, n_orders), unit='D'
    )
    
    # 의도적 오류 삽입
    # 1. 날짜 순서 오류 (5%)
    error_idx = np.random.choice(df.index, size=int(0.05 * n_orders), replace=False)
    df.loc[error_idx, 'delivered_date'] = df.loc[error_idx, 'order_date'] - timedelta(days=1)
    
    # 2. 계산 오류 (3%)
    error_idx = np.random.choice(df.index, size=int(0.03 * n_orders), replace=False)
    df.loc[error_idx, 'total_amount'] = df.loc[error_idx, 'total_amount'] * 1.5
    
    # 3. 음수 수량 (2%)
    error_idx = np.random.choice(df.index, size=int(0.02 * n_orders), replace=False)
    df.loc[error_idx, 'quantity'] = -1
    
    # 4. 잘못된 할인율 (1%)
    error_idx = np.random.choice(df.index, size=int(0.01 * n_orders), replace=False)
    df.loc[error_idx, 'discount_rate'] = 1.5
    
    # 5. 취소된 주문이지만 배송됨 (1%)
    cancelled_orders = df[df['status'] == 'cancelled'].sample(frac=0.1).index
    # 이미 shipped_date와 delivered_date가 있으므로 오류
    
    return df


# 참조 테이블 생성
def create_reference_tables():
    """
    고객 및 제품 참조 테이블 생성
    """
    customers = pd.DataFrame({
        'customer_id': range(1, 201),
        'customer_name': [f'Customer_{i}' for i in range(1, 201)],
        'email': [f'customer{i}@example.com' for i in range(1, 201)]
    })
    
    products = pd.DataFrame({
        'product_id': range(1, 51),
        'product_name': [f'Product_{i}' for i in range(1, 51)],
        'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], 50)
    })
    
    return customers, products


# 전체 검증 실행
def run_comprehensive_validation():
    """
    종합 데이터 검증 실행
    """
    print("="*80)
    print("COMPREHENSIVE DATA VALIDATION PIPELINE")
    print("="*80)
    
    # 데이터 로드
    print("\n1. Loading data...")
    df = create_sample_ecommerce_data()
    customers_df, products_df = create_reference_tables()
    print(f"   Orders: {len(df)} rows")
    print(f"   Customers: {len(customers_df)} rows")
    print(f"   Products: {len(products_df)} rows")
    
    # Phase 1: 교차 필드 검증
    print("\n" + "="*80)
    print("PHASE 1: CROSS-FIELD VALIDATION")
    print("="*80)
    
    cross_validator = CrossFieldValidator(df)
    
    # 날짜 순서 검증
    date_violations = cross_validator.validate_date_sequence(
        ['order_date', 'shipped_date', 'delivered_date']
    )
    print(f"\nDate sequence violations: {len(date_violations)}")
    if len(date_violations) > 0:
        print(date_violations[['order_id', 'order_date', 'shipped_date', 'delivered_date']].head())
    
    # 금액 계산 검증
    amount_violations = cross_validator.validate_calculation(
        result_column='total_amount',
        formula='quantity * unit_price * (1 - discount_rate)',
        tolerance=0.01
    )
    print(f"\nAmount calculation violations: {len(amount_violations)}")
    if len(amount_violations) > 0:
        print(amount_violations[['order_id', 'total_amount', 'calculated_value', 'difference']].head())
    
    # 검증 요약
    cross_summary = cross_validator.get_validation_summary()
    
    # Phase 2: 비즈니스 규칙 검증
    print("\n" + "="*80)
    print("PHASE 2: BUSINESS RULE VALIDATION")
    print("="*80)
    
    rule_validator = create_ecommerce_rules(df)
    rule_summary = rule_validator.validate_all()
    
    print("\n" + "-"*80)
    print("Business Rule Summary:")
    print(rule_summary[['rule_name', 'severity', 'violations_count', 'status']])
    
    # Phase 3: 참조 무결성 검증
    print("\n" + "="*80)
    print("PHASE 3: REFERENTIAL INTEGRITY VALIDATION")
    print("="*80)
    
    integrity_validator = ReferentialIntegrityValidator()
    
    relationships = [
        {
            'df': df,
            'fk_column': 'customer_id',
            'reference_df': customers_df,
            'pk_column': 'customer_id',
            'name': 'orders -> customers'
        },
        {
            'df': df,
            'fk_column': 'product_id',
            'reference_df': products_df,
            'pk_column': 'product_id',
            'name': 'orders -> products'
        }
    ]
    
    integrity_summary = integrity_validator.validate_multiple_relationships(relationships)
    print("\n" + "-"*80)
    print("Referential Integrity Summary:")
    print(integrity_summary[['relationship', 'orphan_records', 'orphan_percentage']])
    
    # 최종 리포트
    print("\n" + "="*80)
    print("VALIDATION SUMMARY REPORT")
    print("="*80)
    
    total_violations = (
        len(date_violations) +
        len(amount_violations) +
        rule_summary[rule_summary['status'] == 'FAIL']['violations_count'].sum()
    )
    
    print(f"\n📊 Total records validated: {len(df)}")
    print(f"⚠️  Total violations found: {total_violations}")
    print(f"📉 Violation rate: {100 * total_violations / len(df):.2f}%")
    
    print("\nValidation breakdown:")
    print(f"  - Cross-field violations: {len(date_violations) + len(amount_violations)}")
    print(f"  - Business rule violations: {rule_summary[rule_summary['status'] == 'FAIL']['violations_count'].sum()}")
    print(f"  - Referential integrity violations: {integrity_summary['orphan_records'].sum()}")
    
    # 심각도별 요약
    if len(cross_summary) > 0:
        print("\nBy severity:")
        severity_counts = cross_summary.groupby('severity')['violations_count'].sum()
        for severity, count in severity_counts.items():
            print(f"  - {severity}: {count} violations")
    
    return {
        'cross_field': {'violations': date_violations, 'summary': cross_summary},
        'business_rules': {'summary': rule_summary},
        'referential_integrity': {'summary': integrity_summary}
    }


# 실행
if __name__ == "__main__":
    results = run_comprehensive_validation()
```

### 4.2 출력 예시

```
================================================================================
COMPREHENSIVE DATA VALIDATION PIPELINE
================================================================================

1. Loading data...
   Orders: 1000 rows
   Customers: 200 rows
   Products: 50 rows

================================================================================
PHASE 1: CROSS-FIELD VALIDATION
================================================================================
1. Validating date sequence...

Date sequence violations: 50
   order_id order_date         shipped_date       delivered_date
0        15 2024-01-01 14:00:00 2024-01-05 14:00:00 2023-12-31 14:00:00
1        42 2024-01-02 17:00:00 2024-01-06 17:00:00 2024-01-01 17:00:00
...

2. Validating amount calculation...

Amount calculation violations: 30
   order_id  total_amount  calculated_value  difference
0        73        425.50            283.67      141.83
1       156        892.75            595.17      297.58
...

⚠️ Found 2 validation rule violations
Total violation records: 80

================================================================================
PHASE 2: BUSINESS RULE VALIDATION
================================================================================
Starting business rule validation...
Total rules to validate: 5

[1/5] Validating: PositiveQuantity
  ⚠️ Found 20 violations (2.00%)
[2/5] Validating: ValidDiscountRate
  ⚠️ Found 10 violations (1.00%)
[3/5] Validating: ShippingCost
  ✅ Passed
[4/5] Validating: CancelledOrderNotShipped
  ⚠️ Found 8 violations (0.80%)
[5/5] Validating: ValidRefundAmount
  ✅ Passed

--------------------------------------------------------------------------------
Business Rule Summary:
                   rule_name severity  violations_count status
0          PositiveQuantity     HIGH                20   FAIL
1        ValidDiscountRate   MEDIUM                10   FAIL
2             ShippingCost      LOW                 0   PASS
3  CancelledOrderNotShipped     HIGH                 8   FAIL
4      ValidRefundAmount      HIGH                 0   PASS

================================================================================
PHASE 3: REFERENTIAL INTEGRITY VALIDATION
================================================================================

Validating: orders -> customers
  ✅ No orphan records found

Validating: orders -> products
  ✅ No orphan records found

--------------------------------------------------------------------------------
Referential Integrity Summary:
              relationship  orphan_records  orphan_percentage
0     orders -> customers               0                0.0
1      orders -> products               0                0.0

================================================================================
VALIDATION SUMMARY REPORT
================================================================================

📊 Total records validated: 1000
⚠️  Total violations found: 118
📉 Violation rate: 11.80%

Validation breakdown:
  - Cross-field violations: 80
  - Business rule violations: 38
  - Referential integrity violations: 0

By severity:
  - HIGH: 50 violations
  - MEDIUM: 30 violations
```

---

## 5. 에이전트 매핑 (Agent Mapping)

### 5.1 Primary Agent

**`data-cleaning-specialist`**
- 역할: 데이터 검증 전체 프로세스 총괄
- 책임:
  - 교차 필드 검증 실행
  - 참조 무결성 검증
  - 비즈니스 규칙 정의 및 실행
  - 검증 결과 리포트 생성

### 5.2 Supporting Agents

**`data-scientist`**
- 역할: 통계적 검증 및 패턴 분석
- 책임:
  - 데이터 분포 검증
  - 이상 패턴 탐지
  - 검증 임계값 설정

**`technical-documentation-writer`**
- 역할: 검증 리포트 문서화
- 책임:
  - 검증 결과 리포트 작성
  - 위반 사항 상세 문서화
  - 데이터 품질 대시보드 생성

### 5.3 관련 스킬

**필수 스킬**:
- pandas (데이터 조작 및 검증)
- numpy (수치 연산)
- great-expectations (자동 검증)
- pandera (스키마 기반 검증)

**선택 스킬**:
- regex (문자열 패턴 검증)
- scipy (통계 검정)
- cerberus (데이터 검증 프레임워크)

---

## 6. 필요 라이브러리 (Required Libraries)

### 6.1 핵심 라이브러리

```bash
# 필수 라이브러리
pip install pandas>=2.0.0
pip install numpy>=1.24.0

# 검증 프레임워크
pip install great-expectations>=0.18.0
pip install pandera>=0.17.0

# 통계 및 분석
pip install scipy>=1.11.0
```

### 6.2 선택 라이브러리

```bash
# 추가 검증 도구
pip install cerberus>=1.3.5
pip install pydantic>=2.0.0
pip install jsonschema>=4.19.0

# 데이터베이스 지원
pip install sqlalchemy>=2.0.0
pip install psycopg2-binary>=2.9.0  # PostgreSQL
pip install pymysql>=1.1.0  # MySQL
```

### 6.3 라이브러리 버전 관리

```python
# requirements-validation.txt
pandas==2.1.4
numpy==1.26.2
great-expectations==0.18.8
pandera==0.17.2
scipy==1.11.4
cerberus==1.3.5
pydantic==2.5.3
jsonschema==4.20.0
```

---

## 7. 체크포인트 (Checkpoints)

### 7.1 검증 전 체크리스트

- [ ] 데이터 로드 완료
- [ ] 컬럼 이름 및 타입 확인
- [ ] 예상 스키마 정의 완료
- [ ] 비즈니스 규칙 문서화 완료
- [ ] 참조 테이블 준비 완료

### 7.2 검증 중 체크리스트

- [ ] 교차 필드 검증 실행
  - [ ] 날짜 순서 검증
  - [ ] 계산식 검증
  - [ ] 조건부 로직 검증

- [ ] 참조 무결성 검증 실행
  - [ ] 외래 키 검증
  - [ ] 고아 레코드 식별
  - [ ] 양방향 관계 검증

- [ ] 비즈니스 규칙 검증 실행
  - [ ] 도메인별 규칙 적용
  - [ ] 위반 사항 기록
  - [ ] 심각도 분류

### 7.3 검증 후 체크리스트

- [ ] 검증 결과 요약 생성
- [ ] 위반 사항 상세 리포트 작성
- [ ] 심각도별 우선순위 결정
- [ ] 후속 조치 계획 수립
- [ ] 검증 리포트 공유

### 7.4 품질 기준

**Level 1: Excellent (우수)**
- ✅ 모든 필드 검증 통과
- ✅ 참조 무결성 100% 유지
- ✅ 비즈니스 규칙 위반 < 1%

**Level 2: Good (양호)**
- ✅ 필드 검증 통과율 > 95%
- ✅ 참조 무결성 > 98%
- ✅ 비즈니스 규칙 위반 < 5%

**Level 3: Acceptable (허용)**
- ⚠️ 필드 검증 통과율 > 90%
- ⚠️ 참조 무결성 > 95%
- ⚠️ 비즈니스 규칙 위반 < 10%

**Level 4: Poor (부족)**
- ❌ 필드 검증 통과율 < 90%
- ❌ 참조 무결성 < 95%
- ❌ 비즈니스 규칙 위반 > 10%

---

## 8. 트러블슈팅 (Troubleshooting)

### 8.1 일반적 오류

**오류 1: 날짜 형식 불일치**
```python
# 문제
df['date'] = pd.to_datetime(df['date'])  # ValueError

# 해결
df['date'] = pd.to_datetime(df['date'], errors='coerce', format='%Y-%m-%d')
# 또는 자동 포맷 감지
df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
```

**오류 2: 부동소수점 비교 오류**
```python
# 문제
df['calculated'] == df['actual']  # False negatives due to floating point precision

# 해결
np.abs(df['calculated'] - df['actual']) < 0.01  # 허용 오차 사용
# 또는
np.isclose(df['calculated'], df['actual'], atol=0.01)
```

**오류 3: NULL 값 처리**
```python
# 문제
df['amount'] > 0  # NaN은 False 반환

# 해결
df['amount'].notna() & (df['amount'] > 0)
```

### 8.2 성능 최적화

**문제: 대용량 데이터 검증 속도 저하**

```python
# 해결책 1: 청크 단위 처리
def validate_in_chunks(df, chunk_size=10000):
    """
    대용량 데이터를 청크로 나눠 검증
    """
    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    results = []
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        result = validate_chunk(chunk)
        results.append(result)
    
    return pd.concat(results)


# 해결책 2: 벡터화 연산 사용
# 느림
def slow_validation(df):
    violations = []
    for idx, row in df.iterrows():
        if row['quantity'] * row['price'] != row['total']:
            violations.append(idx)
    return violations

# 빠름
def fast_validation(df):
    mask = np.abs(df['quantity'] * df['price'] - df['total']) > 0.01
    return df[mask].index.tolist()


# 해결책 3: 병렬 처리
from multiprocessing import Pool

def parallel_validation(df, n_cores=4):
    """
    병렬 처리로 검증 속도 향상
    """
    chunks = np.array_split(df, n_cores)
    
    with Pool(n_cores) as pool:
        results = pool.map(validate_chunk, chunks)
    
    return pd.concat(results)
```

### 8.3 메모리 관리

**문제: 대용량 데이터 메모리 부족**

```python
# 해결책 1: 데이터 타입 최적화
def optimize_dtypes(df):
    """
    데이터 타입 최적화로 메모리 절약
    """
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')
    
    return df


# 해결책 2: 불필요한 컬럼 제거
def validate_minimal_columns(df, required_columns):
    """
    검증에 필요한 컬럼만 로드
    """
    df_minimal = df[required_columns].copy()
    return validate(df_minimal)


# 해결책 3: 청크 읽기
def validate_from_file(filepath, chunk_size=10000):
    """
    파일을 청크로 읽으면서 검증
    """
    violations = []
    
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        chunk_violations = validate_chunk(chunk)
        violations.append(chunk_violations)
    
    return pd.concat(violations)
```

### 8.4 Great Expectations 관련 이슈

**문제: Context 초기화 실패**
```python
# 해결
import great_expectations as gx

# 새 프로젝트 초기화
context = gx.get_context(mode="file")

# 또는 기존 프로젝트 사용
context = gx.get_context(context_root_dir="/path/to/gx/directory")
```

**문제: Expectation Suite 저장 실패**
```python
# 해결
validator.save_expectation_suite(
    discard_failed_expectations=False,
    overwrite_existing=True
)
```

---

## 9. 참고 자료 (References)

### 9.1 공식 문서

**Great Expectations**
- 공식 문서: https://docs.greatexpectations.io/
- GitHub: https://github.com/great-expectations/great_expectations
- 튜토리얼: https://docs.greatexpectations.io/docs/tutorials/

**Pandera**
- 공식 문서: https://pandera.readthedocs.io/
- GitHub: https://github.com/unionai-oss/pandera
- 예시: https://pandera.readthedocs.io/en/stable/examples.html

**Pandas**
- Data Validation Guide: https://pandas.pydata.org/docs/user_guide/indexing.html
- Boolean Indexing: https://pandas.pydata.org/docs/user_guide/indexing.html#boolean-indexing

### 9.2 베스트 프랙티스

**Data Quality Frameworks**
- Data Quality Dimensions: https://www.dataversity.net/six-key-data-quality-dimensions/
- Data Validation Best Practices: https://www.kdnuggets.com/2021/01/data-validation-best-practices.html

**Database Integrity**
- Referential Integrity: https://en.wikipedia.org/wiki/Referential_integrity
- Foreign Key Constraints: https://www.postgresql.org/docs/current/ddl-constraints.html

### 9.3 관련 레퍼런스

**Data-cleansing Skill 레퍼런스**:
- `01-data-quality-assessment.md`: 데이터 품질 평가
- `02-missing-data-patterns.md`: 결측값 패턴 분석
- `12-quality-reporting.md`: 품질 리포트 생성
- `13-data-lineage.md`: 데이터 리니지 추적

**Workflow 매핑**:
- `data-cleansing-workflow.md` Phase 6 (lines 1183-1308)
  - Section 6.1: 교차 필드 검증
  - Section 6.2: 참조 무결성 검증

---

## 마무리 (Conclusion)

데이터 검증은 데이터 클렌징의 최종 단계로, 데이터 품질을 보장하는 가장 중요한 프로세스입니다. 이 레퍼런스에서 다룬 교차 필드 검증, 참조 무결성 검증, 비즈니스 로직 검증을 체계적으로 적용하면 고품질 데이터를 확보할 수 있습니다.

**핵심 원칙**:
1. **계층적 검증**: Field → Cross-Field → Record → Dataset 순서
2. **자동화**: Great Expectations, Pandera 활용
3. **명확한 규칙**: 비즈니스 규칙 문서화
4. **심각도 분류**: HIGH/MEDIUM/LOW 우선순위
5. **재현성**: 모든 검증 스크립트 버전 관리

**다음 단계**:
- 검증 통과 시: `12-quality-reporting.md`로 품질 리포트 생성
- 검증 실패 시: 각 Phase 레퍼런스로 돌아가 데이터 클렌징 재수행
- 자동화: `15-automation-pipeline.md`로 검증 파이프라인 구축

---

**작성자**: Claude Code  
**최종 수정일**: 2025-01-26  
**버전**: 1.0
