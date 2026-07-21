# -*- coding: utf-8 -*-
"""[2] 결측치·중복 처리 + 기본 EDA.

■ 이 모듈이 하는 일
    1) 결측치(?)·중복 행을 처리해 분석용 정제 데이터셋을 만든다. 교차 검증 차원에서
       Pandas·Polars 두 엔진으로 각각 정제해 결과(정제 후 행 수)가 일치하는지도 확인한다.
    2) 정제된 데이터셋에 대해 기본 EDA(shape·dtype·기술통계·결측/중복 현황·범주형 빈도)를
       실행한다. 차트를 그리는 시각화는 다음 단계(eda.py)에서 다룬다.

■ 설계 메모
    - 이 모듈이 받는 입력(df)은 compare.py의 compare_loaders()가 반환한
      load_data_pandas() / load_data_polars() 결과다. 적재는 이 모듈의 책임이 아니다.
    - 다운스트림(통계검정·모델링)에서 쓸 income 이진화(0/1)는 이 모듈의 책임이 아니다.
      clean_data()는 '결측·중복 처리'까지만 하고, 사람이 읽기 쉬운 원본 레이블
      (<=50K / >50K)을 그대로 남겨 EDA가 바로 값 분포를 보여줄 수 있게 한다.
    - Pandas가 파이프라인의 기준(reference) 엔진이다. clean_data()의 반환값이
      이후 단계(eda.py, cda.py, model.py)의 공통 입력이 된다.
"""

import pandas as pd
import polars as pl

from .config import POSITIVE_LABEL, TARGET
from .exceptions import DataError


# ============================================================================ #
# 2-A) 정제 — Pandas (기준 엔진)
# ============================================================================ #
def clean_data(df: pd.DataFrame):
    """결측·중복을 제거한 분석용 데이터셋을 만든다(Pandas, 파이프라인 기준 엔진).

    처리 순서: 결측 행 제거 → 중복 행 제거. income은 <=50K/>50K 원본 레이블을
    그대로 유지한다(이진화는 model.py의 몫).

    Args:
        df: compare.load_data_pandas가 반환한 원본 DataFrame.

    Returns:
        (df_clean, report): 정제 DataFrame과 정제 통계 dict
        (n_before, n_after, n_missing_dropped, n_dup_dropped, missing_by_col,
         income_dist, pos_rate).

    Raises:
        DataError: 정제 후 남은 행이 없을 때.
    """
    df = df.copy()  # 원본 뷰 수정 경고(SettingWithCopy) 방지
    n_before = len(df)

    missing_by_col = {c: int(v) for c, v in df.isna().sum().items() if v > 0}
    df = df.dropna(how="any")
    n_after_na = len(df)

    df = df.drop_duplicates()
    n_after = len(df)

    if n_after == 0:
        raise DataError("정제 후 학습·분석에 쓸 표본이 없음(0행)")

    income_dist = {str(k): int(v) for k, v in df[TARGET].value_counts().items()}
    pos_rate = float((df[TARGET] == POSITIVE_LABEL).mean())

    report = {
        "n_before": n_before,
        "n_after": n_after,
        "n_missing_dropped": n_before - n_after_na,
        "n_dup_dropped": n_after_na - n_after,
        "missing_by_col": missing_by_col,
        "income_dist": income_dist,
        "pos_rate": pos_rate,  # 고소득(>50K) 비율
    }
    return df, report


# ============================================================================ #
# 2-B) 정제 — Polars (교차 검증용)
# ============================================================================ #
def clean_data_polars(df: pl.DataFrame):
    """결측·중복 제거를 Polars로도 수행해 Pandas 결과와 교차 검증한다.

    Pandas clean_data()와 동일한 처리 순서(결측 제거 → 중복 제거)를 적용한다.
    두 엔진의 정제 후 행 수가 일치하면 로딩·정제 로직이 엔진 차이 없이
    일관됨을 뜻한다.

    Args:
        df: compare.load_data_polars가 반환한 원본 DataFrame.

    Returns:
        (df_clean, report): 정제 DataFrame(polars)과 정제 통계 dict.

    Raises:
        DataError: 정제 후 남은 행이 없을 때.
    """
    n_before = df.height

    missing_by_col = {c: int(v) for c, v in zip(df.columns, df.null_count().row(0))
                       if v > 0}
    df = df.drop_nulls()
    n_after_na = df.height

    df = df.unique(maintain_order=True)
    n_after = df.height

    if n_after == 0:
        raise DataError("정제 후 학습·분석에 쓸 표본이 없음(0행)")

    income_dist = {row[TARGET]: row["count"] for row in
                   df.group_by(TARGET).len(name="count").to_dicts()}
    pos_rate = float((df[TARGET] == POSITIVE_LABEL).sum() / n_after)

    report = {
        "n_before": n_before,
        "n_after": n_after,
        "n_missing_dropped": n_before - n_after_na,
        "n_dup_dropped": n_after_na - n_after,
        "missing_by_col": missing_by_col,
        "income_dist": income_dist,
        "pos_rate": pos_rate,
    }
    return df, report


# ============================================================================ #
# 2-C) 기본 EDA — shape · dtype · 기술통계 · 결측/중복 현황 · 범주형 빈도
# ============================================================================ #
def run_eda(df: pd.DataFrame, top_n: int = 5) -> dict:
    """정제된 DataFrame에 대해 시각화 없는 기본 EDA를 실행한다.

    df.info()/describe()에 해당하는 정보를 dict로 구조화해, 로그 출력과
    JSON 저장 양쪽에 그대로 쓸 수 있게 한다. 차트가 필요한 분포·상관관계
    시각화는 다음 단계(eda.py)에서 별도로 다룬다.

    Args:
        df: clean_data()가 반환한 정제 DataFrame.
        top_n: 범주형 컬럼별로 함께 보여줄 최빈값 개수.

    Returns:
        dict(shape, dtypes, missing_total, duplicate_rows, numeric_describe,
             categorical_top_values, target_distribution).
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    numeric_describe = {
        col: {stat: float(val) for stat, val in df[col].describe().items()}
        for col in numeric_cols
    }
    categorical_top_values = {
        col: {str(k): int(v) for k, v in df[col].value_counts().head(top_n).items()}
        for col in categorical_cols
    }

    return {
        "shape": df.shape,
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_describe": numeric_describe,
        "categorical_top_values": categorical_top_values,
        "target_distribution": {str(k): int(v) for k, v in
                                 df[TARGET].value_counts().items()},
    }
