# -*- coding: utf-8 -*-
"""[1] 데이터 적재 — Pandas vs Polars 로딩 결과 비교.

■ 이 모듈이 하는 일
    1) 동일한 원본(adult.data)을 Pandas와 Polars 양쪽으로 각각 로딩해 결과를 비교한다
       (행/열 수, dtype, 결측 개수, 로딩 시간, 메모리 사용량).
    2) 원본(32,561행)만으로는 두 엔진의 로딩 속도 차이가 거의 안 나서, 로딩 속도
       비교 전용으로 원본을 N배 복제한 대용량 CSV를 만들고(build_benchmark_file),
       그 파일로 반복 측정해 평균 로딩 시간을 비교한다(benchmark_loaders).

■ 설계 메모
    - Pandas가 파이프라인의 기준(reference) 엔진이다. compare_loaders()가 반환하는
      df_pandas가 다음 단계(data.py의 clean_data())의 공통 입력이 된다.
    - Polars 쪽은 Pandas의 skipinitialspace에 대응하는 옵션이 없고, 선행 공백이 붙은
      숫자 문자열은 자동 dtype 추론이 실패해 문자열로 남는 문제가 있어 별도 후처리가
      필요하다(load_data_polars 참고).
    - 복제 벤치마크 파일(adult_x25.csv 등)은 로딩 속도 비교 **전용**이다. 복제된 행이
      전부 중복으로 잡혀 정제하면 원본 크기로 줄어들어버리므로, 결측치·중복
      처리/EDA/통계/ML Pipeline 단계에는 절대 쓰지 않는다 — 그 단계들은 계속
      원본 adult.data를 그대로 사용한다.
"""

import os
import time

import pandas as pd
import polars as pl

from .config import COLUMNS, RAW_NA, RAW_NUMERIC_COLUMNS
from .exceptions import DataError


# ============================================================================ #
# 1-A) 적재 — Pandas
# ============================================================================ #
def load_data_pandas(csv_path: str) -> pd.DataFrame:
    """헤더 없는 adult.data를 Pandas로 읽어 규격 컬럼명을 부여한 DataFrame으로 반환한다.

    원본은 콤마 뒤에 공백이 붙는 형식(skipinitialspace)이며 결측은 '?'로 표기된다.
    적재 시점에 공백을 제거하고 '?'를 NaN으로 변환해, 이후 단계가 표준 결측 처리를
    그대로 쓸 수 있게 한다.

    Args:
        csv_path: adult.data 파일 경로.

    Returns:
        15개 컬럼이 부여된 원본 DataFrame.

    Raises:
        DataError: 파일이 없거나, 컬럼 수가 규격과 다르거나, 빈 데이터일 때.
    """
    if not os.path.isfile(csv_path):
        raise DataError(f"데이터 파일을 찾을 수 없음: {csv_path}")

    df = pd.read_csv(csv_path, header=None, names=COLUMNS,
                      skipinitialspace=True, na_values=RAW_NA)
    if df.empty:
        raise DataError("적재 결과가 비어 있음(0행)")
    if df.shape[1] != len(COLUMNS):
        raise DataError(f"컬럼 수 불일치: 기대 {len(COLUMNS)}, 실제 {df.shape[1]}")
    return df


# ============================================================================ #
# 1-B) 적재 — Polars
# ============================================================================ #
def load_data_polars(csv_path: str) -> pl.DataFrame:
    """헤더 없는 adult.data를 Polars로 읽어 규격 컬럼명을 부여한 DataFrame으로 반환한다.

    Polars의 CSV 리더에는 Pandas의 skipinitialspace에 대응하는 옵션이 없다. 게다가
    선행 공백이 붙은 숫자 문자열(예: " 77516")은 Polars의 자동 dtype 추론이 파싱에
    실패해 그 컬럼 전체를 문자열로 남겨버린다. 이를 피하려고 모든 컬럼을 일단
    문자열(Utf8)로 강제 로딩한 뒤, 공백 제거 → '?' 결측 치환 → 수치형 컬럼만
    명시적으로 캐스팅하는 순서로 처리한다. 또한 원본 파일 끝에 있는 빈 줄(trailing
    blank line)은 Pandas는 기본으로 건너뛰지만 Polars는 전체 컬럼이 null인 행으로
    읽어들이므로, 적재 직후 완전 결측 행을 제거해 두 엔진의 행 수를 맞춘다.

    Args:
        csv_path: adult.data 파일 경로.

    Returns:
        15개 컬럼이 부여된 원본 DataFrame(polars), 수치형 컬럼은 Int64로 캐스팅됨.

    Raises:
        DataError: 파일이 없거나, 컬럼 수가 규격과 다르거나, 빈 데이터일 때.
    """
    if not os.path.isfile(csv_path):
        raise DataError(f"데이터 파일을 찾을 수 없음: {csv_path}")

    schema_overrides = {c: pl.Utf8 for c in COLUMNS}
    df = pl.read_csv(csv_path, has_header=False, new_columns=COLUMNS,
                      schema_overrides=schema_overrides)

    # 원본 말미의 빈 줄 등, 전체 컬럼이 null인 행 제거 (Pandas의 skip_blank_lines 기본값과 동치)
    df = df.filter(~pl.all_horizontal(pl.all().is_null()))

    df = df.with_columns([pl.col(c).str.strip_chars() for c in COLUMNS])
    df = df.with_columns([
        pl.when(pl.col(c) == RAW_NA).then(None).otherwise(pl.col(c)).alias(c)
        for c in COLUMNS
    ])
    df = df.with_columns([pl.col(c).cast(pl.Int64) for c in RAW_NUMERIC_COLUMNS])

    if df.is_empty():
        raise DataError("적재 결과가 비어 있음(0행)")
    if df.width != len(COLUMNS):
        raise DataError(f"컬럼 수 불일치: 기대 {len(COLUMNS)}, 실제 {df.width}")
    return df


# ============================================================================ #
# 1-C) Pandas vs Polars 로딩 결과 비교
# ============================================================================ #
def compare_loaders(csv_path: str):
    """동일 원본을 Pandas·Polars로 각각 로딩해 결과를 비교한다.

    비교 항목: 행/열 수 일치 여부, 로딩 소요 시간, 컬럼별 dtype, 컬럼별 결측
    개수 일치 여부, 메모리 사용량(MB). dtype은 두 엔진이 서로 다른 이름 체계를
    쓰므로(예: object vs String, int64 vs Int64) 문자열로 캐스팅해 나란히 비교한다.

    Args:
        csv_path: adult.data 파일 경로.

    Returns:
        (report, df_pandas, df_polars) 튜플.
        report는 아래 키를 가진 dict:
            shape_pandas, shape_polars, same_shape,
            load_time_pandas_sec, load_time_polars_sec,
            dtypes_pandas, dtypes_polars,
            missing_by_col_pandas, missing_by_col_polars, same_missing_counts,
            memory_mb_pandas, memory_mb_polars

    Raises:
        DataError: load_data_pandas / load_data_polars가 실패할 때 그대로 전파.
    """
    t0 = time.perf_counter()
    df_pd = load_data_pandas(csv_path)
    t_pandas = time.perf_counter() - t0

    t0 = time.perf_counter()
    df_pl = load_data_polars(csv_path)
    t_polars = time.perf_counter() - t0

    dtypes_pd = {c: str(t) for c, t in df_pd.dtypes.items()}
    dtypes_pl = {c: str(t) for c, t in zip(df_pl.columns, df_pl.dtypes)}

    missing_pd = {c: int(v) for c, v in df_pd.isna().sum().items() if v > 0}
    missing_pl = {c: int(v) for c, v in zip(df_pl.columns, df_pl.null_count().row(0))
                  if v > 0}

    mem_pd = float(df_pd.memory_usage(deep=True).sum() / 1024 ** 2)
    mem_pl = float(df_pl.estimated_size("mb"))

    report = {
        "shape_pandas": df_pd.shape,
        "shape_polars": df_pl.shape,
        "same_shape": df_pd.shape == df_pl.shape,
        "load_time_pandas_sec": round(t_pandas, 4),
        "load_time_polars_sec": round(t_polars, 4),
        "dtypes_pandas": dtypes_pd,
        "dtypes_polars": dtypes_pl,
        "missing_by_col_pandas": missing_pd,
        "missing_by_col_polars": missing_pl,
        "same_missing_counts": missing_pd == missing_pl,
        "memory_mb_pandas": round(mem_pd, 2),
        "memory_mb_polars": round(mem_pl, 2),
    }
    return report, df_pd, df_pl


# ============================================================================ #
# 1-D) 로딩 속도 벤치마크 — 원본은 너무 작아서 N배 복제한 대용량 파일로 측정
# ============================================================================ #
def build_benchmark_file(csv_path: str, out_path: str, multiplier: int = 25) -> str:
    """원본을 multiplier배 복제한 대용량 CSV를 만들어 로딩 속도 벤치마크용으로 저장한다.

    원본(32,561행)은 두 엔진의 로딩 속도 차이가 거의 안 나서(수 ms 수준), "데이터가
    커지면 Polars가 더 빠르다"는 차이를 보여주려면 훨씬 큰 파일이 필요하다. 정제된
    Pandas DataFrame(load_data_pandas 결과)을 그대로 이어붙여 저장하므로, 이 벤치마크
    파일은 헤더 있는 표준 CSV 형식이다(원본의 헤더 없음·콤마+공백 형식과는 다름).

    이 파일은 로딩 속도 비교 **전용**이다. 복제된 행이 전부 완전 중복이라 정제하면
    원본 크기로 줄어들어버리므로, 결측치·중복 처리/EDA/통계/ML Pipeline 단계에는
    이 파일을 쓰지 않고 원본 adult.data를 그대로 쓴다.

    Args:
        csv_path: 원본 adult.data 경로.
        out_path: 생성할 벤치마크 CSV 경로.
        multiplier: 복제 배수(기본 25배 — 원본 32,561행 → 약 81.4만 행).

    Returns:
        생성된 벤치마크 CSV 경로.

    Raises:
        DataError: load_data_pandas가 실패할 때 그대로 전파.
    """
    df = load_data_pandas(csv_path)
    df_big = pd.concat([df] * multiplier, ignore_index=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_big.to_csv(out_path, index=False)
    return out_path


def benchmark_loaders(csv_path: str, n_runs: int = 3) -> dict:
    """대용량 CSV(build_benchmark_file 결과)로 Pandas·Polars 로딩 속도를 반복 측정한다.

    정교한 프로파일링이 아니라 "데이터가 커지면 Polars가 Pandas보다 빠르다"는 경향
    하나를 보여주는 게 목적이라, 캐시 효과를 줄이려 n_runs번 반복해 평균을 낸다.
    벤치마크 파일은 build_benchmark_file()이 표준 CSV(헤더 있음)로 저장했으므로,
    원본 로딩 때 필요했던 skipinitialspace·na_values 같은 옵션 없이 기본 read_csv로
    읽는다.

    Args:
        csv_path: 벤치마크용 CSV 경로(build_benchmark_file의 반환값).
        n_runs: 반복 측정 횟수(기본 3회).

    Returns:
        dict(n_runs, rows, size_mb, pandas_times_sec, polars_times_sec,
             pandas_avg_sec, polars_avg_sec, speedup_x, msg).

    Raises:
        DataError: 벤치마크 파일이 없을 때.
    """
    if not os.path.isfile(csv_path):
        raise DataError(f"벤치마크 파일을 찾을 수 없음: {csv_path}")

    pandas_times, polars_times = [], []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        df_pd = pd.read_csv(csv_path)
        pandas_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        pl.read_csv(csv_path)
        polars_times.append(time.perf_counter() - t0)

    pandas_avg = sum(pandas_times) / n_runs
    polars_avg = sum(polars_times) / n_runs
    speedup = pandas_avg / polars_avg if polars_avg > 0 else float("inf")

    msg = (f"{n_runs}회 평균 — Pandas {pandas_avg:.3f}s vs Polars {polars_avg:.3f}s "
           f"(Polars가 약 {speedup:.2f}배 빠름)")
    return {
        "n_runs": n_runs,
        "rows": int(df_pd.shape[0]),
        "size_mb": round(os.path.getsize(csv_path) / 1024 ** 2, 2),
        "pandas_times_sec": [round(t, 4) for t in pandas_times],
        "polars_times_sec": [round(t, 4) for t in polars_times],
        "pandas_avg_sec": round(pandas_avg, 4),
        "polars_avg_sec": round(polars_avg, 4),
        "speedup_x": round(speedup, 2),
        "msg": msg,
    }
