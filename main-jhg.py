# -*- coding: utf-8 -*-
"""성인 소득 데이터 분석 프로젝트 — 진입점.

현재 구현 범위: [1] 데이터 적재(Pandas·Polars 비교 + 로딩 속도 벤치마크)
+ [2] 정제(결측·중복 처리) + 기본 EDA.
나머지 단계(시각화·통계검정·ML Pipeline·대시보드·자동 리포트)는 이후 커밋에서
src/eda.py, src/cda.py, src/model.py, src/dashboard.py, src/report.py로 추가된다.

이 파일은 오케스트레이션과 최상위 예외 가드만 갖는다. 실제 로직은 전부 src/ 아래
모듈의 순수 함수에 있다.
"""

import json
import os
import sys

from src.config import BENCHMARK_DATA_PATH, BENCHMARK_MULTIPLIER, DATA_PATH, OUT_PATH
from src.data import clean_data, clean_data_polars, run_eda
from src.exceptions import DataError
from src.compare import benchmark_loaders, build_benchmark_file, compare_loaders
from src.utils import section


def main() -> None:
    """[1] 적재+로더 비교+속도 벤치마크 → [2] 정제(결측·중복) → 기본 EDA 를 순서대로 실행한다."""
    os.makedirs(OUT_PATH, exist_ok=True)

    # ---- [1] Pandas vs Polars 로딩 결과 비교 -------------------------------
    section("[1] 데이터 적재 — Pandas vs Polars 비교")
    loader_report, df_pd_raw, df_pl_raw = compare_loaders(DATA_PATH)
    print(f"* Pandas  shape={loader_report['shape_pandas']}  "
          f"로딩 {loader_report['load_time_pandas_sec']}s  "
          f"메모리 {loader_report['memory_mb_pandas']}MB")
    print(f"* Polars  shape={loader_report['shape_polars']}  "
          f"로딩 {loader_report['load_time_polars_sec']}s  "
          f"메모리 {loader_report['memory_mb_polars']}MB")
    print(f"* shape 일치: {loader_report['same_shape']}  |  "
          f"컬럼별 결측 개수 일치: {loader_report['same_missing_counts']}")
    print("* dtype 비교 (Pandas -> Polars):")
    for col in df_pd_raw.columns:
        print(f"    - {col:16s} {loader_report['dtypes_pandas'][col]:10s} -> "
              f"{loader_report['dtypes_polars'][col]}")
    print("* 컬럼별 결측 개수 (Pandas):", loader_report["missing_by_col_pandas"])
    print("* 컬럼별 결측 개수 (Polars):", loader_report["missing_by_col_polars"])

    with open(os.path.join(OUT_PATH, "loader_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(loader_report, f, ensure_ascii=False, indent=2)
    print(f"* 저장: {os.path.join(OUT_PATH, 'loader_comparison.json')}")

    # ---- [1-B] 로딩 속도 벤치마크 (원본 N배 복제 파일, 속도 비교 전용) --------
    section("[1-B] 로딩 속도 벤치마크 (원본 x{}, 속도 비교 전용)".format(BENCHMARK_MULTIPLIER))
    if os.path.isfile(BENCHMARK_DATA_PATH):
        print(f"* 벤치마크 파일 이미 존재 — 재생성 생략: {BENCHMARK_DATA_PATH}")
    else:
        build_benchmark_file(DATA_PATH, BENCHMARK_DATA_PATH, multiplier=BENCHMARK_MULTIPLIER)
        print(f"* 벤치마크 파일 생성: {BENCHMARK_DATA_PATH}")
    benchmark_report = benchmark_loaders(BENCHMARK_DATA_PATH, n_runs=3)
    print(f"* {benchmark_report['rows']:,}행 / {benchmark_report['size_mb']}MB, "
          f"{benchmark_report['n_runs']}회 반복 측정")
    print(f"  Pandas 개별: {benchmark_report['pandas_times_sec']}")
    print(f"  Polars 개별: {benchmark_report['polars_times_sec']}")
    print(f"  {benchmark_report['msg']}")

    with open(os.path.join(OUT_PATH, "benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, ensure_ascii=False, indent=2)
    print(f"* 저장: {os.path.join(OUT_PATH, 'benchmark.json')}")

    # ---- [2] 결측치·중복 처리 (Pandas 기준 + Polars 교차 검증) --------------
    section("[2] 결측치·중복 처리")
    df_clean, clean_report = clean_data(df_pd_raw)
    print(f"* [Pandas] 원본 {clean_report['n_before']:,}행 -> 정제 {clean_report['n_after']:,}행 "
          f"(결측 {clean_report['n_missing_dropped']:,} · 중복 {clean_report['n_dup_dropped']:,} 제거)")
    print(f"  컬럼별 결측 개수: {clean_report['missing_by_col']}")
    print(f"  income 분포: {clean_report['income_dist']}  (고소득 비율 {clean_report['pos_rate']:.3f})")

    _, clean_report_pl = clean_data_polars(df_pl_raw)
    print(f"* [Polars] 원본 {clean_report_pl['n_before']:,}행 -> 정제 {clean_report_pl['n_after']:,}행 "
          f"(결측 {clean_report_pl['n_missing_dropped']:,} · 중복 {clean_report_pl['n_dup_dropped']:,} 제거)")
    same_after_clean = clean_report["n_after"] == clean_report_pl["n_after"]
    print(f"  Pandas·Polars 정제 후 행 수 일치: {same_after_clean}")
    if not same_after_clean:
        print("  [주의] 두 엔진의 정제 결과가 다름 — 원인 확인 필요"
              "(중복 판정 기준, 문자열 정규화 차이 등)")

    # ---- [2] 기본 EDA -------------------------------------------------------
    section("[2] 기본 EDA (df.info / describe 상당)")
    eda_report = run_eda(df_clean)
    print(f"* shape: {eda_report['shape']}")
    print(f"* 결측 총합: {eda_report['missing_total']}  |  중복 행: {eda_report['duplicate_rows']}")
    print("* 수치형 기술통계 (age 예시):")
    for stat, val in eda_report["numeric_describe"]["age"].items():
        print(f"    - {stat:6s}: {val:.2f}")
    print("* 범주형 최빈값 상위 (education 예시):",
          eda_report["categorical_top_values"]["education"])
    print("* 타깃(income) 분포:", eda_report["target_distribution"])

    with open(os.path.join(OUT_PATH, "eda_summary.json"), "w", encoding="utf-8") as f:
        json.dump(eda_report, f, ensure_ascii=False, indent=2)
    print(f"* 저장: {os.path.join(OUT_PATH, 'eda_summary.json')}")

    clean_csv = os.path.join(OUT_PATH, "adult_clean.csv")
    df_clean.to_csv(clean_csv, index=False)
    print(f"* 저장(정제 데이터, 다음 단계 공용 입력): {clean_csv}")

    section("[완료] 데이터 준비 단계 종료")
    print("  다음 단계(시각화·통계검정·ML Pipeline·대시보드·리포트)는 src/eda.py 이후 모듈에서 이어진다.")


if __name__ == "__main__":
    # 최상위 방어선: 종료 여부 판단은 이 진입점에서만 한다(함수는 예외를 raise할 뿐).
    #  - DataError        : 예상된 입력/정제/검증 실패 -> 원인 메시지만 간결히 출력
    #  - KeyboardInterrupt: 사용자 중단
    #  - 그 외 Exception  : 예상 밖 오류를 위한 마지막 안전망
    try:
        main()
    except DataError as e:
        sys.exit(f"[치명] {e}")
    except KeyboardInterrupt:
        sys.exit("\n[중단] 사용자에 의해 중단됨")
    except Exception as e:  # noqa: BLE001 - 최상위 포괄 예외 처리
        sys.exit(f"[치명] 예기치 못한 오류: {e}")
