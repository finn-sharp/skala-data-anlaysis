# -*- coding: utf-8 -*-
"""성인 소득(Adult Income) 데이터 분석 — 통합 진입점(End2End 파이프라인).

팀원별로 나뉘어 있던 단계를 하나의 실행 흐름으로 묶는다. 각 단계의 실제 로직은
src/ 아래 모듈에 있고, 이 파일은 순서 오케스트레이션과 최상위 예외 가드만 갖는다.

■ 파이프라인 단계
    [1] 데이터 적재 — Pandas vs Polars 로딩 비교 + 로딩 속도 벤치마크   (src.compare, 정한결)
    [2] 결측치·중복 처리 + 기본 EDA                                    (src.data, 정한결)
    [3] 시각화 — Seaborn · Plotly · Cramér's V 히트맵                   (src.eda, 조현강)
    [4] 통계검정(t·카이제곱·ANOVA) + ML 모델 비교(KNN·Logistic·XGBoost)  (src.stats, 김재현)

    ※ [5] report.md 자동 생성은 report-kdi.py(길다인)로 별도 실행한다.

■ 데이터
    팀 표준 입력은 헤더 있는 data/adult.csv 하나다. 각 단계는 필요한 시점에 이 파일을
    직접 읽어 자체 정제하므로 단계 간 상태 공유 없이 독립적으로 동작한다.

■ 실행
    $ python main.py     # 저장소 루트에서 실행
"""

import json
import os
import sys

from src.config import (BENCHMARK_DATA_PATH, BENCHMARK_MULTIPLIER,
                        DATA_PATH, OUT_DIR)
from src.data import clean_data, clean_data_polars, run_eda
from src.exceptions import DataError
from src.compare import benchmark_loaders, build_benchmark_file, compare_loaders
from src.utils import section
from src import eda, stats


def stage_data_prep() -> None:
    """[1] 적재·로더 비교·속도 벤치마크 → [2] 정제(결측·중복)·기본 EDA 를 실행한다(정한결)."""
    os.makedirs(OUT_DIR, exist_ok=True)

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
    with open(os.path.join(OUT_DIR, "loader_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(loader_report, f, ensure_ascii=False, indent=2)
    print(f"* 저장: {os.path.join(OUT_DIR, 'loader_comparison.json')}")

    # ---- [1-B] 로딩 속도 벤치마크 (원본 N배 복제 파일, 속도 비교 전용) --------
    section(f"[1-B] 로딩 속도 벤치마크 (원본 x{BENCHMARK_MULTIPLIER}, 속도 비교 전용)")
    if os.path.isfile(BENCHMARK_DATA_PATH):
        print(f"* 벤치마크 파일 이미 존재 — 재생성 생략: {BENCHMARK_DATA_PATH}")
    else:
        build_benchmark_file(DATA_PATH, BENCHMARK_DATA_PATH, multiplier=BENCHMARK_MULTIPLIER)
        print(f"* 벤치마크 파일 생성: {BENCHMARK_DATA_PATH}")
    benchmark_report = benchmark_loaders(BENCHMARK_DATA_PATH, n_runs=3)
    print(f"* {benchmark_report['rows']:,}행 / {benchmark_report['size_mb']}MB  →  "
          f"{benchmark_report['msg']}")
    with open(os.path.join(OUT_DIR, "benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, ensure_ascii=False, indent=2)

    # ---- [2] 결측치·중복 처리 (Pandas 기준 + Polars 교차 검증) --------------
    section("[2] 결측치·중복 처리")
    df_clean, clean_report = clean_data(df_pd_raw)
    print(f"* [Pandas] 원본 {clean_report['n_before']:,}행 -> 정제 {clean_report['n_after']:,}행 "
          f"(결측 {clean_report['n_missing_dropped']:,} · 중복 {clean_report['n_dup_dropped']:,} 제거)")
    print(f"  income 분포: {clean_report['income_dist']}  (고소득 비율 {clean_report['pos_rate']:.3f})")
    _, clean_report_pl = clean_data_polars(df_pl_raw)
    same = clean_report["n_after"] == clean_report_pl["n_after"]
    print(f"* [Polars] 정제 후 {clean_report_pl['n_after']:,}행  |  Pandas와 일치: {same}")

    # ---- [2] 기본 EDA -------------------------------------------------------
    section("[2] 기본 EDA (df.info / describe 상당)")
    eda_report = run_eda(df_clean)
    print(f"* shape: {eda_report['shape']}  |  결측 총합: {eda_report['missing_total']}  "
          f"|  중복 행: {eda_report['duplicate_rows']}")
    print("* 타깃(income) 분포:", eda_report["target_distribution"])
    with open(os.path.join(OUT_DIR, "eda_summary.json"), "w", encoding="utf-8") as f:
        json.dump(eda_report, f, ensure_ascii=False, indent=2)
    df_clean.to_csv(os.path.join(OUT_DIR, "adult_clean.csv"), index=False)
    print("* 저장: eda_summary.json · adult_clean.csv")


def main() -> None:
    """[1]~[4] 전체 파이프라인을 순서대로 실행한다."""
    stage_data_prep()                                    # [1][2] 데이터 준비 (정한결)

    section("[3] 시각화 — Seaborn · Plotly · Cramér's V")
    eda.main()                                           # [3] 시각화 (조현강)

    section("[4] 통계검정 + ML 모델 비교")
    stats.main()                                         # [4] 통계·ML (김재현)

    section("[완료] 통합 파이프라인 종료 — outputs/ 확인")
    print("  ※ report.md는 `python report-kdi.py`로 별도 생성하세요.")


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
