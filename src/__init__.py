# -*- coding: utf-8 -*-
"""skala-data-final 패키지 루트.

각 처리 단계를 모듈로 분리한다.
    compare   [1] 적재 (load_data_pandas / load_data_polars / compare_loaders)
    data      [2] 정제·기본 EDA (clean_data / clean_data_polars / run_eda)
    eda       [3] 시각화 EDA (추후 작성)
    cda       [4] 통계 검정 (추후 작성)
    model     [5] ML Pipeline (추후 작성)
    dashboard [6] Plotly 대시보드 (추후 작성)
    report    [7] 자동 리포트 (추후 작성)
"""
