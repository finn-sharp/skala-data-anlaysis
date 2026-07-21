# -*- coding: utf-8 -*-
"""프로젝트 전역 설정 — 경로·컬럼·상수.

모든 경로는 '실행 위치'가 아니라 '이 파일의 위치(src/)'를 기준으로 해석한다.
BASE_DIR은 프로젝트 루트(skala-data-final/)를 가리키도록 한 단계 위로 잡는다.
어느 디렉터리에서 실행하든 data/·outputs/·report_template.j2를 안정적으로 찾기 위함이다.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "adult.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
TEMPLATE_PATH = os.path.join(BASE_DIR, "report_template.j2")

# 로딩 속도 벤치마크 전용 대용량 파일 — 원본(32,561행)만으로는 Pandas·Polars 로딩
# 속도 차이가 거의 안 나서, 이 배수로 복제한 별도 CSV를 만들어 속도 비교에만 쓴다.
# 결측치·중복 처리/EDA/통계/ML Pipeline 단계에는 이 파일을 쓰지 않는다.
BENCHMARK_DATA_PATH = os.path.join(BASE_DIR, "data", "adult_x25.csv")
BENCHMARK_MULTIPLIER = 25

# adult.data는 헤더가 없으므로 UCI 규격 순서대로 컬럼명을 직접 부여한다.
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "educational-num",
    "marital-status", "occupation", "relationship", "race", "gender",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]
RAW_NA = "?"  # 원본에서 결측을 표기하는 문자 (콤마 뒤 공백이 붙어 " ?"로도 나타남)

# 원본 스키마상 수치형 컬럼(fnlwgt·educational-num 포함, 모델 피처 여부와 무관).
# Polars 적재 시 문자열로 읽은 뒤 이 목록 기준으로 정확한 dtype으로 캐스팅한다.
RAW_NUMERIC_COLUMNS = ["age", "fnlwgt", "educational-num", "capital-gain",
                        "capital-loss", "hours-per-week"]

# 예측 대상(다음 단계 모델링용). income은 <=50K / >50K 두 범주다.
TARGET = "income"
POSITIVE_LABEL = ">50K"

# 수치형·범주형 피처 구분 (다음 단계 통계분석·모델링에서 사용 예정).
# fnlwgt(표본 가중치)는 예측력이 없어 피처에서 제외.
NUM_FEATURES = ["age", "educational-num", "capital-gain", "capital-loss",
                 "hours-per-week"]
CAT_FEATURES = ["workclass", "marital-status", "occupation", "relationship",
                 "race", "gender", "native-country"]
FEATURES = NUM_FEATURES + CAT_FEATURES
