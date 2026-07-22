# 데이터 준비 단계(1~2번) 완료 — 팀 공유용 정리

작성: 한결 · 2026-07-21

## 완료 상태

- **1번 — Pandas/Polars 로딩 결과 비교**: `src/compare.py`
- **2번 — 결측치·중복 처리 + 기본 EDA**: `src/data.py`

두 항목 모두 실행 검증 완료(아래 "실행 결과 요약" 참고).


## 중요사항
Main-jhg.py로 실행

25배 데이터는 코드 실행시 제작 (/data에 저장)

Src 파일 내 코드는 되도록 수정안하는게 좋음
 

## 전달 파일 목록

핵심 로직은 `src/compare.py`, `src/data.py`, `main.py` 세 개지만, 이 셋만 넘기면
`ImportError`가 남 — 아래 파일들이 없으면 실행이 안 됨(코드에서 그대로 import해서 씀):

| 파일 | 왜 필요한가 |
|---|---|
| `src/compare.py` | [1] 로딩 비교 로직 (핵심) |
| `src/data.py` | [2] 정제·EDA 로직 (핵심) |
| `main.py` | 진입점 (핵심) |
| `src/config.py` | 컬럼명·경로·TARGET 등 상수 — compare.py·data.py·main.py 전부 사용 |
| `src/exceptions.py` | `DataError` — compare.py·data.py·main.py 전부 사용 |
| `src/utils.py` | `section()` 로그 함수 — main.py에서 사용 |
| `src/__init__.py` | 패키지 인식용 (내용은 비어있어도 되지만 파일 자체가 있어야 함) |
| `requirements.txt` | pandas·polars 정확한 버전 (polars는 기본 설치가 안 돼 있을 수 있음) |
| `data/adult.data` | 원본 데이터 (팀원이 이미 갖고 있으면 생략 가능) |

## 파일 구조

```
skala-data-final/
├── main.py                     # 진입점 (main() + __main__ 3단 예외 가드만)
├── report_template.j2          # (5번용, 아직 미사용)
├── requirements.txt
├── data/
│   └── adult.data
├── outputs/                    # main.py 실행 시 자동 생성
│   ├── loader_comparison.json
│   ├── eda_summary.json
│   └── adult_clean.csv         # 정제된 데이터 — 3~6번의 공통 입력
└── src/
    ├── __init__.py
    ├── config.py                # BASE_DIR·컬럼명·TARGET·NUM/CAT_FEATURES 상수
    ├── exceptions.py            # DataError
    ├── utils.py                 # section(), set_korean_font()
    ├── compare.py    [1] 완료   # load_data_pandas / load_data_polars / compare_loaders
    ├── data.py       [2] 완료   # clean_data / clean_data_polars / run_eda
    ├── eda.py        [3] 미착수 # Seaborn·Plotly 시각화
    ├── cda.py        [4] 미착수 # 기술통계·상관계수·t검정 등 통계검정
    ├── model.py      [5] 미착수 # sklearn Pipeline 학습·평가·joblib 저장
    ├── dashboard.py  [6] 미착수 # Plotly 인터랙티브 대시보드
    └── report.py     [6] 미착수 # Jinja2 자동 리포트(report.md/html)
```


```

## 모듈 인터페이스 (다음 단계에서 그대로 가져다 쓰면 됨)

```python
from src.compare import compare_loaders
from src.data import clean_data, clean_data_polars, run_eda

report, df_pandas, df_polars = compare_loaders(DATA_PATH)   # [1] 로딩+비교
df_clean, clean_report = clean_data(df_pandas)               # [2] 정제 (Pandas 기준)
eda_report = run_eda(df_clean)                                # [2] 기본 EDA
```

- `df_clean`: 결측·중복 제거된 DataFrame. **3~6번은 이 df_clean을 그대로 입력으로 받으면 됨**
  (재적재·재정제 불필요). income은 원본 레이블(`<=50K`/`>50K`) 그대로 유지되어 있고,
  이진화(0/1)는 아직 안 되어 있음 — 필요한 모듈(주로 model.py)에서 그때 처리하면 됨.
- 컬럼·타깃·피처 상수는 `src/config.py`에 이미 정의되어 있음: `TARGET`, `POSITIVE_LABEL`,
  `NUM_FEATURES`, `CAT_FEATURES`.

## 실행 결과 요약

| 항목 | 값 |
|---|---|
| 원본 shape | (32,561, 15) — Pandas·Polars 완전 일치 |
| 컬럼별 결측 | workclass 1,836 / occupation 1,843 / native-country 583 |
| 정제 후 shape | (30,139, 15) (결측 2,399 · 중복 23 제거) |
| income 분포 | `<=50K` 75.1% / `>50K` 24.9% (불균형 데이터 — 모델링 시 재현율·F1 같이 볼 것) |
| age 평균 | 38.4세 (표준편차 13.1, 범위 17~90) |

----------------------------------------------------------------------------------------------







# 실행 결과 — 1번(로딩 비교) · 2번(결측치·중복 처리 + 기본 EDA)

실행 파일: `main.py` · 데이터: `data/adult.data` (UCI Adult, 32,561행)

---

# [1] 데이터 적재 — Pandas vs Polars 비교

| 항목 | Pandas | Polars |
|---|---|---|
| shape | (32,561, 15) | (32,561, 15) |
| 로딩 시간 | 0.0276s | 0.0338s |
| 메모리 사용량 | 6.2 MB | 3.95 MB |

**shape 일치: True  ·  컬럼별 결측 개수 일치: True**

컬럼별 dtype 비교 (Pandas → Polars):

| 컬럼 | Pandas | Polars |
|---|---|---|
| age | int64 | Int64 |
| workclass | str | String |
| fnlwgt | int64 | Int64 |
| education | str | String |
| educational-num | int64 | Int64 |
| marital-status | str | String |
| occupation | str | String |
| relationship | str | String |
| race | str | String |
| gender | str | String |
| capital-gain | int64 | Int64 |
| capital-loss | int64 | Int64 |
| hours-per-week | int64 | Int64 |
| native-country | str | String |
| income | str | String |

컬럼별 결측 개수 (Pandas·Polars 동일):

| 컬럼 | 결측 개수 |
|---|---:|
| workclass | 1,836 |
| occupation | 1,843 |
| native-country | 583 |

저장 파일: `outputs/loader_comparison.json`

---

## [2] 결측치·중복 처리

| 엔진 | 원본 행수 | 정제 후 행수 | 결측 제거 | 중복 제거 |
|---|---:|---:|---:|---:|
| Pandas | 32,561 | 30,139 | 2,399 | 23 |
| Polars | 32,561 | 30,139 | 2,399 | 23 |

**Pandas·Polars 정제 후 행 수 일치: True**

income(타깃) 분포 — 정제 후:

| 레이블 | 건수 | 비율 |
|---|---:|---:|
| `<=50K` | 22,633 | 75.1% |
| `>50K` | 7,506 | 24.9% |

> 불균형 데이터(약 3:1) — 이후 모델링 단계에서 정확도만으로 평가하면 착시가 생길 수 있어
> 재현율·F1 등을 함께 봐야 함.

---

## [2] 기본 EDA (df.info / describe 상당)

- shape: (30,139, 15)
- 결측 총합: 0 · 중복 행: 0 (정제 완료 확인)

수치형 기술통계 — `age` 예시:

| 통계량 | 값 |
|---|---:|
| count | 30,139 |
| mean | 38.44 |
| std | 13.13 |
| min | 17.00 |
| 25% | 28.00 |
| 50% | 37.00 |
| 75% | 47.00 |
| max | 90.00 |

범주형 최빈값 상위 — `education` 예시:

| 학력 | 건수 |
|---|---:|
| HS-grad | 9,834 |
| Some-college | 6,669 |
| Bachelors | 5,042 |
| Masters | 1,626 |
| Assoc-voc | 1,307 |

저장 파일: `outputs/eda_summary.json`, `outputs/adult_clean.csv`(정제 데이터, 다음 단계 공용 입력)

---

## 요약

1·2번 모두 Pandas·Polars 두 엔진에서 shape·결측개수·정제후행수가 완전히 일치해 로딩·정제
로직의 정확성을 상호 검증했다. 정제 후 데이터는 결측·중복 0건이며, income 분포가 약
75:25로 불균형하다는 점이 이후 통계검정·모델링 설계에 반영되어야 한다.
