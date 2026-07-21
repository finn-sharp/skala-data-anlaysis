# 성인 소득(Adult Income) 데이터 분석

미국 인구조사 기반 성인 소득 데이터로 **연소득 `>50K` 여부를 예측**하는 팀 분석 프로젝트입니다.
데이터 준비(Pandas·Polars) → 시각화(Seaborn·Plotly) → 통계 검정(t·카이제곱·ANOVA) → ML 파이프라인(sklearn·joblib) → 리포트 자동화(`report.md`)까지 수행합니다.

- **데이터**: `data/adult.csv` (UCI Adult / Census Income) — 32,561행 · 15컬럼
- **타깃**: `income` (`<=50K` / `>50K` 이진 분류)

---

## 목차
1. [프로젝트 구조](#1-프로젝트-구조)
2. [데이터셋 설명](#2-데이터셋-설명)
3. [팀원별 담당 · 정리 문서](#3-팀원별-담당--정리-문서)
4. [분석 파이프라인](#4-분석-파이프라인)
5. [실행 방법](#5-실행-방법)
6. [실행 결과 요약](#6-실행-결과-요약)

---

## 1. 프로젝트 구조

```
skala-data-anlaysis/
├── data/
│   └── adult.csv                        # 입력 데이터 (헤더 있음, 32,561행)
├── requirements.txt                     # 의존성 목록
├── README.md                            # 이 문서
│
├── main-jhg.py                          # [데이터 준비] 진입점 — 정한결(jhg)
├── src/                                 #   ├ compare.py  Pandas·Polars 로딩 비교
│                                        #   ├ data.py     결측·중복 처리 + EDA
│                                        #   └ config·exceptions·utils
├── HK.py                                # [시각화] Seaborn·Plotly·Cramér's V — 조현강(HK)
├── stats-kjh.py                         # [통계·ML] t검정·카이제곱·ANOVA + 모델 비교 — 김재현(kjh)
├── report-kdi.py                        # [자동화] report.md 자동 생성 — 길다인(kdi)
├── test.py                              # 단일 실행 파이프라인 (report-kdi.py가 함수 재사용)
├── report_template.j2                   # HTML 리포트 템플릿 (test.py용)
│
├── outputs/                             # 실행 산출물
│   ├── report.md                        #   자동 생성 분석 리포트 (제출물)
│   ├── model.joblib / report.html       #   직렬화 모델 · HTML 리포트
│   ├── eda_2x2.png / income_by_education_gender.html
│   ├── adult_clean.csv                  #   정제 데이터
│   └── loader_comparison·eda_summary·benchmark.json
├── figure-kjh/                          # 통계 검정 시각화 (kjh)
├── models-kjh/                          # knn·logistic·xgb 모델 (kjh)
│
└── doc/                                 # 팀원별 정리 문서 (3장 참고)
    ├── HANDOFF-jhg.md
    ├── stats_report-kjh.md
    ├── visual-chart-hk.md
    └── report-kdi.md
```

---

## 2. 데이터셋 설명

- **파일**: `data/adult.csv` (UCI Adult / Census Income)
- **크기**: 32,561행, 헤더 있는 CSV (결측은 `?`로 표기)
- **목적**: 인구통계 정보로 연소득이 **`>50K`인지 `<=50K`인지** 예측하는 이진 분류
- **컬럼 15개**:

| # | 컬럼 | 설명 | # | 컬럼 | 설명 |
|---|------|------|---|------|------|
| 1 | age | 나이 | 9 | race | 인종 |
| 2 | workclass | 고용 형태 | 10 | gender | 성별 |
| 3 | fnlwgt | 인구 가중치 | 11 | capital-gain | 자본 이득 |
| 4 | education | 학력(문자열) | 12 | capital-loss | 자본 손실 |
| 5 | educational-num | 학력 연수(숫자) | 13 | hours-per-week | 주당 근로시간 |
| 6 | marital-status | 결혼 상태 | 14 | native-country | 출신 국가 |
| 7 | occupation | 직업 | 15 | **income** | **타깃** (`<=50K`/`>50K`) |
| 8 | relationship | 가족 내 관계 | | | |

- 결측치는 `workclass`, `occupation`, `native-country` 세 컬럼에 `?`로 존재 (총 4,262건).

---

## 3. 팀원별 담당 · 정리 문서

각 팀원이 담당 파트의 작업 내용·결과·의견을 정리한 문서입니다.

| 담당 | 파트 | 주요 코드 | 정리 문서 |
|---|---|---|---|
| **정한결(jhg)** | 데이터 준비 (로딩 비교·정제·EDA) | `main-jhg.py`, `src/` | [doc/HANDOFF-jhg.md](doc/HANDOFF-jhg.md) |
| **김재현(kjh)** | 통계 검정 · ML 모델 비교 | `stats-kjh.py` | [doc/stats_report-kjh.md](doc/stats_report-kjh.md) |
| **조현강(HK)** | 시각화 (Seaborn·Plotly·Cramér's V) | `HK.py` | [doc/visual-chart-hk.md](doc/visual-chart-hk.md) |
| **길다인(kdi)** | 리포트 자동화 · 데이터 입력 통합 | `report-kdi.py`, `test.py` | [doc/report-kdi.md](doc/report-kdi.md) |

---

## 4. 분석 파이프라인

과제 요구사항별 담당 코드는 다음과 같습니다.

| 단계 | 내용 | 담당 코드 |
|---|---|---|
| **① 데이터 준비** | Pandas·Polars 로딩 비교, 결측·중복 처리, 기본 EDA | `src/compare.py`, `src/data.py` |
| **② 시각화** | Seaborn 정적 차트 · Plotly 인터랙티브 차트 (분포·그룹비교·범주 연관성) | `HK.py`, `test.py` |
| **③ 통계 검정** | 기술통계·상관계수, t검정·카이제곱·ANOVA + p-value 해석 | `stats-kjh.py`, `test.py` |
| **④ ML 파이프라인** | `sklearn.Pipeline`(전처리+모델), 평가 지표, `joblib` 모델 저장 | `test.py`, `stats-kjh.py` |
| **⑤ 자동화** | 분석 결과를 `report.md`로 자동 생성 | `report-kdi.py` |

> `report-kdi.py`는 `adult.csv`를 자체 정제한 뒤, 통계·모델 계산은 `test.py`의 함수를 `import`해 재사용하여 `outputs/report.md`를 생성합니다.

---

## 5. 실행 방법

```bash
# 1) 의존성 설치 (venv 권장)
pip install -r requirements.txt

# 2) 파트별 실행
python main-jhg.py      # 데이터 준비 (Pandas·Polars 비교, 정제, EDA)
python HK.py            # 시각화 (Seaborn·Plotly·Cramér's V)
python stats-kjh.py     # 통계 검정 + 모델 비교
python report-kdi.py    # report.md 자동 생성  → outputs/report.md
python test.py          # 단일 파이프라인 (EDA·검정·모델·리포트 4종 산출)
```

- **환경**: Python 3.9+
- **입력**: `data/adult.csv`
- 모든 경로는 실행 위치가 아니라 각 스크립트 파일(`__file__`) 기준으로 해석되므로 어느 디렉터리에서 실행해도 동작합니다.

---

## 6. 실행 결과 요약

**정제** (Pandas·Polars 결과 일치 확인)
- 원본 32,561행 → 정제 30,139행 (결측 2,399 · 중복 23 제거)
- 고소득(`>50K`) 비율 24.9% — 약 3:1 불균형

**통계 검정 (CDA, 모두 p<0.05 유의)**
- **t검정**: 고소득 평균 44.0세 vs 저소득 36.6세 → 연령 차이 유의
- **카이제곱**: 성별 × 소득 → 연관 있음, chi2 = 1413.80
- **ANOVA**: 직군(workclass)별 근로시간 차이 유의, F = 134.69

**모델 성능** (관심 클래스 = 고소득)
| 모델 | 정확도 | 고소득 F1 | 비고 |
|---|:---:|:---:|---|
| KNN | 0.836 | 0.648 | `test.py` / `report-kdi.py` |
| XGBoost | 0.877 | 0.726 | `stats-kjh.py` (팀 최고 성능) |

> 불균형 데이터라 정확도만으로는 오해가 생길 수 있어, 고소득(소수 클래스)의 재현율·F1을 함께 확인했습니다. 상세 수치는 `outputs/report.md`와 각 팀원 정리 문서를 참고하세요.

---

_광주캠퍼스 4반 — 정한결 · 김재현 · 조현강 · 길다인_
