"""report-kdi.py — adult.csv를 자체 정제하고 분석 결과를 report.md로 자동 생성하는 실행 스크립트.

목적:
    성인 소득(Adult) 데이터를 입력으로 받아 '정제 → 통계검정 → 모델링 → 집계'를 수행한 뒤,
    그 결과를 사람이 읽는 마크다운 리포트(report.md)로 자동 조립·저장한다. 제출물 규격인
    report.md 형식을 담당하며, HTML 리포트(report_template.j2)와 동일한 분석 내용을 다룬다.

설계:
    - 데이터 적재·정제(adult.csv)는 이 스크립트가 자체 수행한다. 팀 표준 입력이 헤더 있는
      adult.csv이므로, 헤더 없는 adult.data용 로더(test.py) 대신 여기서 전용 정제를 둔다.
    - 통계 검정·모델 학습 같은 분석 로직은 다인의 파이프라인(test.py)을 '재사용'한다.
      test.py는 임포트만으로 부수효과가 없도록 설계돼 있어(무거운 I/O는 함수 내부에서만
      발생) 안전하게 함수 단위로 가져다 쓸 수 있다. test.py 자체는 수정하지 않는다.
    - 마크다운 렌더링은 외부 템플릿 없이 순수 파이썬 문자열 빌더로 처리한다(render_report_md).
      숫자 포맷팅만 담당하고 값 계산은 하지 않는다(계산은 재사용하는 각 분석 함수의 책임).

render_report_md가 기대하는 context 키 규칙:
    generated_at : str
    clean        : dict(n_before, n_after, n_missing_dropped, n_dup_dropped, pos_rate)
    ttest        : dict(t, p, mean_a, mean_b, n_a, n_b, significant, msg)
    chi2         : dict(chi2, p, dof, significant, msg)
    anova        : dict(f, p, k_groups, significant, msg)
    metrics      : dict(accuracy, precision_pos, recall_pos, f1_pos, f1_macro,
                        confusion(2x2), n_train, n_test, reload_ok, msg)
    edu_top      : list[dict(education, income_rate)]
    features     : dict(numeric, categorical, target)
    plotly_file  : str

산출물:
    outputs/report.md — 마크다운 분석 리포트(EDA 이미지는 상대경로로 참조)
    참조하는 outputs/eda_2x2.png · income_by_education_gender.html는 없을 때만 생성하고,
    모델 파일 outputs/model.joblib은 매 실행 시 재학습·저장한다.

실행:
    $ python report-kdi.py     # 저장소 루트에서 실행(같은 폴더의 test.py를 재사용)

작성자: 광주캠퍼스 4반 길다인
"""

import os
import sys
from datetime import datetime

# 분석 로직(통계검정·모델링)은 다인의 파이프라인 test.py를 '재사용'한다. test.py는
# __main__ 가드로 보호되고 무거운 I/O가 함수 내부에서만 일어나므로, 임포트만으로는
# 파이프라인이 실행되지 않는다(수정 없이 함수만 가져다 씀).
import test

# adult.csv 자체 정제용. test 임포트 시점에 pandas 등 필수 의존성은 이미 검증된다
# (미설치면 test.py가 설치 안내와 함께 종료).
import pandas as pd


# ---- 경로 설정 --------------------------------------------------------------- #
# 실행 위치와 무관하게 '이 파일'을 기준으로 데이터·산출물 경로를 해석한다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "adult.csv")     # 헤더 있는 팀 표준 입력
OUT_DIR = os.path.join(BASE_DIR, "outputs")
REPORT_MD = os.path.join(OUT_DIR, "report.md")              # 제출 리포트(이 스크립트의 산출물)
EDA_PNG = os.path.join(OUT_DIR, "eda_2x2.png")              # 리포트가 참조하는 EDA 이미지
PLOTLY_HTML = os.path.join(OUT_DIR, "income_by_education_gender.html")
MODEL_PATH = os.path.join(OUT_DIR, "model.joblib")          # joblib 직렬화 모델


# ---- 마크다운 조립 헬퍼 ------------------------------------------------------ #
def _md_table(headers: list, rows: list) -> str:
    """헤더와 행 목록을 받아 마크다운 표 문자열을 만든다.

    Args:
        headers: 열 제목 리스트.
        rows: 각 행을 나타내는 셀 값 리스트의 리스트(셀은 str로 변환).

    Returns:
        헤더·구분선·본문으로 구성된 마크다운 표 문자열.
    """
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(c) for c in row) + " |" for row in rows
    )
    return "\n".join([head, sep, body])


def _sig_label(significant: bool) -> str:
    """유의성 판단(bool)을 리포트에 표기할 한글 라벨로 변환한다."""
    return "**유의미**" if significant else "비유의"


# ---- 데이터 적재·자체 정제 (adult.csv 전용) --------------------------------- #
def load_and_clean_csv(csv_path: str):
    """헤더 있는 adult.csv를 읽어 결측·중복 제거 + income 이진화한 분석용 데이터를 만든다.

    test.py의 clean_data와 동일한 규칙(결측 제거 → 중복 제거 → 관심 클래스 >50K를 1로
    이진화)을 따르되, 입력이 헤더 없는 adult.data가 아니라 헤더 있는 adult.csv라는 점만
    다르다. 이렇게 정제해야 재사용하는 test.py의 검정·모델 함수가 기대하는 형태(income
    0/1, 규격 컬럼명)와 정확히 맞물린다.

    Args:
        csv_path: adult.csv 경로(헤더 있음, 결측은 '?', 콤마 뒤 공백 가능).

    Returns:
        (df_clean, clean_report): 정제 DataFrame과 정제 통계 dict
        (n_before, n_after, n_missing_dropped, n_dup_dropped, missing_by_col, pos_rate).

    Raises:
        test.DataError: 파일이 없거나 정제 후 남은 행이 없을 때.
    """
    if not os.path.isfile(csv_path):
        raise test.DataError(f"데이터 파일을 찾을 수 없음: {csv_path}")

    # 헤더 있는 CSV. 콤마 뒤 공백을 제거(skipinitialspace)하고 '?'를 결측으로 처리한다.
    df = pd.read_csv(csv_path, skipinitialspace=True, na_values="?")
    n_before = len(df)

    missing_by_col = {c: int(v) for c, v in df.isna().sum().items() if v > 0}
    df = df.dropna(how="any")
    n_after_na = len(df)

    df = df.drop_duplicates()
    n_after = len(df)
    if n_after == 0:
        raise test.DataError("정제 후 분석에 쓸 표본이 없음(0행)")

    # 타깃 이진화: 관심 클래스(>50K)를 1로 둔다(test.py 규칙과 동일).
    df[test.TARGET] = (df[test.TARGET] == test.POSITIVE_LABEL).astype(int)

    clean_report = {
        "n_before": n_before,
        "n_after": n_after,
        "n_missing_dropped": n_before - n_after_na,
        "n_dup_dropped": n_after_na - n_after,
        "missing_by_col": missing_by_col,
        "pos_rate": float(df[test.TARGET].mean()),            # 고소득(>50K) 비율
    }
    return df, clean_report


# ---- 분석 실행 → context 조립 (test.py 함수 재사용) ------------------------- #
def build_context(df, clean_report: dict) -> dict:
    """정제 데이터로 검정·모델·집계를 수행해 render_report_md가 기대하는 context를 만든다.

    통계 검정(t·카이제곱·ANOVA)과 모델 학습·평가·직렬화는 test.py의 검증된 함수를 그대로
    재사용하고, 이 함수는 그 결과를 리포트 계약(context 키 규칙)에 맞춰 모으는 역할만 한다.
    리포트가 참조하는 EDA 이미지·Plotly HTML은 없을 때만 생성해 불필요한 재생성을 피한다.

    Args:
        df: load_and_clean_csv가 반환한 정제 DataFrame(income 0/1).
        clean_report: 정제 통계 dict.

    Returns:
        render_report_md에 넘길 context dict.
    """
    # 리포트가 참조하는 시각화 산출물은 이미 있으면 재생성하지 않는다(기존 산출물 보존).
    if not os.path.isfile(EDA_PNG):
        test.eda_plot(df, EDA_PNG)
    if not os.path.isfile(PLOTLY_HTML):
        test.plotly_chart(test.build_agg(df), PLOTLY_HTML)

    # [통계 검정] test.py 함수 재사용 — income은 이미 0/1이라 group_a=1(고소득), group_b=0.
    ttest = test.run_ttest(df, test.TARGET, "age", 1, 0)
    chi2 = test.run_chi2(df, "gender", test.TARGET)
    anova = test.run_anova(df, "workclass", "hours-per-week")

    # [모델] 전처리+KNN 파이프라인 학습·평가·joblib 저장·재적재 검증까지 test.py에 위임.
    pipe = test.build_pipeline(test.NUM_FEATURES, test.CAT_FEATURES)
    metrics = test.train_eval_save(df, pipe, test.NUM_FEATURES, test.CAT_FEATURES,
                                   test.TARGET, MODEL_PATH)

    # [집계] 학력별 고소득률 상위 10.
    edu_top = (df.groupby("education")[test.TARGET].mean()
                 .sort_values(ascending=False).head(10))

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "clean": clean_report,
        "ttest": ttest, "chi2": chi2, "anova": anova, "metrics": metrics,
        "edu_top": [{"education": k, "income_rate": float(v)} for k, v in edu_top.items()],
        "plotly_file": os.path.basename(PLOTLY_HTML),
        "features": {"numeric": test.NUM_FEATURES,
                     "categorical": test.CAT_FEATURES, "target": test.TARGET},
    }


# ---- 리포트 렌더링 ---------------------------------------------------------- #
def render_report_md(context: dict, out_md: str,
                     eda_img: str = "eda_2x2.png") -> str:
    """분석 결과(context)를 마크다운 리포트로 렌더링해 파일로 저장한다.

    HTML 리포트와 동일한 섹션 구성(정제요약 → EDA → 통계검정 → 모델 → 학력별표)을
    마크다운으로 옮긴다. 검정 결과는 통계값과 함께 각 함수가 담아 둔 ``msg``(해석
    문장)를 그대로 실어, 수치와 p-value 해석을 한 번에 제공한다.

    Args:
        context: 파이프라인 결과를 모은 dict(키 규칙은 모듈 docstring 참조).
        out_md: 저장할 마크다운 파일 경로(예: outputs/report.md).
        eda_img: 리포트에서 참조할 EDA 이미지 경로. report.md와 같은 폴더에
            저장되므로 기본값은 파일명(상대경로)이다. base64 data URI 대신
            상대경로를 쓰는 이유는 GitHub·에디터에서 그대로 렌더되기 때문이다.

    Returns:
        저장된 마크다운 파일 경로.
    """
    clean = context["clean"]
    ttest = context["ttest"]
    chi2 = context["chi2"]
    anova = context["anova"]
    metrics = context["metrics"]
    features = context["features"]
    cm = metrics["confusion"]

    lines = []

    # ── 헤더 ──────────────────────────────────────────────────────────────
    lines += [
        "# 성인 소득(Adult Income) 데이터 분석 리포트",
        "",
        f"> 생성 시각: {context['generated_at']} · 작성자: 광주캠퍼스 4반 길다인",
        "",
    ]

    # ── 0. 데이터 정제 요약 ───────────────────────────────────────────────
    lines += [
        "## 0. 데이터 정제 요약",
        "",
        _md_table(
            ["항목", "값"],
            [
                ["원본 행 수", f"{clean['n_before']:,}"],
                ["정제 후 행 수", f"{clean['n_after']:,}"],
                ["결측 제거", f"{clean['n_missing_dropped']:,}"],
                ["중복 제거", f"{clean['n_dup_dropped']:,}"],
                ["고소득(>50K) 비율", f"{clean['pos_rate'] * 100:.1f}%"],
            ],
        ),
        "",
    ]

    # ── 1. 탐색적 분석(EDA) ───────────────────────────────────────────────
    lines += [
        "## 1. 탐색적 분석 (EDA)",
        "",
        f"![EDA 2x2 대시보드]({eda_img})",
        "",
    ]

    # ── 2. 통계 가설 검정(CDA) ────────────────────────────────────────────
    # 검정별로 '통계값 + 유의성 라벨 + 해석(msg)'을 함께 실어 수치와 해석을 붙인다.
    lines += [
        "## 2. 통계 가설 검정 (CDA)",
        "",
        f"### ① t검정 — 소득집단 간 평균 연령 차이 · {_sig_label(ttest['significant'])}",
        "",
        f"- 고소득 평균 {ttest['mean_a']:.1f}세 (n={ttest['n_a']:,}) · "
        f"저소득 평균 {ttest['mean_b']:.1f}세 (n={ttest['n_b']:,})",
        f"- t={ttest['t']:.4f} · p={ttest['p']:.4g}",
        f"- 해석: {ttest['msg']}",
        "",
        f"### ② 카이제곱 — 성별과 소득의 연관성 · {_sig_label(chi2['significant'])}",
        "",
        f"- chi2={chi2['chi2']:.2f} · dof={chi2['dof']} · p={chi2['p']:.4g}",
        f"- 해석: {chi2['msg']}",
        "",
        f"### ③ ANOVA — 직군(workclass)별 근로시간 차이 · {_sig_label(anova['significant'])}",
        "",
        f"- F={anova['f']:.2f} · 그룹수={anova['k_groups']} · p={anova['p']:.4g}",
        f"- 해석: {anova['msg']}",
        "",
    ]

    # ── 3. 소득 예측 모델 ─────────────────────────────────────────────────
    reload_txt = "OK" if metrics["reload_ok"] else "실패"
    lines += [
        "## 3. 소득 예측 모델 (KNN 분류)",
        "",
        _md_table(
            ["지표", "값"],
            [
                ["정확도(accuracy)", f"{metrics['accuracy']:.3f}"],
                ["고소득 정밀도(precision)", f"{metrics['precision_pos']:.3f}"],
                ["고소득 재현율(recall)", f"{metrics['recall_pos']:.3f}"],
                ["고소득 F1", f"{metrics['f1_pos']:.3f}"],
                ["Macro F1", f"{metrics['f1_macro']:.3f}"],
            ],
        ),
        "",
        "**혼동행렬** (행=실제, 열=예측)",
        "",
        _md_table(
            ["실제 \\ 예측", "≤50K (0)", ">50K (1)"],
            [
                ["≤50K (0)", cm[0][0], cm[0][1]],
                [">50K (1)", cm[1][0], cm[1][1]],
            ],
        ),
        "",
        f"- 학습 {metrics['n_train']:,} / 평가 {metrics['n_test']:,} · "
        f"재로딩 예측 일치 검증: {reload_txt}",
        f"- {metrics['msg']}",
        f"- 피처 — 수치형: `{', '.join(features['numeric'])}` · "
        f"범주형: `{', '.join(features['categorical'])}`",
        "",
    ]

    # ── 4. 학력별 고소득률 상위 10 ────────────────────────────────────────
    lines += [
        "## 4. 학력별 고소득률 상위 10",
        "",
        _md_table(
            ["학력(education)", "고소득(>50K) 비율"],
            [[row["education"], f"{row['income_rate'] * 100:.1f}%"]
             for row in context["edu_top"]],
        ),
        "",
        f"인터랙티브 대시보드: `{context['plotly_file']}` (outputs/ 폴더에서 열람)",
        "",
    ]

    # 파일 저장(상위 폴더 없으면 생성). 끝에 개행 하나를 둬 POSIX 텍스트 규격을 지킨다.
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    return out_md


# ---- 실행 진입점 ------------------------------------------------------------- #
def main() -> None:
    """adult.csv 자체 정제 → 분석(test.py 재사용) → report.md 생성까지 실행한다."""
    test.section("[0] adult.csv 적재 및 자체 정제")
    df, clean_report = load_and_clean_csv(DATA_PATH)
    print(f"* 원본 {clean_report['n_before']:,}행 → 정제 {clean_report['n_after']:,}행 "
          f"(결측 {clean_report['n_missing_dropped']:,} · 중복 {clean_report['n_dup_dropped']:,} 제거)")
    print(f"* 고소득(>50K) 비율: {clean_report['pos_rate']:.3f}")

    test.section("[1] 분석 실행 — 통계 검정 · 모델 학습 (test.py 재사용)")
    context = build_context(df, clean_report)
    print(f"* t검정 p={context['ttest']['p']:.4g} · 카이제곱 p={context['chi2']['p']:.4g} "
          f"· ANOVA p={context['anova']['p']:.4g}")
    print(f"* 모델 정확도={context['metrics']['accuracy']:.3f} "
          f"· 고소득 F1={context['metrics']['f1_pos']:.3f}")

    test.section("[2] 마크다운 리포트 자동 생성")
    render_report_md(context, REPORT_MD)
    print(f"* 저장: {REPORT_MD}")


if __name__ == "__main__":
    # 최상위 방어선: 프로세스 종료 여부는 이 진입점에서만 판단한다(함수는 예외를 raise할 뿐).
    #  - DataError        : 예상된 입력/정제 실패 → 원인 메시지만 간결히 출력
    #  - KeyboardInterrupt: 사용자 중단
    #  - 그 외 Exception  : 예상 밖 오류를 위한 마지막 안전망
    try:
        main()
    except test.DataError as e:
        sys.exit(f"[치명] {e}")
    except KeyboardInterrupt:
        sys.exit("\n[중단] 사용자에 의해 중단됨")
    except Exception as e:  # noqa: BLE001 - 최상위 포괄 예외 처리
        sys.exit(f"[치명] 예기치 못한 오류: {e}")
