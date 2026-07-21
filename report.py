"""report.py — 분석 결과(context)를 마크다운 리포트(report.md)로 자동 생성하는 모듈.

목적:
    파이프라인 각 단계(정제·EDA·검정·모델링·집계)가 만든 결과 dict를 하나로 모은
    ``context``를 입력받아, 사람이 읽는 마크다운 리포트를 자동으로 조립·저장한다.
    HTML 리포트(report_template.j2)와 동일한 내용을 다루되, 제출물 규격인
    ``report.md`` 형식으로 출력한다.

설계:
    - 이 모듈은 '파이프라인 맨 끝의 소비자'로, 앞 단계가 어떻게 파일로 쪼개지든
      ``context`` 계약(아래 키 규칙)만 지키면 다른 모듈에 의존하지 않는다.
    - 렌더링은 외부 템플릿 없이 순수 파이썬 문자열 빌더로 처리한다. 마크다운은
      HTML과 달리 이스케이프 이슈가 적어 템플릿 의존성을 두지 않는 편이 단순하다.
    - 숫자 포맷팅만 담당하고 값 계산은 하지 않는다(계산은 각 분석 함수의 책임).

기대하는 context 키 규칙:
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

작성자: 광주캠퍼스 4반 길다인
"""

import os


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
