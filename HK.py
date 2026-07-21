# -*- coding: utf-8 -*-
"""UCI Adult 데이터의 Seaborn 정적 차트와 Plotly 인터랙티브 차트 생성."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 화면 없이 이미지 파일을 만들 수 있도록 설정

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import font_manager


# 이 파일과 같은 위치를 기준으로 데이터 및 결과 경로를 설정합니다.
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "adult.data"
OUTPUT_DIR = BASE_DIR / "outputs"

SEABORN_OUTPUT = OUTPUT_DIR / "seaborn_age_distribution.png"
PLOTLY_OUTPUT = OUTPUT_DIR / "plotly_income_by_education_gender.html"

# UCI Adult 데이터에는 헤더가 없으므로 열 이름을 직접 지정합니다.
COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "educational-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def set_korean_font() -> None:
    """설치된 한글 폰트를 찾아 Matplotlib에 적용합니다."""
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    candidates = [
        "AppleGothic",
        "Apple SD Gothic Neo",
        "Malgun Gothic",
        "NanumGothic",
        "Noto Sans CJK KR",
    ]

    for font_name in candidates:
        if font_name in installed_fonts:
            plt.rcParams["font.family"] = font_name
            break

    plt.rcParams["axes.unicode_minus"] = False


def load_and_clean_data(data_path: Path) -> pd.DataFrame:
    """Adult 데이터를 읽고 시각화에 필요한 형태로 정제합니다."""
    if not data_path.is_file():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")

    df = pd.read_csv(
        data_path,
        header=None,
        names=COLUMNS,
        skipinitialspace=True,
        na_values="?",
    )

    # 문자열 앞뒤의 공백과 adult.test에 붙을 수 있는 마침표를 제거합니다.
    text_columns = df.select_dtypes(include="str").columns
    for column in text_columns:
        df[column] = df[column].str.strip()
    df["income"] = df["income"].str.rstrip(".")

    df = df.dropna().drop_duplicates().copy()
    df["income_label"] = df["income"].where(df["income"] == ">50K", "<=50K")
    df["is_high_income"] = (df["income"] == ">50K").astype(int)

    if df.empty:
        raise ValueError("정제 후 시각화할 데이터가 없습니다.")

    return df


def create_seaborn_chart(df: pd.DataFrame, output_path: Path) -> None:
    """소득 집단별 연령 분포를 Seaborn 정적 차트로 저장합니다."""
    sns.set_theme(style="whitegrid")
    set_korean_font()

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.histplot(
        data=df,
        x="age",
        hue="income_label",
        hue_order=["<=50K", ">50K"],
        bins=30,
        kde=True,
        stat="density",
        common_norm=False,
        element="step",
        ax=ax,
    )

    ax.set_title("소득 집단별 연령 분포")
    ax.set_xlabel("연령")
    ax.set_ylabel("밀도")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_plotly_chart(df: pd.DataFrame, output_path: Path) -> None:
    """학력·성별 고소득률을 Plotly 인터랙티브 차트로 저장합니다."""
    grouped = (
        df.groupby(["education", "gender"], as_index=False)
        .agg(
            income_rate=("is_high_income", "mean"),
            sample_count=("is_high_income", "size"),
        )
    )

    # 전체 고소득률이 높은 학력부터 보이도록 x축 순서를 정합니다.
    education_order = (
        df.groupby("education")["is_high_income"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig = px.bar(
        grouped,
        x="education",
        y="income_rate",
        color="gender",
        barmode="group",
        category_orders={"education": education_order},
        custom_data=["sample_count"],
        title="학력·성별에 따른 고소득(>50K) 비율",
        labels={
            "education": "학력",
            "income_rate": "고소득 비율",
            "gender": "성별",
        },
    )

    fig.update_traces(
        hovertemplate=(
            "학력=%{x}<br>"
            "고소득 비율=%{y:.1%}<br>"
            "표본 수=%{customdata[0]:,}명"
            "<extra></extra>"
        )
    )
    fig.update_layout(
        template="plotly_white",
        hovermode="closest",
        xaxis_tickangle=-45,
        legend_title_text="성별",
    )
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Plotly 자바스크립트를 파일에 포함하여 인터넷 없이도 열 수 있게 합니다.
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)


def main() -> None:
    df = load_and_clean_data(DATA_PATH)
    create_seaborn_chart(df, SEABORN_OUTPUT)
    create_plotly_chart(df, PLOTLY_OUTPUT)

    print(f"Seaborn 차트 저장 완료: {SEABORN_OUTPUT}")
    print(f"Plotly 차트 저장 완료: {PLOTLY_OUTPUT}")


if __name__ == "__main__":
    main()
