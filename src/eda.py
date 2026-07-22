# -*- coding: utf-8 -*-
"""UCI Adult 데이터의 Seaborn 정적 차트와 Plotly 인터랙티브 차트 생성."""

import os
from pathlib import Path
from dotenv import load_dotenv

import plotly.express as px
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
matplotlib.use("Agg")  # 화면 없이 이미지 파일을 만들 수 있도록 설정

from src.utils import set_korean_font

load_dotenv(".env.stats")  
# 이 파일과 같은 위치를 기준으로 데이터 및 결과 경로를 설정합니다.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
DATA_PATH = BASE_DIR / os.getenv("DATA_DIR") / os.getenv("DATA_FILE")
OUTPUT_DIR = BASE_DIR / os.getenv("OUTPUT_DIR")
EDA_FIGURE_DIR = BASE_DIR / os.getenv("EDA_FIGURE_DIR")

SEABORN_OUTPUT = BASE_DIR / os.getenv("SEABORN_OUTPUT")
PLOTLY_OUTPUT = BASE_DIR / os.getenv("PLOTLY_OUTPUT")
CRAMERS_V_PNG = BASE_DIR / os.getenv("CRAMERS_V_PNG")
CRAMERS_V_HTML = BASE_DIR / os.getenv("CRAMERS_V_HTML")

COLUMNS = os.getenv("COLUMNS", "").split(",")
CATEGORICAL_FEATURES = os.getenv("CATEGORICAL_FEATURES", "").split(",")

def load_and_clean_data(data_path: Path) -> pd.DataFrame:
    """Adult 데이터를 읽고 시각화에 필요한 형태로 정제합니다."""
    if not data_path.is_file():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")

    # adult.csv의 첫 번째 행에 있는 컬럼명을 헤더로 읽습니다.
    df = pd.read_csv(data_path, skipinitialspace=True, na_values="?")

    missing_columns = [column for column in COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")

    # 시각화에 필요한 컬럼만 정해진 순서로 사용합니다.
    df = df[COLUMNS].copy()

    # 문자열 앞뒤의 공백과 adult.test에 붙을 수 있는 마침표를 제거합니다.
    text_columns = df.select_dtypes(include="object").columns
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


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """두 범주형 변수의 연관성을 편향 보정 Cramér's V로 계산합니다."""
    table = pd.crosstab(x, y)
    if table.empty or min(table.shape) < 2:
        return 0.0

    n = int(table.to_numpy().sum())
    if n <= 1:
        return 0.0

    chi2 = chi2_contingency(table, correction=False)[0]
    phi2 = chi2 / n
    rows, columns = table.shape

    # 표본 크기와 범주 수에 따라 Cramér's V가 커지는 편향을 보정합니다.
    phi2_corrected = max(
        0.0,
        phi2 - ((columns - 1) * (rows - 1)) / (n - 1),
    )
    rows_corrected = rows - ((rows - 1) ** 2) / (n - 1)
    columns_corrected = columns - ((columns - 1) ** 2) / (n - 1)
    denominator = min(rows_corrected - 1, columns_corrected - 1)

    if denominator <= 0:
        return 0.0
    return float(np.sqrt(phi2_corrected / denominator))


def build_cramers_v_matrix(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """여러 범주형 변수 사이의 Cramér's V 연관성 행렬을 생성합니다."""
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"범주형 분석에 필요한 컬럼이 없습니다: {missing_columns}")

    matrix = pd.DataFrame(
        np.eye(len(columns)),
        index=columns,
        columns=columns,
        dtype=float,
    )

    for i, column_a in enumerate(columns):
        for j in range(i + 1, len(columns)):
            column_b = columns[j]
            value = cramers_v(df[column_a], df[column_b])
            matrix.loc[column_a, column_b] = value
            matrix.loc[column_b, column_a] = value

    return matrix


def create_cramers_v_seaborn_chart(
    matrix: pd.DataFrame,
    output_path: Path,
) -> None:
    """범주형 변수 연관성 행렬을 Seaborn 정적 히트맵으로 저장합니다."""
    sns.set_theme(style="white")
    set_korean_font()

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Cramer's V"},
        ax=ax,
    )

    # AppleGothic에 없는 악센트 문자 경고를 피하기 위해 차트에는 ASCII 표기를 사용합니다.
    ax.set_title("범주형 변수 간 연관성 — Cramer's V")
    ax.set_xlabel("범주형 변수")
    ax.set_ylabel("범주형 변수")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_cramers_v_plotly_chart(
    matrix: pd.DataFrame,
    output_path: Path,
) -> None:
    """범주형 변수 연관성 행렬을 Plotly 인터랙티브 히트맵으로 저장합니다."""
    fig = px.imshow(
        matrix,
        text_auto=".2f",
        zmin=0,
        zmax=1,
        aspect="auto",
        color_continuous_scale="Blues",
        title="범주형 변수 간 연관성 — Cramér's V",
        labels={
            "x": "범주형 변수",
            "y": "범주형 변수",
            "color": "Cramér's V",
        },
    )
    fig.update_traces(
        hovertemplate=(
            "변수 1=%{x}<br>"
            "변수 2=%{y}<br>"
            "Cramér's V=%{z:.3f}"
            "<extra></extra>"
        )
    )
    fig.update_layout(template="plotly_white", xaxis_tickangle=-45)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)


def main() -> None:
    df = load_and_clean_data(DATA_PATH)
    create_seaborn_chart(df, SEABORN_OUTPUT)
    create_plotly_chart(df, PLOTLY_OUTPUT)

    cramers_matrix = build_cramers_v_matrix(df, CATEGORICAL_FEATURES)
    create_cramers_v_seaborn_chart(cramers_matrix, CRAMERS_V_PNG)
    create_cramers_v_plotly_chart(cramers_matrix, CRAMERS_V_HTML)

    print(f"Seaborn 차트 저장 완료: {SEABORN_OUTPUT}")
    print(f"Plotly 차트 저장 완료: {PLOTLY_OUTPUT}")
    print(f"범주형 연관성 이미지 저장 완료: {CRAMERS_V_PNG}")
    print(f"범주형 연관성 HTML 저장 완료: {CRAMERS_V_HTML}")


if __name__ == "__main__":
    main()
