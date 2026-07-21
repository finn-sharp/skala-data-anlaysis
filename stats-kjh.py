"""성인 소득 데이터 분석 파이프라인 — 시각화 · 통계검정 · 소득 예측 모델링.

■ 변경 이력
    2026-07-21  길다인  파이프라인 초기 구축
                         - 반복문(for) 기반 다중 컬럼 통계 검정 및 조합별 개별 시각화(PNG) 저장 기능 추가
"""

import os
import pathlib
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "adult.csv"
FIGURE_DIR = pathlib.Path("figure-kjh")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                                precision_recall_fscore_support)
import joblib


def set_korean_font() -> None:
    """실행 환경(OS)에 설치된 한글 폰트를 자동 선택해 차트의 한글 깨짐을 방지한다."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("AppleGothic", "Malgun Gothic", "NanumGothic",
                 "AppleSDGothicNeo", "Noto Sans CJK KR"):
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    else:
        print("[경고] 한글 폰트를 찾지 못함 — 그래프의 한글이 깨질 수 있음")
    plt.rcParams["axes.unicode_minus"] = False


class DataError(Exception):
    """데이터 적재·정제·검증 단계에서 발생하는 '복구 불가' 오류."""
    pass


def temp(path=DATA_PATH):
    df = pd.read_csv(path, header=0)
    return df


def preprocessing(df):
    df = df.copy()                                       
    n_before = len(df)

    missing_by_col = {c: int(v) for c, v in df.isna().sum().items() if v > 0}
    df = df.dropna(how="any")
    n_after_na = len(df)

    df = df.drop_duplicates()
    n_after = len(df)

    if n_after == 0:
        raise DataError("정제 후 학습·분석에 쓸 표본이 없음(0행)")

    df['income'] = (df['income'] == ">50K").astype(int)

    report = {
        "n_before": n_before,
        "n_after": n_after,
        "n_missing_dropped": n_before - n_after_na,
        "n_dup_dropped": n_after_na - n_after,
        "missing_by_col": missing_by_col,
        "pos_rate": float(df['income'].mean()),
    }
    return df, report    


# ============================================================================ #
# 조합별 파일명 인자로 받는 개별 시각화 함수들
# ============================================================================ #
def plot_ttest_result(result: dict, save_name: str, alpha: float = 0.05):
    t_stat = result["t"]
    df_deg = result["n_a"] + result["n_b"] - 2

    x = np.linspace(-4, 4, 1000)
    y = stats.t.pdf(x, df_deg)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, color="black", linewidth=2, label=f"t distribution(df={df_deg})")
    critical = stats.t.ppf(1 - alpha / 2, df_deg)

    ax.fill_between(x[x <= -critical], y[x <= -critical], color="red", alpha=0.3)
    ax.fill_between(x[x >= critical], y[x >= critical], color="red", alpha=0.3, label="Rejection Region")
    ax.axvline(t_stat, color="blue", linestyle="--", linewidth=2, label=f"Observed t={t_stat:.3f}")

    ax.set_title(f"Welch T-Test: {save_name}", fontweight="bold")
    ax.set_xlabel("t statistic")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"ttest_{save_name}.png", dpi=300)
    plt.close()


def plot_chi2_result(result: dict, save_name: str, alpha: float = 0.05):
    chi2_stat = result["chi2"]
    dof = result["dof"]
    max_x = max(chi2_stat * 1.3, dof + 10)
    x = np.linspace(0, max_x, 1000)
    y = stats.chi2.pdf(x, dof)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, color="black", linewidth=2, label=f"Chi-square(df={dof})")
    critical = stats.chi2.ppf(1 - alpha, dof)
    ax.fill_between(x[x >= critical], y[x >= critical], color="red", alpha=0.3, label="Rejection Region")
    ax.axvline(chi2_stat, color="blue", linestyle="--", linewidth=2, label=f"Observed χ²={chi2_stat:.3f}")

    ax.set_title(f"Chi-Square Test: {save_name}", fontweight="bold")
    ax.set_xlabel("Chi-square statistic")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"chi2_{save_name}.png", dpi=300)
    plt.close()


def plot_anova_result(result: dict, save_name: str, alpha: float = 0.05):
    f_stat = result["f"]
    k = result["k_groups"]
    df1 = k - 1
    df2 = 100
    x = np.linspace(0, max(f_stat * 1.5, 10), 1000)
    y = stats.f.pdf(x, df1, df2)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, color="black", linewidth=2, label=f"F distribution(df1={df1}, df2={df2})")
    critical = stats.f.ppf(1 - alpha, df1, df2)
    ax.fill_between(x[x >= critical], y[x >= critical], color="red", alpha=0.3, label="Rejection Region")
    ax.axvline(f_stat, color="blue", linestyle="--", linewidth=2, label=f"Observed F={f_stat:.3f}")

    ax.set_title(f"One-way ANOVA: {save_name}", fontweight="bold")
    ax.set_xlabel("F statistic")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"anova_{save_name}.png", dpi=300)
    plt.close()


def run_ttest(df: pd.DataFrame, group_col: str, value_col: str,
              group_a, group_b, alpha: float = 0.05) -> dict:
    a = df.loc[df[group_col] == group_a, value_col].dropna()
    b = df.loc[df[group_col] == group_b, value_col].dropna()
    if a.empty or b.empty:
        raise DataError(f"t검정 표본 부족: {group_a}={a.size}, {group_b}={b.size}")

    t, p = stats.ttest_ind(a, b, equal_var=False)
    significant = bool(p < alpha)
    msg = (f"p={p:.4g} < {alpha} → [기각] 두 그룹의 평균 {value_col} 차이가 통계적으로 유의미함"
           if significant else
           f"p={p:.4g} ≥ {alpha} → [채택] 평균 {value_col} 차이가 유의미하지 않음(우연 수준)")
    return {"t": float(t), "p": float(p),
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "n_a": int(a.size), "n_b": int(b.size),
            "significant": significant, "msg": msg}


def run_chi2(df: pd.DataFrame, col1: str, col2: str, alpha: float = 0.05) -> dict:
    table = pd.crosstab(df[col1], df[col2])
    if table.size == 0 or int(table.values.sum()) == 0:
        raise DataError(f"카이제곱 분할표가 비어 검정 불가: {col1} × {col2}")

    chi2, p, dof, _ = stats.chi2_contingency(table)
    significant = bool(p < alpha)
    msg = (f"p={p:.4g} < {alpha} → [기각] 두 변수는 독립이 아님 (통계적으로 연관성 있음)"
           if significant else
           f"p={p:.4g} ≥ {alpha} → [채택] 두 변수는 서로 독립임 (연관성 없음)")
    return {"chi2": float(chi2), "p": float(p), "dof": int(dof),
            "significant": significant, "msg": msg}


def run_anova(df: pd.DataFrame, group_col: str, value_col: str,
              alpha: float = 0.05) -> dict:
    groups = [g[value_col].dropna().values
              for _, g in df.groupby(group_col) if g[value_col].dropna().size > 1]
    if len(groups) < 2:
        raise DataError(f"ANOVA 유효 그룹 부족: {group_col} 기준 {len(groups)}개")

    f, p = stats.f_oneway(*groups)
    significant = bool(p < alpha)
    msg = (f"p={p:.4g} < {alpha} → [기각] 그룹 간 평균 {value_col} 차이가 통계적으로 유의미함"
           if significant else
           f"p={p:.4g} ≥ {alpha} → [채택] 그룹 간 평균 {value_col} 차이가 유의미하지 않음")
    return {"f": float(f), "p": float(p), "k_groups": len(groups),
            "significant": significant, "msg": msg}


# ============================================================================ #
# 모델 통합 파이프라인 구성 함수
# ============================================================================ #
def build_pipeline(num_features: list, cat_features: list,
                   model_type: str = "knn", **model_kwargs) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ])
    
    if model_type == "knn":
        n_neighbors = model_kwargs.get("n_neighbors", 5)
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model_type == "logistic":
        max_iter = model_kwargs.get("max_iter", 2000)
        random_state = model_kwargs.get("random_state", 42)
        clf = LogisticRegression(max_iter=max_iter, solver="saga", random_state=random_state)
    elif model_type == "xgb":
        random_state = model_kwargs.get("random_state", 42)
        clf = XGBClassifier(eval_metric="logloss", random_state=random_state, **model_kwargs)
    else:
        raise ValueError(f"지원하지 않는 model_type 입니다: {model_type}")

    return Pipeline([
        ("pre", preprocessor),
        ("clf", clf),
    ])


def train_eval_save_model(df: pd.DataFrame, pipeline: Pipeline,
                          num_features: list, cat_features: list, target: str,
                          model_path: str, model_name: str = "model",
                          test_size: float = 0.2, random_state: int = 42) -> dict:
    features = num_features + cat_features
    data = df.dropna(subset=features + [target])
    if data.empty:
        raise DataError(f"[{model_name}] 학습 표본이 비어 있음(결측 제거 후 0행)")

    X, y = data[features], data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, pred, labels=[1], average=None, zero_division=0)
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_test, pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)

    reloaded = joblib.load(model_path)
    reload_ok = bool(np.array_equal(
        reloaded.predict(X_test.head(100)), pred[:100]))

    return {"model_name": model_name,
            "accuracy": float(accuracy),
            "precision_pos": float(prec[0]), "recall_pos": float(rec[0]),
            "f1_pos": float(f1[0]), "f1_macro": float(f1_macro),
            "confusion": cm.tolist(),
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "reload_ok": reload_ok, "model_path": model_path}


def compare_all_models(df: pd.DataFrame, num_features: list, cat_features: list, 
                       target: str, save_dir: str = "./models") -> pd.DataFrame:
    models_config = [
        {"type": "knn", "name": "KNN", "path": os.path.join(save_dir, "knn_model.pkl")},
        {"type": "logistic", "name": "LogisticRegression", "path": os.path.join(save_dir, "logistic_model.pkl")},
        {"type": "xgb", "name": "XGBoost", "path": os.path.join(save_dir, "xgb_model.pkl")}
    ]
    
    results = []
    for config in models_config:
        pipeline = build_pipeline(num_features, cat_features, model_type=config["type"])
        res = train_eval_save_model(
            df=df, pipeline=pipeline, 
            num_features=num_features, cat_features=cat_features, 
            target=target, model_path=config["path"], model_name=config["name"]
        )
        results.append(res)
    
    summary_df = pd.DataFrame([{
        "Model": r["model_name"],
        "Accuracy": r["accuracy"],
        "Precision(>50K)": r["precision_pos"],
        "Recall(>50K)": r["recall_pos"],
        "F1(>50K)": r["f1_pos"],
        "F1(Macro)": r["f1_macro"],
        "Reload OK": r["reload_ok"]
    } for r in results])
    
    return summary_df.sort_values(by="F1(>50K)", ascending=False).reset_index(drop=True)


def main():
    set_korean_font()
    
    # 1. 데이터 로드 및 전처리
    df = temp()
    df_preprocess, report_preprocess = preprocessing(df)
    print("=== 타겟 변수(income) 분포 ===")
    print(df_preprocess.income.value_counts())
    print("-" * 50)

    alpha = 0.05

    # ========================================================================== #
    # 2. 다중 컬럼 조합에 대한 반복문(for) 기반 통계 검정 및 개별 시각화 저장
    # ========================================================================== #
    print("=== [통계 검정 및 가설 검증 결과 리포트 (다중 조합 반복 실행 및 시각화)] ===")

    # 2-1. Welch's t-test 조합 리스트
    ttest_combinations = [
        {"group_col": "income", "value_col": "age", "g_a": 0, "g_b": 1, "desc": "소득 집단 간 연령(age) 차이"},
        {"group_col": "income", "value_col": "hours-per-week", "g_a": 0, "g_b": 1, "desc": "소득 집단 간 주당 근로시간 차이"},
        {"group_col": "gender", "value_col": "educational-num", "g_a": "Male", "g_b": "Female", "desc": "성별 간 교육 연수 차이"}
    ]

    print("\n[1] Welch's t-test 다중 조합 검정 및 시각화 저장")
    for idx, combo in enumerate(ttest_combinations, 1):
        res = run_ttest(df_preprocess, combo["group_col"], combo["value_col"], combo["g_a"], combo["g_b"], alpha=alpha)
        
        # 조합별 고유 파일명 지정 후 시각화 함수 호출
        save_name = f"{combo['group_col']}_by_{combo['value_col']}"
        plot_ttest_result(res, save_name=save_name, alpha=alpha)

        print(f"  ({idx}) 검정 대상: {combo['desc']} ({combo['group_col']} vs {combo['value_col']})")
        print(f"      · 귀무가설(H₀): 두 집단 간 {combo['value_col']} 평균은 차이가 없다.")
        print(f"      · 대립가설(H₁): 두 집단 간 {combo['value_col']} 평균은 차이가 있다.")
        print(f"      · 통계량(t): {res['t']:.3f} | P-value: {res['p']:.4e}")
        print(f"      · 판정 결과: {res['msg']}")
        print(f"      · [저장완료] figure-kjh/ttest_{save_name}.png\n")

    # 2-2. Chi-Square Test 조합 리스트
    chi2_combinations = [
        {"col1": "gender", "col2": "income", "desc": "성별(gender)과 소득(income) 연관성"},
        {"col1": "race", "col2": "income", "desc": "인종(race)과 소득(income) 연관성"},
        {"col1": "workclass", "col2": "income", "desc": "직군(workclass)과 소득(income) 연관성"}
    ]

    print("[2] Chi-Square Test 다중 조합 검정 및 시각화 저장")
    for idx, combo in enumerate(chi2_combinations, 1):
        res = run_chi2(df_preprocess, combo["col1"], combo["col2"], alpha=alpha)
        
        # 조합별 고유 파일명 지정 후 시각화 함수 호출
        save_name = f"{combo['col1']}_vs_{combo['col2']}"
        plot_chi2_result(res, save_name=save_name, alpha=alpha)

        print(f"  ({idx}) 검정 대상: {combo['desc']} ({combo['col1']} × {combo['col2']})")
        print(f"      · 귀무가설(H₀): {combo['col1']}과(와) {combo['col2']}은(는) 서로 독립적이다 (연관성 없음).")
        print(f"      · 대립가설(H₁): {combo['col1']}과(와) {combo['col2']}은(는) 독립이 아니다 (연관성 있음).")
        print(f"      · 통계량(χ²): {res['chi2']:.3f} (자유도: {res['dof']}) | P-value: {res['p']:.4e}")
        print(f"      · 판정 결과: {res['msg']}")
        print(f"      · [저장완료] figure-kjh/chi2_{save_name}.png\n")

    # 2-3. One-way ANOVA 조합 리스트
    anova_combinations = [
        {"group_col": "workclass", "value_col": "hours-per-week", "desc": "직군(workclass)별 주당 근로시간 차이"},
        {"group_col": "educational-num", "value_col": "age", "desc": "교육 연수별 연령 차이"}
    ]

    print("[3] One-way ANOVA 다중 조합 검정 및 시각화 저장")
    for idx, combo in enumerate(anova_combinations, 1):
        res = run_anova(df_preprocess, combo["group_col"], combo["value_col"], alpha=alpha)
        
        # 조합별 고유 파일명 지정 후 시각화 함수 호출
        save_name = f"{combo['value_col']}_by_{combo['group_col']}"
        plot_anova_result(res, save_name=save_name, alpha=alpha)

        print(f"  ({idx}) 검정 대상: {combo['desc']} ({combo['group_col']} 그룹별 {combo['value_col']})")
        print(f"      · 귀무가설(H₀): 모든 {combo['group_col']} 그룹 간 {combo['value_col']} 평균은 동일하다.")
        print(f"      · 대립가설(H₁): 적어도 하나의 그룹은 {combo['value_col']} 평균이 다르다.")
        print(f"      · 통계량(F): {res['f']:.3f} (그룹 수: {res['k_groups']}개) | P-value: {res['p']:.4e}")
        print(f"      · 판정 결과: {res['msg']}")
        print(f"      · [저장완료] figure-kjh/anova_{save_name}.png\n")

    print("-" * 50)

    # 3. 머신러닝 모델 학습 및 성능 비교 파이프라인 실행
    print("=== 머신러닝 모델 비교 학습 시작 (KNN, LogisticRegression, XGBoost) ===")
    
    num_features = ["age", "educational-num", "capital-gain", "capital-loss", "hours-per-week"]
    cat_features = ["workclass", "marital-status", "occupation", "relationship", "race", "gender", "native-country"]
    target = "income"

    model_summary_df = compare_all_models(
        df=df_preprocess, 
        num_features=num_features, 
        cat_features=cat_features, 
        target=target, 
        save_dir="./models"
    )

    print("\n=== [최종] 모델 성능 비교 결과 요약 ===")
    print(model_summary_df.to_string(index=False))
    print("-" * 50)

    print("모든 분석 및 모델링 파이프라인이 성공적으로 완료되었습니다!")
    return


if __name__ == "__main__":
    main()