from src.utils import set_korean_font
from src.stats import temp
from src.stats import preprocessing
from src.stats import run_ttest, run_chi2, run_anova
from src.stats import plot_ttest_result, plot_chi2_result, plot_anova_result
from src.stats import compare_all_models

from dotenv import load_dotenv
import os
from pathlib import Path
import pathlib

load_dotenv(".env.stats")
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
DATA_PATH = DATA_DIR / os.getenv("DATA_FILE", "adult.csv")
CDA_FIGURE_DIR = BASE_DIR / os.getenv("CDA_FIGURE_DIR", "figure")
CDA_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = BASE_DIR / os.getenv("MODEL_DIR", "model")


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
        print(f"      · [저장완료] output/figure_cda/ttest_{save_name}.png\n")

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
        print(f"      · [저장완료] output/figure_cda//chi2_{save_name}.png\n")

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
        print(f"      · [저장완료] output/figure_cda/anova_{save_name}.png\n")

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
        save_dir=MODEL_DIR
    )

    print("\n=== [최종] 모델 성능 비교 결과 요약 ===")
    print(model_summary_df.to_string(index=False))
    print("-" * 50)

    print("모든 분석 및 모델링 파이프라인이 성공적으로 완료되었습니다!")
    return


if __name__ == "__main__":
    main()