# -*- coding: utf-8 -*-
"""공통 헬퍼 — 로그 배너, 한글 폰트 설정.

시각화(eda.py 등)에서만 쓰는 matplotlib 의존은 함수 내부에서 지연 임포트해,
data.py 같은 비-시각화 모듈을 쓸 때는 matplotlib 없이도 동작하게 한다.
"""


def section(title: str) -> None:
    """구분선과 함께 섹션 제목을 표준출력에 남긴다(진행 로그용, 반환값 없음)."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def set_korean_font() -> None:
    """실행 환경(OS)에 설치된 한글 폰트를 자동 선택해 차트의 한글 깨짐을 방지한다.

    후보 폰트를 우선순위대로 확인해 처음 발견한 것을 적용한다. 사용 가능한 폰트가
    하나도 없으면 경고만 남기고 진행한다(차트는 그려지되 한글이 깨질 수 있음).
    음수 축 라벨(−)이 네모로 깨지는 문제도 함께 비활성화한다.
    """
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("AppleGothic", "Malgun Gothic", "NanumGothic",
                 "AppleSDGothicNeo", "Noto Sans CJK KR"):
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    else:
        print("[경고] 한글 폰트를 찾지 못함 — 그래프의 한글이 깨질 수 있음")
    plt.rcParams["axes.unicode_minus"] = False
