from numpy import ndarray
from .base import BiomarkerPoolBase, check_split_data


class EMBP(BiomarkerPoolBase):

    def __init__(self) -> None:
        pass

    def fit(
        X: ndarray,
        S: ndarray,
        W: ndarray,
        Y: ndarray,
        Z: ndarray | None = None,
    ) -> None:
        dats = check_split_data(X, S, W, Y, Z)

        # 1. 使用OLS得到初始值

        # 2. E step, 计算hat x

        # 3. M step，更新参数值

        # 4. 检查是否收敛
        pass
