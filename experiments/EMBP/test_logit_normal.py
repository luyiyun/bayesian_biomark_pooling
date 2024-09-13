from time import perf_counter

import numpy as np
from scipy.special import ndtri, expit
from scipy.interpolate import RectBivariateSpline, dfitpack
from sklearn.svm import SVR


class TimeCounter:

    def __enter__(self):
        self._t1 = perf_counter()
        return self

    def __exit__(self, a, b, c):
        self._t2 = perf_counter()
        self._time = self._t2 - self._t1

    @property
    def time(self):
        return self._time


def logit_normal_statistics(
    mu: np.ndarray,
    sigma: np.ndarray,
    K: int = 1000,
):
    if mu.ndim == 2:
        base = ndtri(np.arange(1, K) / K)[:, None, None]
    elif mu.ndim == 1:
        base = ndtri(np.arange(1, K) / K)[:, None]
    else:
        raise ValueError

    h = base * sigma + mu
    p = expit(h)
    y1 = p.mean(axis=0)
    y2 = expit(h * base).mean(axis=0)
    p2 = p * (1 - p)
    y3 = p2.mean(axis=0)
    y4 = (p2 * h).mean(axis=0)
    y5 = (p2 * h**2).mean(axis=0)
    return np.stack([y1, y2, y3, y4, y5], axis=0)


# 得到训练数据集
nx, ny = 50, 50
# tx = ndtri(np.arange(1, nx) / nx)
tx = np.linspace(-10, 10, nx)
ty = np.exp(np.linspace(np.log(0.01), np.log(100), num=ny))
txx, tyy = np.meshgrid(tx, ty, indexing="ij")
res = logit_normal_statistics(txx, tyy, K=10000)

# 对每一个值，建立其样条函数
splines = []
for resi in res:
    spli_func = RectBivariateSpline(tx, ty, resi, s=0.001)
    splines.append(spli_func.tck)

# 训练一个svm来预测
classifier = SVR()
classifier.fit(
    np.stack([txx.flatten(), tyy.flatten()], axis=1), res[0].flatten()
)

# 测试每个值的拟合精度
test_x = np.random.uniform(-100, 100, 200)
test_y = np.exp(np.random.uniform(np.log(0.01), np.log(100), size=200))

with TimeCounter() as tc:
    pred = []
    for tcki in splines:
        txi, tyi, ci = tcki
        predi, _ = dfitpack.bispeu(txi, tyi, ci, 3, 3, test_x, test_y)
        pred.append(predi)
    pred = np.stack(pred, axis=0)
print(f"The time of spline method: {tc.time}")

with TimeCounter() as tc:
    label = logit_normal_statistics(test_x, test_y, K=10000)
mse_spl = (pred - label).mean(axis=1)
print(
    f"The time of quasi-MC(K=10000): {tc.time}, "
    f"the mse of spline method is {mse_spl}"
)

with TimeCounter() as tc:
    pred_1000 = logit_normal_statistics(test_x, test_y, K=1000)
mse_1000 = (pred_1000 - label).mean(axis=1)
print(f"The time of quasi-MC(K=1000): {tc.time}, the mse of it is {mse_1000}")

with TimeCounter() as tc:
    pred_svm = classifier.predict(np.stack([test_x, test_y], axis=1))
mse_svm = (pred_svm - label[0]).mean()
print(f"The time of svm: {tc.time}, the mse of it is {mse_svm}")
