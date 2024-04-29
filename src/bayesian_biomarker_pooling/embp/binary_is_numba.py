import numpy as np

# from scipy.special import expit, log_expit, softmax
# from scipy.stats import norm as Normal
from numpy import ndarray

# from tqdm import tqdm
from numba import jit

# from .utils import logistic


EPS = 1e-5
LOGIT_3 = 6.9067548


@jit(nopython=True)
def expit(arr: np.ndarray):
    return 1 / (1 + np.exp(-arr))


@jit(nopython=True)
def log_expit(arr: np.ndarray):
    return -np.log(1 + np.exp(arr))


@jit(nopython=True)
def softmax(arr: np.ndarray):
    arr_exp = np.exp(arr)
    return arr_exp / arr_exp.sum(axis=0)


@jit(nopython=True)
def IS_binary_EM(
    X: ndarray,
    S: ndarray,
    W: ndarray,
    Y: ndarray,
    Z: ndarray | None = None,
    init_params: ndarray | None = None,
    max_iter: int = 300,
    max_iter_inner: int = 100,
    delta1: float = 1e-3,
    delta1_inner: float = 1e-4,
    delta1_var: float = 1e-2,
    delta2: float = 1e-2,
    delta2_inner: float = 1e-6,
    delta2_var: float = 1e-2,
    pbar: bool = True,
    random_seed: int | None = None,
    lr: float = 1.0,
    min_nIS: int = 100,
    max_nIS: int = 5000,
    gem: bool = True,
) -> tuple[ndarray, ndarray]:
    # initialization
    # seed = np.random.default_rng(random_seed)
    nIS = min_nIS

    # prepare
    is_m = np.isnan(X)
    is_o = ~is_m
    ind_m = np.nonzero(is_m)[0]
    ind_o = np.nonzero(is_o)[0]

    Xo = X[ind_o]
    Yo = Y[ind_o]
    # Wo = W[ind_o]
    Ym = Y[ind_m]
    Wm = W[ind_m]
    if Z is not None:
        Zo = Z[ind_o]
        Zm = Z[ind_m]
    else:
        Zo = Zm = None
    studies = np.unique(S)
    ind_inv = np.zeros_like(S)
    for i in range(len(studies)):
        ind_inv[S == studies[i]] = i
    ind_m_inv = ind_inv[is_m]
    # ind_o_inv = ind_inv[is_o]

    # the transpose of 1-d array is still 1-d array
    ind_S = [np.nonzero(S == s)[0] for s in studies]
    # ind_Sm = [np.nonzero((S.T == s) & is_m)[0] for s in studies]
    ind_So = [np.nonzero((S.T == s) & is_o)[0] for s in studies]

    n = Y.shape[-1]
    ns = studies.shape[-1]
    nz = 0 if Z is None else Z.shape[-1]
    n_o = is_o.sum()
    n_m = is_m.sum()
    # n_s = np.array([indi.shape[-1] for indi in ind_S])

    wbar_s = np.zeros(ns, dtype=np.double)
    wwbar_s = np.zeros(ns, dtype=np.double)
    for i in range(len(ind_S)):
        wbar_s[i] = W[ind_S[i]].mean()
        wwbar_s[i] = (W[ind_S[i]] ** 2).mean()

    # sigma_ind = np.array(
    #     [1]
    #     + list(range(2 + 2 * ns, 2 + 2 * (ns + 1)))
    #     + list(range(3 + 4 * ns + nz, 3 + 5 * ns + nz))
    # )

    C = np.zeros((n, ns))
    for i in range(ns):
        C[ind_S[i], i] = 1
    Co = C[is_o, :]
    Cm_des = np.zeros((n_m, ns + nz), dtype=np.double)
    Cm_des[:, :ns] = C[is_m, :]
    if Z is not None:
        Cm_des[:, ns:] = Zm

    Xo_des = np.zeros((n_o, ns + 1 + nz), dtype=np.double)
    Xo_des[:, 0] = Xo
    Xo_des[:, 1 : (ns + 1)] = Co
    if Z is not None:
        Xo_des[:, (ns + 1) :] = Zo
    Xm = np.zeros(n_m, dtype=np.double)  # 用于e-step中的newton algorithm

    # params_ind = {
    #     "mu_x": slice(0, 1),
    #     "sigma2_x": slice(1, 2),
    #     "a": slice(2, 2 + ns),
    #     "b": slice(2 + ns, 2 + 2 * ns),
    #     "sigma2_w": slice(2 + 2 * ns, 2 + 3 * ns),
    #     "beta_x": slice(2 + 3 * ns, 3 + 3 * ns),
    #     "beta_0": slice(3 + 3 * ns, 3 + 4 * ns),
    #     "beta_z": slice(3 + 4 * ns, 3 + 4 * ns + nz),
    # }
    n_p = 3 + 4 * ns + nz

    Xhat = np.copy(X)
    Xhat2 = Xhat**2

    # init
    if init_params is not None:
        params = init_params
    else:
        mu_x = Xo.mean()
        sigma2_x = np.var(Xo)
        a = np.zeros(ns, dtype=np.double)
        b = np.zeros(ns, dtype=np.double)
        sigma2_w = np.ones(ns, dtype=np.double)
        for i in range(ns):
            ind_so_i = ind_So[i]
            if len(ind_so_i) == 0:
                continue
            # abi, sigma2_ws_i = ols(X[ind_so_i], W[ind_so_i])
            X_o_i = X[ind_so_i]
            W_o_i = W[ind_so_i]
            X_des_o_i = np.ones((ind_so_i.shape[0], 2 + nz), dtype=np.double)
            X_des_o_i[:, 0] = X_o_i
            if Z is not None:
                X_des_o_i[:, 2:] = Z[ind_so_i]
            ls_res = np.linalg.lstsq(X_des_o_i, W_o_i)
            abi = ls_res[0]
            resid = ls_res[1]
            a[i] = abi[1]
            b[i] = abi[0]
            sigma2_w[i] = resid.item()

        # beta = logistic(Xo, Yo, Zo)
        X_des_o = np.ones((Xo.shape[0], 2 + nz), dtype=np.double)
        X_des_o[:, 0] = Xo
        if Z is not None:
            X_des_o[:, 2:] = Zo
        beta_logistic = np.zeros(X_des_o.shape[1])
        for i in range(max_iter_inner):
            p = expit(X_des_o @ beta_logistic)
            grad = X_des_o.T @ (p - Yo)
            hess = X_des_o.T @ np.diag(p * (1 - p)) @ X_des_o

            beta_logistic_delta = lr * np.linalg.solve(hess, grad)
            beta_logistic -= beta_logistic_delta

            rdiff = np.max(
                np.abs(beta_logistic_delta)
                / (np.abs(beta_logistic) + delta1_inner)
            )
            # logger_embp.info(
            #     f"Init step Newton-Raphson: iter={i+1} diff={rdiff:.4f}"
            # )
            if rdiff < delta2_inner:
                break
        # else:
        #     logger_embp.warning(
        #         f"Init Newton-Raphson (max_iter={max_iter}) doesn't converge"
        #     )
        params = np.zeros(n_p, dtype=np.double)
        params[0] = mu_x
        params[1] = sigma2_x
        params[2 : (2 + ns)] = a
        params[(2 + ns) : (2 + 2 * ns)] = b
        params[(2 + 2 * ns) : (2 + 3 * ns)] = sigma2_w
        params[(2 + 3 * ns)] = beta_logistic[0]
        params[(3 + 3 * ns) : (3 + 4 * ns)] = beta_logistic[1]
        if Z is not None:
            params[(3 + 4 * ns) :] = beta_logistic[2:]

    # main loop
    params_hist_ = np.zeros((max_iter, n_p), dtype=np.double)
    params_hist_[0, :] = params
    # for iter_i in tqdm(
    #     range(1, max_iter + 1),
    #     desc="EM: ",
    #     disable=not pbar,
    # ):
    for iter_i in range(1, max_iter + 1):
        # --------------- e step start ---------------
        mu_x = params[0]
        sigma2_x = params[1]
        a = params[2 : (2 + ns)]
        b = params[(2 + ns) : (2 + 2 * ns)]
        sigma2_w = params[(2 + 2 * ns) : (2 + 3 * ns)]
        beta_x = params[(2 + 3 * ns)]
        beta_0 = params[(3 + 3 * ns) : (3 + 4 * ns)]
        beta_z = params[(3 + 4 * ns) :] if Z is not None else 0.0
        beta_0_m_long = beta_0[ind_m_inv]
        a_m_long = a[ind_m_inv]
        b_m_long = b[ind_m_inv]
        sigma2_w_m_long = sigma2_w[ind_m_inv]

        # 使用newton-raphson方法得到Laplacian approximation
        # NOTE: 使用Scipy.Newton-CG无法收敛，反而自己的这个每次在3个步骤之内就收敛了
        p_mult = sigma2_w * sigma2_x * beta_x
        p2_mult = p_mult * beta_x
        x_mult = sigma2_x * b**2 + sigma2_w
        const_part = sigma2_w * mu_x - sigma2_x * b * a

        const_m_long = (
            p_mult[ind_m_inv] * Ym
            + (sigma2_x * b)[ind_m_inv] * Wm
            + const_part[ind_m_inv]
        )
        p_mult_m_long = p_mult[ind_m_inv]
        p2_mult_m_long = p2_mult[ind_m_inv]
        x_mult_m_long = x_mult[ind_m_inv]

        Z_part_m = 0.0 if Z is None else Zm @ beta_z
        delta_part = beta_0_m_long + Z_part_m

        for i in range(1, max_iter_inner + 1):
            p = expit(Xm * beta_x + delta_part)

            xdelta = (
                lr
                * (p_mult_m_long * p + x_mult_m_long * Xm - const_m_long)
                / (p2_mult_m_long * p * (1 - p) + x_mult_m_long)
            )
            Xm -= xdelta

            rdiff = np.max(np.abs(xdelta) / (np.abs(Xm) + delta1_inner))
            # logger_embp.info(
            #     f"E step Newton-Raphson: iter={i} diff={rdiff:.4f}"
            # )
            if rdiff < delta2_inner:
                break
        # else:
        #     logger_embp.warning(
        #         f"E step Newton-Raphson (max_iter={max_iter_inner})"
        #         " doesn't converge"
        #     )

        # 重新计算一次hessian
        # 不要使用multivariate_norm，会导致维数灾难，
        # 因为Xm的每个分量都是独立的，使用单变量的norm会好一些
        p = expit(Xm * beta_x + delta_part)
        hess_inv = (sigma2_w * sigma2_x)[ind_m_inv] / (
            p2_mult_m_long * p * (1 - p) + x_mult_m_long
        )
        # norm_lap = Normal(
        #     loc=Xm, scale=np.sqrt(hess_inv) + EPS
        # )  # TODO: 这个EPS可能不是必须的

        # 进行IS采样
        eps = np.random.rand(nIS, n_m)
        XIS = eps * np.sqrt(hess_inv) + Xm
        # XIS = norm_lap.rvs(size=(nIS, n_m), random_state=seed)  # N x n_m

        # 计算对应的(normalized) importance weights
        pIS = log_expit((2 * Ym - 1) * (beta_x * XIS + delta_part))
        pIS -= 0.5 * (
            (Wm - a_m_long - b_m_long * XIS) ** 2 / sigma2_w_m_long
            + (XIS - mu_x) ** 2 / sigma2_x
        )
        # pIS = pIS - norm_lap.logpdf(XIS)
        pIS = pIS + eps / 2
        # NOTE: 尽管归一化因子对于求极值没有贡献，但有助于稳定训练
        WIS = softmax(pIS)
        # if logger_embp.level <= logging.INFO:
        #     Seff = 1 / np.sum(WIS**2, axis=0)
        #     logger_embp.info(
        #         "Importance effective size "
        #         + f"is {Seff.mean():.2f}±{Seff.std():.2f}"
        #     )

        # 计算Xhat和Xhat2, 并讲Xm更新为IS计算的后验均值
        Xhat[is_m] = Xm = np.sum(XIS * WIS, axis=0)
        Xhat2[is_m] = np.sum(XIS**2 * WIS, axis=0)
        # --------------- e step end ---------------

        # --------------- m step start ---------------
        vbar = Xhat2.mean()
        xbars = np.array([Xhat[sind].mean() for sind in ind_S])
        vbars = np.array([Xhat2[sind].mean() for sind in ind_S])
        wxbars = np.array([np.mean(W[sind] * Xhat[sind]) for sind in ind_S])

        # 更新参数：mu_x,sigma2_x,a,b,sigma2_w
        mu_x = Xhat.mean()
        sigma2_x = vbar - mu_x**2
        b = (wxbars - wbar_s * xbars) / (vbars - xbars**2)
        a = wbar_s - b * xbars
        sigma2_w = (
            wwbar_s
            + vbars * b**2
            + a**2
            - 2 * (a * wbar_s + b * wxbars - a * b * xbars)
        )

        # 使用newton-raphson算法更新beta_x,beta_0,beta_z
        beta_all = params[(2 + 3 * ns) :]
        # NOTE: 我自己的实现更快
        WXIS = XIS * WIS
        for i in range(max_iter_inner):
            # grad_o
            p_o = expit(Xo_des @ beta_all)  # ns
            grad = Xo_des.T @ (p_o - Yo)
            # grad_m
            p_m = expit(XIS * beta_all[0] + Cm_des @ beta_all[1:])  # N x nm
            Esig = (p_m * WIS).sum(axis=0)
            Esigx = (p_m * WXIS).sum(axis=0)
            grad[0] += (Esigx - Ym * Xm).sum()
            grad[1:] += Cm_des.T @ (Esig - Ym)

            # hess_o
            hess = Xo_des.T @ np.diag(p_o * (1 - p_o)) @ Xo_des
            # hess = np.einsum("ij,i,ik->jk", Xo_des, p_o * (1 - p_o), Xo_des)
            p_m2 = p_m * (1 - p_m)
            hess_m_00 = (p_m2 * XIS**2 * WIS).sum(axis=0).sum()
            hess_m_01 = Cm_des.T @ ((p_m2 * WXIS).sum(axis=0))
            hess_m_11 = Cm_des.T @ np.diag((p_m2 * WIS).sum(axis=0)) @ Cm_des
            hess[0, 0] += hess_m_00
            hess[0, 1:] += hess_m_01
            hess[1:, 0] += hess_m_01
            hess[1:, 1:] += hess_m_11
            beta_delta = np.linalg.solve(hess, grad)
            if gem:
                beta_all -= beta_delta
                break

            rdiff = np.max(
                np.abs(beta_delta) / (np.abs(beta_all) + delta1_inner)
            )
            beta_all = beta_all - beta_delta
            # logger_embp.info(
            #     f"M step Newton-Raphson: iter={i+1} diff={rdiff:.4f}"
            # )
            if rdiff < delta2_inner:
                break
        # else:
        #     logger_embp.warning(
        #         f"M step Newton-Raphson (max_iter={max_iter_inner})"
        #         " doesn't converge"
        #     )

        params_new = np.zeros(n_p, dtype=np.double)
        params_new[0] = mu_x
        params_new[1] = sigma2_x
        params_new[2 : (2 + ns)] = a
        params_new[(2 + ns) : (2 + 2 * ns)] = b
        params_new[(2 + 2 * ns) : (2 + 3 * ns)] = sigma2_w
        params_new[(2 + 3 * ns) :] = beta_all
        # --------------- m step end ---------------

        rdiff = np.max(
            np.abs(params - params_new) / (np.abs(params) + delta1),
        )
        # logger_embp.info(
        #     f"EM iteration {iter_i}: "
        #     f"relative difference is {rdiff: .4f}"
        # )
        if pbar:
            print("EM iteration ", iter_i, "/", max_iter)
            # print(
            #     "EM iteration %d/%d: " % (iter_i, max_iter),
            #     "relative difference is %d.4f" % rdiff
            # )
        params = params_new  # 更新
        params_hist_[iter_i, :] = params
        if rdiff < delta2:  # TODO: 这里有优化的空间
            iter_convergence_ = iter_i
            break

        if max_nIS > min_nIS:
            nIS = min_nIS + int(
                (max_nIS - min_nIS)
                * expit(2 * LOGIT_3 * iter_i / max_iter - LOGIT_3)
            )
            # logger_embp.info(f"Update Monte Carlo Sampling size to {nIS}.")
    # else:
    #     logger_embp.warning(
    #         f"EM iteration (max_iter={max_iter}) " "doesn't converge"
    #     )
    params_hist_ = params_hist_[: (iter_convergence_ + 1)]

    return params, params_hist_
