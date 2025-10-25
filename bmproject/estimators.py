from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
from numpy.linalg import lstsq
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


@dataclass
class MBCResult:
    """与 R 版返回值结构对应:点估计 est、常规标准误 se、影响函数式标准误 AIse。"""
    est: float
    se: float
    AIse: float


def _as_2d(X: np.ndarray) -> np.ndarray:
    """
    工具函数:把输入变成 (n, p) 的二维数组。
    - 如果是一维向量,视作单变量特征,reshape(-1, 1)
    - 其余维度保持不变
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _poly_deg2_with_interactions_Rorder(X: np.ndarray) -> np.ndarray:
    """
    复刻 R: as.data.frame(poly(X, 2, raw=TRUE))（X为矩阵）：
    列顺序（p列特征）：
      for j in [0..p-1]:
        [ x_j,  (for k in [0..j-1]: x_k*x_j),  x_j^2 ]
    共 p(线性) + p(p-1)/2(交叉) + p(平方) = p(p+3)/2 列。
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    cols = []
    for j in range(p):
        # 线性
        cols.append(X[:, j])
        # 与之前变量的交叉项（按 k 从小到大）
        for k in range(j):
            cols.append(X[:, k] * X[:, j])
        # 平方
        cols.append(X[:, j] ** 2)
    return np.column_stack(cols)



def series(Y: np.ndarray, X: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    """
    与 R 版一致的 series()，但更稳健：
    1) 先按训练集对 X 做列标准化，再构造二次多项式+交互；
    2) 回归先用 lstsq；若遇到 SVD 不收敛，回退到 pinv（或轻微岭回归）。
    """
    import numpy as np
    from numpy.linalg import lstsq

    X = np.asarray(X, dtype=float)
    X_eval = np.asarray(X_eval, dtype=float)
    y = np.asarray(Y, dtype=float).reshape(-1)

    # 训练集标准化参数
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1, keepdims=True)
    sd[(sd == 0) | ~np.isfinite(sd)] = 1.0

    Xs = (X - mu) / sd
    Xes = (X_eval - mu) / sd

    Phi      = _poly_deg2_with_interactions_Rorder(Xs)
    Phi_eval = _poly_deg2_with_interactions_Rorder(Xes)

    # 加截距
    Phi      = np.column_stack([np.ones(Phi.shape[0]), Phi])
    Phi_eval = np.column_stack([np.ones(Phi_eval.shape[0]), Phi_eval])

    # 清理异常值（极端情况下自助法可能造成非有限数）
    if not np.isfinite(Phi).all() or not np.isfinite(y).all():
        mask = np.isfinite(Phi).all(axis=1) & np.isfinite(y)
        Phi = Phi[mask]
        y = y[mask]

    # 先尝试常规最小二乘，失败则回退到伪逆/岭
    try:
        beta, *_ = lstsq(Phi, y, rcond=None)
    except np.linalg.LinAlgError:
        # 方案 A：伪逆
        beta = np.linalg.pinv(Phi, rcond=1e-12) @ y
        # 方案 B（可选）：极弱岭回归（如需）
        # lam = 1e-8
        # beta = np.linalg.solve(Phi.T @ Phi + lam * np.eye(Phi.shape[1]), Phi.T @ y)

    return Phi_eval @ beta


def mbc(X: np.ndarray,
        Y: np.ndarray,
        Tr: np.ndarray,
        M: int,
        Model1: Optional[np.ndarray] = None,
        Model0: Optional[np.ndarray] = None) -> MBCResult:
    """
    Python 复现 R 中的 mbc() 主函数:
      - X: 协变量矩阵 (n, p)
      - Y: 结果变量向量 (n,)
      - Tr: 处理指示 (n,), 1 表示处理组,0 表示对照组
      - M: KNN 中每个点的邻居数
      - Model1, Model0: 可选的“工作模型”对 E[Y|Tr=1,X] 与 E[Y|Tr=0,X] 的全样本预测
        若不提供,则相当于 R 里传入 0(即全部为 0 的向量)
    计算输出:
      - est: ATE 估计(与 R 版相同配方)
      - se:   常规标准误
      - AIse: 影响函数型(Asymptotically linear)标准误
    """

    # -------- 预处理:转为 numpy,保证形状 --------
    X = _as_2d(X).astype(float)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    Tr = np.asarray(Tr, dtype=int).reshape(-1)

    # 基本维度检查
    n = X.shape[0]
    assert Y.shape[0] == n and Tr.shape[0] == n, "X, Y, Tr 的长度必须一致"
    assert np.all(np.isin(Tr, [0, 1])), "Tr 必须只包含 0/1"

    # -------- 组内划分 --------
    mask1 = (Tr == 1)
    mask0 = (Tr == 0)
    X1, X0 = X[mask1], X[mask0]
    Y1, Y0 = Y[mask1], Y[mask0]
    N1, N0 = X1.shape[0], X0.shape[0]
    N = N1 + N0

    # -------- 标准化(与 R 的 scale(X) 一致:对整表列均值0、方差1)--------
    Xc = X - X.mean(axis=0, keepdims=True)
    sd = Xc.std(axis=0, ddof=1, keepdims=True)
    sd_safe = sd.copy()
    sd_safe[(sd_safe == 0) | ~np.isfinite(sd_safe)] = 1.0
    X_scaled = Xc / sd_safe

    X1 = X_scaled[mask1]
    X0 = X_scaled[mask0]

    # -------- KNN:Index1/Index0(与 FNN::knnx.index 等价)--------
    # Index1: 对于每个对照样本 X0,找来自处理组 X1 的 M 个最近邻(kd_tree)
    # Index0: 对于每个处理样本 X1,找来自对照组 X0 的 M 个最近邻
    # 返回的是基于“被检索集合”的 0-based 索引(与 R 的 1-based 不同)
    nn_1 = NearestNeighbors(n_neighbors=M, algorithm='kd_tree').fit(X1)
    Index1 = nn_1.kneighbors(X0, return_distance=False)  # 形状 (N0, M),值域在 [0, N1-1]

    nn_0 = NearestNeighbors(n_neighbors=M, algorithm='kd_tree').fit(X0)
    Index0 = nn_0.kneighbors(X1, return_distance=False)  # 形状 (N1, M),值域在 [0, N0-1]

    # -------- 计算 K1M / K0M:某个被“作为邻居”出现的频率(按 R 中 tabulate 逻辑)--------
    # R: K1M = tabulate(c(Index1), nbins = N1)/M
    # Python:np.bincount 统计 0..N1-1 的出现次数,再 / M
    K1M = np.bincount(Index1.ravel(), minlength=N1) / float(M)
    K0M = np.bincount(Index0.ravel(), minlength=N0) / float(M)

    # -------- 工作模型(Model1/Model0)--------
    # 若未传入,按 R 默认设为 0 向量(对 Y 的“中心化”项会直接用 Y)
    if Model1 is None:
        Model1 = np.zeros(n, dtype=float)
    else:
        Model1 = np.asarray(Model1, dtype=float).reshape(-1)
        assert Model1.shape[0] == n, "Model1 的长度应等于样本量 n"

    if Model0 is None:
        Model0 = np.zeros(n, dtype=float)
    else:
        Model0 = np.asarray(Model0, dtype=float).reshape(-1)
        assert Model0.shape[0] == n, "Model0 的长度应等于样本量 n"

    # -------- 残差残差权重项(与 R 保持一致)--------
    Res1 = (1.0 + K1M) * (Y[mask1] - Model1[mask1])   # 处理组
    Res0 = (1.0 + K0M) * (Y[mask0] - Model0[mask0])   # 对照组

    # -------- ATE 点估计 est --------
    # R: est = mean(Model1-Model0) + mean(c(Res1, -Res0))
    est = float(np.mean(Model1 - Model0) + np.mean(np.concatenate([Res1, -Res0])))

    # -------- 常规标准误 se --------
    # R: se = sqrt(mean(c(((Model1-Model0-est)[Tr==1]+Res1)^2, ((Model1-Model0-est)[Tr==0]-Res0)^2))/N)
    part1 = ((Model1 - Model0 - est)[mask1] + Res1) ** 2
    part0 = ((Model1 - Model0 - est)[mask0] - Res0) ** 2
    se = float(np.sqrt(np.mean(np.concatenate([part1, part0])) / N))

    # -------- AI 标准误(两部分方差相加)--------
    # 1) AIvar1:基于“跨组 M 个邻居的 Y 平均值”构造
    #   - Y1hat: 对每个对照样本 X0,取其在处理组 X1 的 M 个最近邻的 Y(处理组)均值
    #   - Y0hat: 对每个处理样本 X1,取其在对照组 X0 的 M 个最近邻的 Y(对照组)均值
    Y1_neighbors = Y1[Index1]            # 形状 (N0, M)
    Y0_neighbors = Y0[Index0]            # 形状 (N1, M)
    Y1hat = Y1_neighbors.mean(axis=1)    # 对照样本对应的“处理组”均值
    Y0hat = Y0_neighbors.mean(axis=1)    # 处理样本对应的“对照组”均值

    # R: AIvar1 = mean(c((Y1hat - Y0 - est)^2, (Y1 - Y0hat - est)^2)) / N
    AIvar1 = np.mean(
        np.concatenate([
            (Y1hat - Y0 - est) ** 2,
            (Y1 - Y0hat - est) ** 2
        ])
    ) / N

    # 2) AIvar2:方差项(需要“组内最近邻”的方差估计 varhat1/varhat0)
    # R 里用 knn.index(X1, 1) / knn.index(X0, 1)
    # 注意:同集内 1 近邻若包含“自身”,方差会成为 0。
    # 为稳妥,我们取 n_neighbors=2,然后丢掉每个点“自己”(距离 0 的第 1 个),
    # 以获得“最近的其他样本”的索引。
    def _self_excluded_nn_index(Xa: np.ndarray) -> np.ndarray:
        # 返回每个样本最近的“另一个样本”的索引(长度 = 样本数)
        nn = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(Xa)
        inds = nn.kneighbors(Xa, return_distance=False)  # 形状 (n, 2),inds[:,0] 为自身
        return inds[:, 1]  # 取每行第 2 个,排除自身

    Indexvar1 = _self_excluded_nn_index(X1)  # (N1,)
    Indexvar0 = _self_excluded_nn_index(X0)  # (N0,)

    # R: varhat1 = (Y1 - Y1[Indexvar1])^2 / 2 ；varhat0 同理
    varhat1 = (Y1 - Y1[Indexvar1]) ** 2 / 2.0
    varhat0 = (Y0 - Y0[Indexvar0]) ** 2 / 2.0

    # R: AIvar2 = mean(c((K1M^2+(2-1/M)*K1M)*varhat1,
    #                    (K0M^2+(2-1/M)*K0M)*varhat0)) / N
    coef1 = (K1M ** 2 + (2.0 - 1.0 / M) * K1M)
    coef0 = (K0M ** 2 + (2.0 - 1.0 / M) * K0M)
    AIvar2 = np.mean(np.concatenate([coef1 * varhat1, coef0 * varhat0])) / N

    AIse = float(np.sqrt(AIvar1 + AIvar2))

    return MBCResult(est=est, se=se, AIse=AIse)



def BCM(X: np.ndarray, Y: np.ndarray, Tr: np.ndarray, M: int) -> MBCResult:
    """
    对应 R 中的 BCM():
      - 用 series() 分别在处理组/对照组上训练“二次多项式回归”,
        然后对“全样本 X”做预测,得到 Model1 / Model0
      - 把上述工作模型丢给 mbc() 计算 ATE 与标准误
    """
    X = _as_2d(X)
    Y = np.asarray(Y).reshape(-1)
    Tr = np.asarray(Tr).reshape(-1)

    # 基于各自子样本训练,再对全样本 X 预测
    Model1 = series(Y[Tr == 1], X[Tr == 1, :], X)
    Model0 = series(Y[Tr == 0], X[Tr == 0, :], X)
    
    # 调用 mbc 完成估计
    return mbc(X, Y, Tr, M, Model1=Model1, Model0=Model0)

