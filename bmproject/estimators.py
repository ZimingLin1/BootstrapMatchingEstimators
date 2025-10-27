from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.linalg import lstsq
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler  # (unused; kept for parity with earlier envs)
from sklearn.linear_model import LinearRegression  # (unused; kept for parity with earlier envs)


@dataclass
class MBCResult:
    """
    Return container matching the R implementation:
      - est : point estimate
      - se  : conventional standard error
      - AIse: asymptotically linear (influence-function-based) standard error
    """
    est: float
    se: float
    AIse: float


def _as_2d(X: np.ndarray) -> np.ndarray:
    """
    Utility: ensure a 2D array of shape (n, p).
      - If X is 1D, treat it as a single feature and reshape to (-1, 1).
      - Otherwise keep its shape.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _poly_deg2_with_interactions_Rorder(X: np.ndarray) -> np.ndarray:
    """
    Reproduce R's as.data.frame(poly(X, 2, raw=TRUE)) for a matrix X,
    with the specific column order used there.

    For p input features, the columns (by j = 0..p-1) are:
      [ x_j,  (for k in 0..j-1: x_k * x_j),  x_j^2 ]

    Total columns: p (linear) + p(p-1)/2 (interactions) + p (squares) = p(p+3)/2.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    cols = []
    for j in range(p):
        # linear
        cols.append(X[:, j])
        # interactions with earlier variables (k increasing)
        for k in range(j):
            cols.append(X[:, k] * X[:, j])
        # square
        cols.append(X[:, j] ** 2)
    return np.column_stack(cols)


def series(Y: np.ndarray, X: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    """
    series() consistent with the R version, but with more robust numerics:
      1) Standardize columns of X using training data stats; then build
         degree-2 polynomial features with interactions.
      2) Fit via least squares; if SVD fails to converge, fall back
         to pseudoinverse (or optionally a tiny ridge).
    """
    import numpy as np
    from numpy.linalg import lstsq

    X = np.asarray(X, dtype=float)
    X_eval = np.asarray(X_eval, dtype=float)
    y = np.asarray(Y, dtype=float).reshape(-1)

    # Standardization parameters from training X
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1, keepdims=True)
    sd[(sd == 0) | ~np.isfinite(sd)] = 1.0

    Xs = (X - mu) / sd
    Xes = (X_eval - mu) / sd

    Phi      = _poly_deg2_with_interactions_Rorder(Xs)
    Phi_eval = _poly_deg2_with_interactions_Rorder(Xes)

    # Add intercept
    Phi      = np.column_stack([np.ones(Phi.shape[0]), Phi])
    Phi_eval = np.column_stack([np.ones(Phi_eval.shape[0]), Phi_eval])

    # Clean non-finite rows (extreme bootstrap cases can propagate NaNs/Infs)
    if not np.isfinite(Phi).all() or not np.isfinite(y).all():
        mask = np.isfinite(Phi).all(axis=1) & np.isfinite(y)
        Phi = Phi[mask]
        y = y[mask]

    # Try ordinary least squares; on failure, fall back to pinv (or tiny ridge)
    try:
        beta, *_ = lstsq(Phi, y, rcond=None)
    except np.linalg.LinAlgError:
        # Option A: pseudoinverse
        beta = np.linalg.pinv(Phi, rcond=1e-12) @ y
        # Option B (optional): ultra-weak ridge, if ever needed
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
    Python re-implementation of the R function mbc():

      Inputs
      ------
      - X : covariate matrix (n, p)
      - Y : outcome vector (n,)
      - Tr: treatment indicator (n,), 1 = treated, 0 = control
      - M : number of neighbors in KNN
      - Model1, Model0 : optional working-model predictions of
          E[Y | Tr=1, X] and E[Y | Tr=0, X] for the full sample.
          If omitted, they default to zeros (as in the R code).

      Outputs
      -------
      - est : ATE estimate (same formula as the R version)
      - se  : conventional standard error
      - AIse: asymptotically linear (influence-function) standard error
    """

    # ----- Convert to numpy arrays and check shapes -----
    X = _as_2d(X).astype(float)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    Tr = np.asarray(Tr, dtype=int).reshape(-1)

    # Basic checks
    n = X.shape[0]
    assert Y.shape[0] == n and Tr.shape[0] == n, "Lengths of X, Y, Tr must match."
    assert np.all(np.isin(Tr, [0, 1])), "Tr must be binary in {0,1}."

    # ----- Split by treatment arms -----
    mask1 = (Tr == 1)
    mask0 = (Tr == 0)
    X1, X0 = X[mask1], X[mask0]
    Y1, Y0 = Y[mask1], Y[mask0]
    N1, N0 = X1.shape[0], X0.shape[0]
    N = N1 + N0

    # ----- Standardize whole table (as R's scale does) to mean 0 / unit sd -----
    Xc = X - X.mean(axis=0, keepdims=True)
    sd = Xc.std(axis=0, ddof=1, keepdims=True)
    sd_safe = sd.copy()
    sd_safe[(sd_safe == 0) | ~np.isfinite(sd_safe)] = 1.0
    X_scaled = Xc / sd_safe

    X1 = X_scaled[mask1]
    X0 = X_scaled[mask0]

    # ----- KNN indices: Index1 / Index0 (equivalent to FNN::knnx.index) -----
    # Index1: for each control X0 row, find its M nearest neighbors from treated X1 (kd_tree)
    # Index0: for each treated X1 row, find its M nearest neighbors from control X0
    # Indices are 0-based (R is 1-based).
    nn_1 = NearestNeighbors(n_neighbors=M, algorithm='kd_tree').fit(X1)
    Index1 = nn_1.kneighbors(X0, return_distance=False)  # shape (N0, M), in [0, N1-1]

    nn_0 = NearestNeighbors(n_neighbors=M, algorithm='kd_tree').fit(X0)
    Index0 = nn_0.kneighbors(X1, return_distance=False)  # shape (N1, M), in [0, N0-1]

    # ----- K1M / K0M: frequency of being used as a neighbor (tabulate-style) -----
    # R: K1M = tabulate(c(Index1), nbins = N1)/M
    # Python: np.bincount over [0..N1-1], then divide by M
    K1M = np.bincount(Index1.ravel(), minlength=N1) / float(M)
    K0M = np.bincount(Index0.ravel(), minlength=N0) / float(M)

    # ----- Working models (Model1 / Model0) -----
    # If not provided, default to zeros (as in the R code).
    if Model1 is None:
        Model1 = np.zeros(n, dtype=float)
    else:
        Model1 = np.asarray(Model1, dtype=float).reshape(-1)
        assert Model1.shape[0] == n, "Model1 must have length n."

    if Model0 is None:
        Model0 = np.zeros(n, dtype=float)
    else:
        Model0 = np.asarray(Model0, dtype=float).reshape(-1)
        assert Model0.shape[0] == n, "Model0 must have length n."

    # ----- Residual-weight terms (matching the R code) -----
    Res1 = (1.0 + K1M) * (Y[mask1] - Model1[mask1])  # treated arm
    Res0 = (1.0 + K0M) * (Y[mask0] - Model0[mask0])  # control arm

    # ----- ATE point estimate -----
    # R: est = mean(Model1-Model0) + mean(c(Res1, -Res0))
    est = float(np.mean(Model1 - Model0) + np.mean(np.concatenate([Res1, -Res0])))

    # ----- Conventional standard error -----
    # R: se = sqrt(mean(c(((Model1-Model0-est)[Tr==1]+Res1)^2, ((Model1-Model0-est)[Tr==0]-Res0)^2))/N)
    part1 = ((Model1 - Model0 - est)[mask1] + Res1) ** 2
    part0 = ((Model1 - Model0 - est)[mask0] - Res0) ** 2
    se = float(np.sqrt(np.mean(np.concatenate([part1, part0])) / N))

    # ----- AI (influence-function) standard error: two variance components -----
    # 1) AIvar1: uses cross-arm M-NN averages of Y
    #    - Y1hat: for each control X0, the mean Y among its M nearest treated neighbors
    #    - Y0hat: for each treated X1, the mean Y among its M nearest control neighbors
    Y1_neighbors = Y1[Index1]            # shape (N0, M)
    Y0_neighbors = Y0[Index0]            # shape (N1, M)
    Y1hat = Y1_neighbors.mean(axis=1)    # control units' "treated mean"
    Y0hat = Y0_neighbors.mean(axis=1)    # treated units' "control mean"

    # R: AIvar1 = mean(c((Y1hat - Y0 - est)^2, (Y1 - Y0hat - est)^2)) / N
    AIvar1 = np.mean(
        np.concatenate([
            (Y1hat - Y0 - est) ** 2,
            (Y1 - Y0hat - est) ** 2
        ])
    ) / N

    # 2) AIvar2: within-arm variance component via nearest neighbor within the *same* arm
    # In R: knn.index(X1, 1) / knn.index(X0, 1). If the nearest neighbor could be itself,
    # the variance would be zero. To avoid that, use n_neighbors=2 and drop self (first).
    def _self_excluded_nn_index(Xa: np.ndarray) -> np.ndarray:
        # Return, for each row, the index of its nearest *other* sample (length = n_rows)
        nn = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(Xa)
        inds = nn.kneighbors(Xa, return_distance=False)  # shape (n, 2); inds[:,0] is self
        return inds[:, 1]  # take the second column (nearest non-self)

    Indexvar1 = _self_excluded_nn_index(X1)  # (N1,)
    Indexvar0 = _self_excluded_nn_index(X0)  # (N0,)

    # R: varhat1 = (Y1 - Y1[Indexvar1])^2 / 2 ; varhat0 similarly
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
    BCM() corresponding to the R version:
      - Fit series() separately within treated and control arms using
        degree-2 polynomial-with-interactions features; then predict
        on the *full* X to obtain Model1 / Model0.
      - Feed these working models into mbc() to compute the ATE and SEs.
    """
    X = _as_2d(X)
    Y = np.asarray(Y).reshape(-1)
    Tr = np.asarray(Tr).reshape(-1)

    # Fit on arm-specific subsets, then predict on the full X
    Model1 = series(Y[Tr == 1], X[Tr == 1, :], X)
    Model0 = series(Y[Tr == 0], X[Tr == 0, :], X)

    # Hand over to mbc() for the final estimate and standard errors
    return mbc(X, Y, Tr, M, Model1=Model1, Model0=Model0)
