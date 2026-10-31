"""Positionnement GNSS par moindres carrés sur pseudodistances."""

import numpy as np
from numpy.linalg import norm


def standalone_positioning(pseudoranges, sv_positions, x0=None, max_iter=10, tol=1e-6):
    """Position du récepteur et décalage d'horloge par Gauss-Newton.

    pseudoranges : (n,) pseudodistances corrigées (m) ; sv_positions : (n, 3) positions des
    satellites (m), dans le repère de la position cherchée ; n >= 4.

    Retourne (X = [x, y, z, c.dtu] en m, norme des résidus en m, HDOP sur les composantes x et y).
    """
    pseudoranges = np.asarray(pseudoranges, dtype=float)
    sv_positions = np.asarray(sv_positions, dtype=float)
    n = pseudoranges.shape[0]
    assert n >= 4, f"{n} pseudoranges, can not compute any position."

    X = np.zeros(4) if x0 is None else np.array(x0, dtype=float)

    for _ in range(max_iter):
        d = X[:3] - sv_positions
        rho = norm(d, axis=1)
        H = np.hstack((d / rho[:, None], np.ones((n, 1))))
        residuals = pseudoranges - (rho + X[3])
        dX, *_ = np.linalg.lstsq(H, residuals, rcond=None)
        X = X + dX
        if norm(dX) < tol:
            break

    d = X[:3] - sv_positions
    rho = norm(d, axis=1)
    H = np.hstack((d / rho[:, None], np.ones((n, 1))))
    pr_res = pseudoranges - (rho + X[3])
    pr_res_norm = norm(pr_res)

    Q = np.linalg.inv(H.T @ H)
    hdop = np.sqrt(Q[0, 0] + Q[1, 1])

    return X, pr_res_norm, hdop
