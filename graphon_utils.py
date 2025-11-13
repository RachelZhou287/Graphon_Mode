import numpy as np

# ============================================================
# graphon_utils.py
# Utility functions for computing graphon values w(x, y)
# for constant, powerlaw, star, and piecewise graphons.
# ============================================================


# ------------------------------------------------------------
# Constant Graphon: w(x, y) = p
# ------------------------------------------------------------
def constant_graphon(x_vec, y_vec, p):
    """
    x_vec: (n_x,) array
    y_vec: (n_y,) array
    Returns: (n_x, n_y) matrix
    """
    n_x = x_vec.shape[0]
    n_y = y_vec.shape[0]
    return np.full((n_x, n_y), p, dtype=np.float64)
# ------------------------------------------------------------
# Power-law Graphon: w(x, y) = (x*y)^(-g)
# ------------------------------------------------------------
def powerlaw_graphon(x_vec, y_vec, g):
    """
    Compute power-law graphon: w(x,y) = (x*y)^(-g)

    x_vec: (n,) array
    y_vec: (n,) array
    g: float
    """
    X = x_vec.reshape(-1, 1)
    Y = y_vec.reshape(1, -1)

    eps = 1e-8
    Xc = np.clip(X, eps, 1.0)
    Yc = np.clip(Y, eps, 1.0)

    W = (Xc * Yc) ** (-g)
    return W.astype(np.float64)


# ------------------------------------------------------------
# Star Graphon
# Hubs = agents with id >= 1 - hub_threshold
# ------------------------------------------------------------
def star_graphon(x_vec, y_vec, hub_threshold):
    """
    star graphon:
        High weight if either node is a hub.
    """
    X = x_vec.reshape(-1, 1)
    Y = y_vec.reshape(1, -1)

    hubs_x = (X > (1.0 - hub_threshold)).astype(float)
    hubs_y = (Y > (1.0 - hub_threshold)).astype(float)

    W = np.where(hubs_x + hubs_y > 0, 1.0, 0.3)
    return W.astype(np.float64)


# ------------------------------------------------------------
# Piecewise Graphon (K-block)
# ------------------------------------------------------------
def piecewise_graphon(x_vec, y_vec, Wblocks, group_bounds):
    n_x = x_vec.shape[0]
    n_y = y_vec.shape[0]
    K = len(group_bounds) - 1

    # Find groups
    def group_of(z):
        for k in range(K):
            if group_bounds[k] <= z < group_bounds[k+1]:
                return k
        return K-1  # safe fallback

    group_x = np.array([ group_of(a) for a in x_vec ])
    group_y = np.array([ group_of(b) for b in y_vec ])

    W = np.zeros((n_x, n_y))
    for i in range(n_x):
        for j in range(n_y):
            W[i,j] = Wblocks[group_x[i]][group_y[j]]

    return W
