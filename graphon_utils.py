# graphon_utils.py
import numpy as np

def build_graphon(cfg):
    """
    Return a graphon function w(x,y) depending on cfg["graphon_mode"].
    """

    mode = cfg.get("graphon_mode", "constant")

    if mode == "constant":
        p = cfg.get("p", 1.0)
        return lambda x, y: p

    elif mode == "powerlaw":
        g = cfg.get("g", -0.2)
        return lambda x, y: (x * y) ** (-g)

    elif mode == "star":
        threshold = cfg.get("hub_threshold", 0.1)
        return lambda x, y: 1.0 if (x < threshold or y < threshold) else 0.1

    elif mode == "piecewise":
        groups = np.array(cfg["piecewise"]["groups"], dtype=float)
        conn = np.array(cfg["piecewise"]["connection_matrix"], dtype=float)

        # --- Sanity checks ---
        assert np.isclose(groups.sum(), 1.0), \
            f"Piecewise group proportions must sum to 1. Got {groups.sum()}"

        n_groups = len(groups)
        assert conn.shape == (n_groups, n_groups), \
            f"Connection matrix must be {n_groups}x{n_groups}, got {conn.shape}"

        # Boundaries of cumulative proportions
        boundaries = np.cumsum(groups)

        def group_idx(val: float) -> int:
            """Assign an agent in [0,1] to a group index (0..n_groups-1)."""
            idx = np.searchsorted(boundaries, val, side="right")
            return min(idx, n_groups-1)  # clamp last agent into final group

        def w(x, y):
            i, j = group_idx(x), group_idx(y)
            return conn[i, j]

        return w


    else:
        raise ValueError(f"Unknown graphon_mode {mode}")
