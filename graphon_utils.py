import numpy as np
import tensorflow as tf

#graphon_utils.py
def build_graphon(cfg):
    """
    Build and return a graphon function w(x, y) based on the configuration.
    Supports TensorFlow operations for differentiability when used inside TF graphs.
    """

    mode = cfg.get("graphon_mode", "constant")

    # ------------------------------------------------------------
    # 1) Constant graphon: all connections equal to p
    # ------------------------------------------------------------
    if mode == "constant":
        p = float(cfg.get("p", 1.0))
        def w(x, y):
            return tf.cast(p, tf.float64) if isinstance(x, tf.Tensor) else p
        return w

    # ------------------------------------------------------------
    # 2) Power-law graphon: w(x, y) = (max(x, eps)*max(y, eps))^g
    #    TensorFlow-safe version with clamping for small x,y
    # ------------------------------------------------------------
    elif mode == "powerlaw":
        g = float(cfg.get("g", 0.3))
        eps = float(cfg.get("graphon_eps", 1e-8))

        def w(x, y):
            # Support both TensorFlow tensors and numpy/scalars
            if isinstance(x, tf.Tensor) or isinstance(y, tf.Tensor):
                x_tf = tf.cast(x, tf.float64)
                y_tf = tf.cast(y, tf.float64)
                return tf.pow(tf.maximum(x_tf, eps) * tf.maximum(y_tf, eps), g)
            else:
                xv = np.maximum(np.array(x, dtype=float), eps)
                yv = np.maximum(np.array(y, dtype=float), eps)
                return (xv * yv) ** g

        return w

    # ------------------------------------------------------------
    # 3) Star graphon: a small set of "hub" agents highly connected
    # ------------------------------------------------------------
    elif mode == "star":
        threshold = float(cfg.get("hub_threshold", 0.1))
        high_val = float(cfg.get("hub_value", 1.0))
        low_val = float(cfg.get("nonhub_value", 0.1))

        def w(x, y):
            if isinstance(x, tf.Tensor) or isinstance(y, tf.Tensor):
                x_tf = tf.cast(x, tf.float64)
                y_tf = tf.cast(y, tf.float64)
                cond = tf.logical_or(x_tf < threshold, y_tf < threshold)
                return tf.where(cond,
                                tf.cast(high_val, tf.float64),
                                tf.cast(low_val, tf.float64))
            else:
                return high_val if (x < threshold or y < threshold) else low_val

        return w

    # ------------------------------------------------------------
    # 4) Piecewise graphon: block structure for community models
    # ------------------------------------------------------------
    elif mode == "piecewise":
        groups = np.array(cfg["piecewise"]["groups"], dtype=float)
        conn = np.array(cfg["piecewise"]["connection_matrix"], dtype=float)

        assert np.isclose(groups.sum(), 1.0), \
            f"Piecewise group proportions must sum to 1. Got {groups.sum()}"

        n_groups = len(groups)
        assert conn.shape == (n_groups, n_groups), \
            f"Connection matrix must be {n_groups}x{n_groups}, got {conn.shape}"

        # Compute boundaries (cumulative)
        boundaries = np.cumsum(groups)

        def group_idx(val):
            """Assign an agent in [0,1] to a group index (0..n_groups-1)."""
            if isinstance(val, tf.Tensor):
                # For TensorFlow tensors
                idx = tf.searchsorted(tf.constant(boundaries, dtype=tf.float64),
                                      tf.cast(val, tf.float64),
                                      side="right")
                return tf.minimum(idx, n_groups - 1)
            else:
                # For numpy / scalar inputs
                idx = np.searchsorted(boundaries, val, side="right")
                return min(idx, n_groups - 1)

        def w(x, y):
            i = group_idx(x)
            j = group_idx(y)
            if isinstance(i, tf.Tensor) or isinstance(j, tf.Tensor):
                conn_tf = tf.convert_to_tensor(conn, dtype=tf.float64)
                return tf.gather_nd(conn_tf, tf.stack([i, j], axis=-1))
            else:
                return conn[i, j]

        return w

    # ------------------------------------------------------------
    # 5) Unknown graphon mode
    # ------------------------------------------------------------
    else:
        raise ValueError(f"Unknown graphon_mode: {mode}")
