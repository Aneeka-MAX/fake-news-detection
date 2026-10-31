"""Stage 3 - Probabilistic PCA fitted with the EM algorithm
(Tipping & Bishop, 1999). Latent model:  x = W z + mu + eps,  eps ~ N(0, s2 I)."""
import numpy as np


class PPCA:
    def __init__(self, n_components=50, max_iter=30, tol=1e-4, seed=42):
        self.M, self.max_iter, self.tol, self.seed = n_components, max_iter, tol, seed

    def fit(self, X, log=print):
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        self.mu_ = X.mean(axis=0)
        Xc = X - self.mu_
        rng = np.random.default_rng(self.seed)
        W = rng.normal(scale=0.1, size=(D, self.M))
        s2 = 1.0
        prev = None
        total_var = (Xc ** 2).sum()
        for it in range(self.max_iter):
            # E-step
            Minv = np.linalg.inv(W.T @ W + s2 * np.eye(self.M))
            Ez = Minv @ W.T @ Xc.T                       # (M, N)
            Ezz = N * s2 * Minv + Ez @ Ez.T              # (M, M)
            # M-step
            W_new = (Xc.T @ Ez.T) @ np.linalg.inv(Ezz)   # (D, M)
            s2 = (total_var - 2 * np.sum((W_new.T @ Xc.T) * Ez)
                  + np.trace(Ezz @ (W_new.T @ W_new))) / (N * D)
            s2 = max(s2, 1e-8)
            W = W_new
            if prev is not None and abs(prev - s2) < self.tol * prev:
                break
            prev = s2
        log(f"   PPCA converged after {it + 1} EM iterations (sigma^2={s2:.5f})")
        self.W_, self.s2_ = W, s2
        self._Minv = np.linalg.inv(W.T @ W + s2 * np.eye(self.M))
        Z = self.transform(X)
        self.z_mean_, self.z_std_ = Z.mean(0), Z.std(0) + 1e-8
        return self

    def transform(self, X, batch=4096):
        out = []
        for i in range(0, X.shape[0], batch):
            Xc = np.asarray(X[i:i + batch], dtype=np.float64) - self.mu_
            out.append(Xc @ self.W_ @ self._Minv.T)      # posterior mean E[z|x]
        Z = np.vstack(out)
        if hasattr(self, "z_mean_"):
            Z = (Z - self.z_mean_) / self.z_std_
        return Z.astype(np.float32)

    def save(self, path):
        np.savez(path, mu=self.mu_, W=self.W_, s2=self.s2_,
                 zm=self.z_mean_, zs=self.z_std_)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        o = cls(n_components=d["W"].shape[1])
        o.mu_, o.W_, o.s2_ = d["mu"], d["W"], float(d["s2"])
        o._Minv = np.linalg.inv(o.W_.T @ o.W_ + o.s2_ * np.eye(o.M))
        o.z_mean_, o.z_std_ = d["zm"], d["zs"]
        return o
