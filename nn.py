"""Tiny numpy MLP (tanh hidden layers, linear output) with Adam. Weights ship as .npz, so no torch is needed."""
import numpy as np


class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.standard_normal((a, b)) * np.sqrt(1.0 / a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.mu = np.zeros(sizes[0])
        self.sd = np.ones(sizes[0])
        self.ymu = 0.0
        self.ysd = 1.0

    def _fwd(self, x):
        acts = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.tanh(x)
            acts.append(x)
        return acts

    def predict(self, X):
        X = np.atleast_2d(X)
        out = self._fwd((X - self.mu) / self.sd)[-1]
        return out * self.ysd + self.ymu

    def fit(self, X, Y, mask=None, epochs=30, batch=512, lr=2e-3, wd=1e-5, seed=0, val=None, log=print):
        """Masked MSE on standardised targets. mask (same shape as Y) is 1 where a target exists."""
        rng = np.random.default_rng(seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        if mask is None:
            mask = np.ones_like(Y)
        self.ymu = float((Y * mask).sum() / mask.sum())
        self.ysd = float(np.sqrt((((Y - self.ymu) ** 2) * mask).sum() / mask.sum()) + 1e-6)
        Xs, Ys = (X - self.mu) / self.sd, (Y - self.ymu) / self.ysd
        m = [np.zeros_like(p) for p in self.W + self.b]
        v = [np.zeros_like(p) for p in self.W + self.b]
        t = 0
        N = len(Xs)
        for ep in range(epochs):
            perm = rng.permutation(N)
            cur_lr = lr * (0.5 * (1 + np.cos(np.pi * ep / epochs)) * 0.95 + 0.05)
            tot = 0.0
            for i in range(0, N, batch):
                idx = perm[i:i + batch]
                acts = self._fwd(Xs[idx])
                diff = (acts[-1] - Ys[idx]) * mask[idx]
                tot += float((diff ** 2).sum())
                g = 2 * diff / max(1.0, mask[idx].sum() / mask.shape[1] if mask.ndim > 1 else len(idx))
                gW, gb = [None] * len(self.W), [None] * len(self.W)
                for l in range(len(self.W) - 1, -1, -1):
                    gW[l] = acts[l].T @ g + wd * self.W[l]
                    gb[l] = g.sum(0)
                    if l > 0:
                        g = (g @ self.W[l].T) * (1 - acts[l] ** 2)
                t += 1
                params, grads = self.W + self.b, gW + gb
                for k, (p, gr) in enumerate(zip(params, grads)):
                    m[k] = 0.9 * m[k] + 0.1 * gr
                    v[k] = 0.999 * v[k] + 0.001 * gr * gr
                    p -= cur_lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
            if log and (ep % 5 == 0 or ep == epochs - 1):
                msg = f"epoch {ep:3d} train mse {tot / mask.sum():.4f}"
                if val is not None:
                    Xv, Yv, Mv = val
                    pv = self.predict(Xv)
                    msg += f"  val rmse {np.sqrt((((pv - Yv) ** 2) * Mv).sum() / Mv.sum()):.3f}"
                log(msg)

    def save(self, path):
        d = {f"W{i}": w for i, w in enumerate(self.W)}
        d.update({f"b{i}": b for i, b in enumerate(self.b)})
        d.update(mu=self.mu, sd=self.sd, ymu=np.array(self.ymu), ysd=np.array(self.ysd))
        np.savez(path, **d)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        n = sum(1 for k in z.files if k.startswith("W"))
        o = cls.__new__(cls)
        o.W = [z[f"W{i}"] for i in range(n)]
        o.b = [z[f"b{i}"] for i in range(n)]
        o.mu, o.sd, o.ymu, o.ysd = z["mu"], z["sd"], float(z["ymu"]), float(z["ysd"])
        return o
