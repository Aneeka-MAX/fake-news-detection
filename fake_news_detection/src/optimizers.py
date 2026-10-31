"""Stage 4 - metaheuristic hyper-parameter optimisers.
 * grey_wolf   : Grey Wolf Optimizer (Mirjalili et al., 2014)  -> proposed
 * levy_flight : Levy-flight random-walk search (Mantegna)     -> baseline
Search space: [log10(learning rate), LSTM units, dropout]."""
from math import gamma, pi, sin
import numpy as np

LB = np.array([-4.0, 16.0, 0.0])
UB = np.array([-2.0, 128.0, 0.5])


def decode(x):
    return {"lr": float(10 ** x[0]), "units": int(round(x[1])), "dropout": float(x[2])}


def grey_wolf(fitness, pop=6, iters=5, seed=42, log=print):
    rng = np.random.default_rng(seed)
    dim = len(LB)
    X = rng.uniform(LB, UB, (pop, dim))
    fit = np.array([fitness(decode(x)) for x in X])
    best_x, best_f = X[fit.argmin()].copy(), fit.min()
    history = [float(best_f)]
    log(f"   [GWO] init best loss={best_f:.4f}")
    for t in range(iters):
        order = np.argsort(fit)
        leaders = [X[order[k]].copy() for k in range(3)]       # alpha, beta, delta
        a = 2 * (1 - t / iters)
        for i in range(pop):
            new = np.zeros(dim)
            for L in leaders:
                A = 2 * a * rng.random(dim) - a
                C = 2 * rng.random(dim)
                new += L - A * np.abs(C * L - X[i])
            X[i] = np.clip(new / 3, LB, UB)
            fit[i] = fitness(decode(X[i]))
            if fit[i] < best_f:
                best_x, best_f = X[i].copy(), fit[i]
        history.append(float(best_f))
        log(f"   [GWO] iter {t + 1}/{iters} best loss={best_f:.4f}")
    return decode(best_x), history


def _levy(rng, dim, beta=1.5):
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u, v = rng.normal(0, sigma, dim), rng.normal(0, 1, dim)
    return np.clip(u / np.abs(v) ** (1 / beta), -10, 10)


def levy_flight(fitness, pop=6, iters=5, seed=42, log=print):
    rng = np.random.default_rng(seed)
    dim = len(LB)
    X = rng.uniform(LB, UB, (pop, dim))
    fit = np.array([fitness(decode(x)) for x in X])
    best_x, best_f = X[fit.argmin()].copy(), fit.min()
    history = [float(best_f)]
    log(f"   [LF ] init best loss={best_f:.4f}")
    for t in range(iters):
        for i in range(pop):
            step = 0.05 * (UB - LB) * _levy(rng, dim)
            cand = np.clip(X[i] + step + 0.3 * rng.random() * (best_x - X[i]), LB, UB)
            f = fitness(decode(cand))
            if f < fit[i]:                     # greedy acceptance
                X[i], fit[i] = cand, f
                if f < best_f:
                    best_x, best_f = cand.copy(), f
        history.append(float(best_f))
        log(f"   [LF ] iter {t + 1}/{iters} best loss={best_f:.4f}")
    return decode(best_x), history
