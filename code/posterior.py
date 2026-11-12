"""General algos and proba tools for Layers 2 and 3"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import betaln, expit, log_expit, logit, logsumexp


#  Beta updates with EM algo

def beta_mode(successes, trials, alpha, beta):
    """Return the mode of the Beta posterior after x successes out of y trials"""
    return (successes + alpha - 1) / (trials + alpha + beta - 2)


def log_odds_prior(prior):
    """Return the Beta prior whose mode on the 0-1 scale is the log-odds mode"""
    return {name: (alpha + 1, beta + 1) for name, (alpha, beta) in prior.items()}


def em_from_starts(em, starts, skips, mask, prior):
    """Run EM from several starts; return a table, the best mode and every history"""
    # POUR ANA: j'ai mis em en argument comme ça tu peux l'adapter à un markov model
    rows = []
    modes = []
    histories = []
    for start in starts:
        mode, history = em(start, skips, mask, prior=prior)
        modes.append(mode)
        histories.append(history)
        rows.append({
            **{name: float(value) for name, value in mode.items()},
            "log-posterior": history[-1],
            # EM never decreases the log-posterior so a drop can only be rounding error (0 means it never drops)
            "largest decrease": max(0.0, float(-np.diff(history).min())),
        })
    table = pd.DataFrame(rows)
    best = modes[int(table["log-posterior"].idxmax())]
    return table, best, histories


#  Log-odds scale to avoid underflow 

def to_vector(params, names):
    """Turn a dict of probabilities into a vector of log-odds"""
    return np.array([logit(params[name]) for name in names])


def to_params(phi, names):
    """Turn log-odds (one vector, or one row per draw) into a dict of probabilities"""
    phi = np.asarray(phi)
    return {name: expit(phi[..., i]) for i, name in enumerate(names)}


def log_beta_prior_log_odds(phi, prior, names):
    """Return the log Beta prior density written on the log-odds scale"""
    # Beta density p^(alpha-1) (1-p)^(beta-1) / B(alpha, beta), times the
    # Jacobian dp/dphi = p (1 - p) (Ch. 2, Ch. 4 Ex. 4.1), gives
    # p^alpha (1-p)^beta / B(alpha, beta)
    # log_expit(phi) = log p and log_expit(-phi) = log(1 - p), computed without
    # rounding p to exactly 0 or 1 when phi is large
    phi = np.asarray(phi)
    alpha = np.array([prior[name][0] for name in names])
    beta = np.array([prior[name][1] for name in names])
    terms = alpha * log_expit(phi) + beta * log_expit(-phi) - betaln(alpha, beta)
    return terms.sum(axis=-1)


#  Derivatives by finite differences 

def numerical_gradient(f, x, step=1e-4):
    """Return the gradient of f at x by central differences"""
    gradient = np.zeros(len(x))
    for i in range(len(x)):
        shift = np.zeros(len(x))
        shift[i] = step
        gradient[i] = (f(x + shift) - f(x - shift)) / (2 * step)
    return gradient


def numerical_hessian(f, x, step=1e-3):
    """Return the matrix of second derivatives of f at x by central differences"""
    # One formula for every pair (i, j) with i <= j:
    # [f(x+hi+hj) - f(x+hi-hj) - f(x-hi+hj) + f(x-hi-hj)] / (4 h^2)
    # A single loop over the pairs, the lower half is a copy (symmetry)
    d = len(x)
    hessian = np.zeros((d, d))
    for i, j in zip(*np.triu_indices(d)):
        hi = np.zeros(d)
        hj = np.zeros(d)
        hi[i] = step
        hj[j] = step
        hessian[i, j] = (f(x + hi + hj) - f(x + hi - hj)
                         - f(x - hi + hj) + f(x - hi - hj)) / (4 * step ** 2)
        hessian[j, i] = hessian[i, j]
    return hessian


# Proposals 

def student_t_draws(center, shape, df, n, rng):
    """Draw n points from a multivariate Student-t"""
    # Ch. 2: a Normal divided by sqrt(chi-square / df) is a Student-t
    # L L^T = shape, so L z has covariance `shape` when z ~ N(0, I)
    # The small chi-square values give the heavy tails
    L = np.linalg.cholesky(shape)
    z = rng.standard_normal((n, len(center)))
    w = rng.chisquare(df, size=n)
    return center + (z @ L.T) / np.sqrt(w / df)[:, None]


def student_t_log_density(x, center, shape, df):
    """Return the log density of the multivariate Student-t at each row of x"""
    return stats.multivariate_t.logpdf(x, loc=center, shape=shape, df=df)


def normal_draws(center, cov, n, rng):
    """Draw n points from a multivariate Normal"""
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n, len(center)))
    return center + z @ L.T


def normal_log_density(x, center, cov):
    """Return the log density of the multivariate Normal at each row of x"""
    return stats.multivariate_normal.logpdf(x, mean=center, cov=cov)


def prior_draws_log_odds(prior, names, n, rng):
    """Draw n points from the prior and return them on the log-odds scale"""
    return np.column_stack([logit(rng.beta(*prior[name], size=n)) for name in names])


# Importance sampling (Ch. 3) 

def normalized_weights(log_weights):
    """Turn log-weights into weights that sum to 1 (all 0 if no draw is valid)"""
    # A draw outside the support (e.g. theta_e >= theta_b) has log-weight -inf
    # If every draw is like that, there is nothing to normalize
    if not np.isfinite(log_weights).any():
        return np.zeros(len(log_weights))
    # log-sum-exp: subtract the largest log-weight before exp, so nothing overflows
    return np.exp(log_weights - logsumexp(log_weights))


def importance_sampling(log_posterior, draws, log_proposal):
    """Weight proposal draws by posterior / proposal; return weights and diagnostics"""
    # One loop over draws: log_posterior takes one vector at a time
    log_target = np.array([log_posterior(phi) for phi in draws])
    log_weights = log_target - log_proposal
    weights = normalized_weights(log_weights)
    ess = effective_sample_size(weights)
    return {
        "draws": draws,
        "log_weights": log_weights,
        "weights": weights,
        "n_valid": int(np.isfinite(log_weights).sum()),
        "ess": ess,
        "ess_share": ess / len(draws),
        "largest_weight": weights.max(),
    }


def importance_sampling_at_mode(log_posterior, center, n_draws, rng, df=4):
    """Importance sampling with a Student-t proposal centred at the mode, scale (-Hessian)^-1"""
    # Near the mode the log-posterior is close to a parabola, whose curvature
    # is the Hessian; minus its inverse is the covariance of the Normal
    # approximation. The Student-t has the same centre and shape, with heavier
    # tails than the target (Ch. 3, Prop. 3.3)
    hessian = numerical_hessian(log_posterior, center)
    shape = np.linalg.inv(-hessian)
    draws = student_t_draws(center, shape, df, n_draws, rng)
    result = importance_sampling(log_posterior, draws,
                                 student_t_log_density(draws, center, shape, df))
    result.update(center=center, hessian=hessian, shape=shape)
    return result


def proposal_diagnostics(log_posterior, result):
    """Return checks of the importance sampling run: mode, Hessian and weights"""
    center = result["center"]
    # The Hessian should barely move when the finite-difference step is halved
    half_step = numerical_hessian(log_posterior, center, step=5e-4)
    change = np.abs(half_step - result["hessian"]).max() / np.abs(result["hessian"]).max()
    n_draws = len(result["draws"])
    return pd.Series({
        "largest |gradient| at the centre": np.abs(numerical_gradient(log_posterior, center)).max(),
        "Hessian change when the step is halved": change,
        "smallest eigenvalue of -Hessian": np.linalg.eigvalsh(-result["hessian"]).min(),
        "draws outside the support (weight 0)": n_draws - result["n_valid"],
        "draws": n_draws,
        "ESS": result["ess"],
        "ESS share": result["ess_share"],
        "largest weight": result["largest_weight"],
    })


def effective_sample_size(weights):
    """Return ESS = 1 / sum of squared normalized weights (0 if no valid draw)"""
    # Equal weights 1/N give ESS = N; one weight of 1 gives ESS = 1 (Ch. 3)
    total = (weights ** 2).sum()
    return 0.0 if total == 0 else 1 / total


def weighted_quantile(values, weights, q):
    """Return the q-quantiles of values under the weights"""
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    index = np.searchsorted(cumulative, q)
    return values[order][np.minimum(index, len(values) - 1)]


def monte_carlo_standard_error(values, weights):
    """Return the Monte Carlo standard error of a self-normalized weighted mean"""
    # Delta-method formula (Robert & Casella, 2004): sqrt(sum w^2 (h - h_hat)^2)
    # With equal weights 1/N it is the usual sd / sqrt(N)
    estimate = (weights * values).sum()
    return np.sqrt((weights ** 2 * (values - estimate) ** 2).sum())


def weighted_summary(values, weights):
    """Return the weighted mean, sd, 2.5/50/97.5% quantiles and MC standard error"""
    if weights.sum() == 0:
        raise ValueError("No valid draw: every weight is zero, so there is no summary")
    mean = (weights * values).sum()
    low, median, high = weighted_quantile(values, weights, [0.025, 0.5, 0.975])
    return {
        "mean": mean,
        "sd": np.sqrt((weights * (values - mean) ** 2).sum()),
        "low": low,
        "median": median,
        "high": high,
        "mcse": monte_carlo_standard_error(values, weights),
    }


def resample(draws, weights, n, rng):
    """Draw n rows from the weighted draws, giving equally weighted posterior draws"""
    # Each row is picked with probability equal to its weight (Ch. 3)
    index = rng.choice(len(draws), size=n, p=weights)
    return draws[index]


def posterior_table(result, names, truth=None):
    """Return weighted posterior summaries, one row per parameter (0-1 scale)"""
    params = to_params(result["draws"], names)
    rows = {name: weighted_summary(params[name], result["weights"]) for name in names}
    table = pd.DataFrame(rows).T
    if truth is not None:
        table["true value"] = pd.Series(truth)
        table["inside 95%"] = (table["low"] <= table["true value"]) & (table["true value"] <= table["high"])
    return table


def weighted_correlation(result, names):
    """Return the posterior correlation matrix of the parameters (0-1 scale)"""
    params = to_params(result["draws"], names)
    values = np.column_stack([params[name] for name in names])
    weights = result["weights"]
    centred = values - weights @ values
    covariance = (weights[:, None] * centred).T @ centred
    sd = np.sqrt(np.diag(covariance))
    return pd.DataFrame(covariance / np.outer(sd, sd), index=names, columns=names)


#  Posterior predictive checks (Ch. 1) 

def predictive_checks(simulate, params_draws, lengths, rng, check):
    """For each posterior draw, simulate data with the same lengths and compute the checks"""
    # simulate(params, lengths, rng) returns skips, mask and the hidden states
    # check(skips, mask, rng) returns a dict of statistics
    rows = []
    for params in params_draws:
        skips, mask, _ = simulate(params, lengths, rng)
        rows.append(check(skips, mask, rng))
    return pd.DataFrame(rows)


def predictive_table(simulated, observed):
    """Compare the real statistics with their simulated distribution, one row per statistic"""
    # The last column is the share of simulations at least as large as the real
    # value; close to 0 or 1 means the model cannot reproduce this statistic
    return pd.DataFrame({
        "real data": pd.Series(observed),
        "simulated mean": simulated.mean(),
        "simulated 2.5%": simulated.quantile(0.025),
        "simulated 97.5%": simulated.quantile(0.975),
        "share at least as large": (simulated >= pd.Series(observed)).mean(),
    })
