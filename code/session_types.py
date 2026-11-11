"""Layer 2 model: two session types (Engaged, Browsing), no switching

Each session has one hidden type for all its tracks
Given the type, tracks are skipped independently with rate theta_e or theta_b
hmm.py (Layer 3) uses the same function names and inputs
"""

import numpy as np
from scipy import stats

from posterior import beta_mode, log_beta_prior_log_odds, to_params

# Order of the parameters in the log-odds vector used by the samplers
NAMES = ["theta_e", "theta_b", "p0"]

# Beta(2, 2) for every parameter: weakly informative, keeps values away from 0 and 1
PRIOR = {"theta_e": (2, 2), "theta_b": (2, 2), "p0": (2, 2)}


def session_counts(skips, mask):
    """Return the number of tracks n and of skips s of each session"""
    # Given the type, the order of the tracks does not matter,
    # so these two counts are all the model needs (Ch. 1, sufficient statistics)
    n = mask.sum(axis=1)
    s = (skips * mask).sum(axis=1)
    return n, s


def log_terms_by_type(params, n, s):
    """Return log P(type, skips) for Engaged and for Browsing, one value per session"""
    # P(Engaged) * theta_e^s * (1 - theta_e)^(n - s), written in logs so that
    # 20 small numbers multiplied together do not round to 0 (Ch. 1, underflow)
    log_engaged = (np.log(params["p0"])
                   + s * np.log(params["theta_e"])
                   + (n - s) * np.log(1 - params["theta_e"]))
    log_browsing = (np.log(1 - params["p0"])
                    + s * np.log(params["theta_b"])
                    + (n - s) * np.log(1 - params["theta_b"]))
    return log_engaged, log_browsing


def log_likelihood(params, skips, mask):
    """Return the log-likelihood of all sessions"""
    n, s = session_counts(skips, mask)
    log_engaged, log_browsing = log_terms_by_type(params, n, s)
    # p(y_i) = P(E, y_i) + P(B, y_i); logaddexp adds two numbers given in logs
    # without leaving the log scale (Ch. 4, mixture likelihood)
    return np.logaddexp(log_engaged, log_browsing).sum()


def type_probabilities(params, skips, mask):
    """E-step: return P(Engaged type | session) for each session"""
    n, s = session_counts(skips, mask)
    log_engaged, log_browsing = log_terms_by_type(params, n, s)
    # Bayes' rule: P(E | y) = P(E, y) / [P(E, y) + P(B, y)] (Ch. 1, Ch. 4)
    return np.exp(log_engaged - np.logaddexp(log_engaged, log_browsing))


def simulate(params, lengths, rng):
    """Simulate sessions with the given lengths; return skips, mask and types"""
    lengths = np.asarray(lengths)
    n_columns = lengths.max()
    mask = np.arange(n_columns) < lengths[:, None]
    # Discrete inverse transform (Ch. 2): a uniform below p0 means Engaged
    engaged = rng.random(len(lengths)) < params["p0"]
    rate = np.where(engaged, params["theta_e"], params["theta_b"])
    # Same idea for each track: a uniform below the session's rate is a skip
    # Cells after the end of a session stay 0, as in checks.to_matrix
    skips = ((rng.random(mask.shape) < rate[:, None]) & mask).astype(int)
    return skips, mask, engaged


def relabel(params):
    """Swap the two types if needed, so that theta_e < theta_b"""
    # Swapping the names gives the same likelihood (Ch. 4, label switching)
    # The posterior value does not change either, because theta_e and theta_b
    # have the same prior and the prior of p0 is symmetric
    if params["theta_e"] <= params["theta_b"]:
        return params
    return {"theta_e": params["theta_b"], "theta_b": params["theta_e"],
            "p0": 1 - params["p0"]}


def log_prior_probability_scale(params, prior):
    """Return the log Beta prior density of the parameters on the 0-1 scale"""
    return sum(stats.beta.logpdf(params[name], alpha, beta)
               for name, (alpha, beta) in prior.items())


def log_posterior_probability_scale(params, skips, mask, prior):
    """Return log-likelihood + log prior on the 0-1 scale (EM must never lower it)"""
    return log_likelihood(params, skips, mask) + log_prior_probability_scale(params, prior)


def em_step(params, skips, mask, prior):
    """Do one E-step and one M-step, then relabel"""
    n, s = session_counts(skips, mask)
    # E-step: how much each session belongs to the Engaged type (Ch. 4)
    r = type_probabilities(params, skips, mask)
    # M-step: Beta updates where each session counts with weight r or 1 - r
    # p0: "successes" are Engaged sessions, "trials" are all sessions
    # theta: "successes" are skips, "trials" are tracks, in sessions of that type
    new = {
        "theta_e": beta_mode((r * s).sum(), (r * n).sum(), *prior["theta_e"]),
        "theta_b": beta_mode(((1 - r) * s).sum(), ((1 - r) * n).sum(), *prior["theta_b"]),
        "p0": beta_mode(r.sum(), len(r), *prior["p0"]),
    }
    return relabel(new)


def em(params, skips, mask, n_iter=1000, prior=PRIOR, tol=1e-9):
    """Run EM from `params`; return the mode and the log-posterior at each step"""
    history = [log_posterior_probability_scale(params, skips, mask, prior)]
    for _ in range(n_iter):
        params = em_step(params, skips, mask, prior)
        history.append(log_posterior_probability_scale(params, skips, mask, prior))
        # Stop when one more step changes the log-posterior by less than tol
        if abs(history[-1] - history[-2]) < tol:
            break
    return params, np.array(history)


def random_start(rng, prior=PRIOR):
    """Draw a starting point from the prior, with theta_e < theta_b"""
    start = {name: rng.beta(alpha, beta) for name, (alpha, beta) in prior.items()}
    return relabel(start)


def log_posterior(phi, skips, mask, prior=PRIOR):
    """Return the log-posterior (up to a constant) at the log-odds vector phi"""
    params = to_params(phi, NAMES)
    # The constraint theta_e < theta_b fixes the labels (Ch. 4)
    # Outside it the posterior is 0, so the log is -inf (weight 0 in Ch. 3)
    if params["theta_e"] >= params["theta_b"]:
        return -np.inf
    # The prior includes the Jacobian, because phi is the log-odds (Ch. 2)
    return log_likelihood(params, skips, mask) + log_beta_prior_log_odds(phi, prior, NAMES)


def skip_probabilities(params, skips, mask):
    """Return P(skip at track t | tracks before t) for every track of every session"""
    # Counts of tracks and skips strictly before t: cumulative sum minus the track itself
    seen = skips * mask
    s_before = np.cumsum(seen, axis=1) - seen
    n_before = np.cumsum(mask, axis=1) - mask
    # Bayes' rule with the tracks seen so far (Ch. 1): at t = 1 this is p0,
    # then it moves towards the type that explains the past skips best
    log_engaged, log_browsing = log_terms_by_type(params, n_before, s_before)
    p_engaged = np.exp(log_engaged - np.logaddexp(log_engaged, log_browsing))
    # Next track: skipped with theta_e if Engaged, theta_b if Browsing
    return p_engaged * params["theta_e"] + (1 - p_engaged) * params["theta_b"]
