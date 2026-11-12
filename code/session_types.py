"""Layer 2 model: two session types (Engaged, Browsing), no switching -> each session has one hidden type for all its tracks, and iven the type, tracks are skipped independently with rate theta_e or theta_b
"""

import numpy as np
from scipy import stats

from posterior import beta_mode, log_beta_prior_log_odds, to_params

# Order of the parameters in the log-odds vector 
# -> theta_e: proba a track is skipped when the session is engaged
# -> theta_b: proba a track is skipped when the session is browsin
# -> p0: the proba a session is engaged 
NAMES = ["theta_e", "theta_b", "p0"]

# Beta(2, 2): initially we kept the same Beta(1,1) but we had an issue of a mode of 0 becoming -inf in log-odds so we switched to Beta(2,2)
PRIOR = {"theta_e": (2, 2), "theta_b": (2, 2), "p0": (2, 2)}


def session_counts(skips, mask):
    """Return the number of tracks n and of skips s of each session"""
    n = mask.sum(axis=1)
    s = (skips * mask).sum(axis=1)
    return n, s


def log_terms_by_type(params, n, s):
    """Return log P(type, skips) for Engaged and for Browsing, one value per session"""
    # P(Engaged) * theta_e^s * (1 - theta_e)^(n - s), written in logs so that
    #log to prevent underflow
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
    # p(y_i) = P(E, y_i) + P(B, y_i)
    return np.logaddexp(log_engaged, log_browsing).sum()


def type_probabilities(params, skips, mask):
    """E-step: return P(Engaged type | session) for each session"""
    n, s = session_counts(skips, mask)
    log_engaged, log_browsing = log_terms_by_type(params, n, s)
    # basic bayes' rule: P(E | y) = P(E, y) / [P(E, y) + P(B, y)] 
    return np.exp(log_engaged - np.logaddexp(log_engaged, log_browsing))


def simulate(params, lengths, rng):
    """Simulate sessions with the given lengths, return skips, mask and types"""
    lengths = np.asarray(lengths)
    n_columns = lengths.max()
    mask = np.arange(n_columns) < lengths[:, None]
    # Discrete inverse transform: a uniform below p0 means Engaged
    engaged = rng.random(len(lengths)) < params["p0"]
    rate = np.where(engaged, params["theta_e"], params["theta_b"])
    # Same idea for each track: a uniform below the session's rate is a skip
    # Cells after the end of a session stay 0 (similar to checks.to_matrix)
    skips = ((rng.random(mask.shape) < rate[:, None]) & mask).astype(int)
    return skips, mask, engaged


def relabel(params):
    """Check we have theta_e < theta_b, or else we rename"""
    if params["theta_e"] <= params["theta_b"]:
        return params
    return {"theta_e": params["theta_b"], "theta_b": params["theta_e"],
            "p0": 1 - params["p0"]}


def log_prior_probability_scale(params, prior):
    """Return the log Beta prior density of the parameters on the 0-1 scale"""
    return sum(stats.beta.logpdf(params[name], alpha, beta)
               for name, (alpha, beta) in prior.items())


def log_posterior_probability_scale(params, skips, mask, prior):
    """Return log-likelihood + log prior on the 0-1 scale"""
    return log_likelihood(params, skips, mask) + log_prior_probability_scale(params, prior)


def em_step(params, skips, mask, prior):
    """Do one E-step and one M-step, then relabel if needed """
    n, s = session_counts(skips, mask)
    # E-step: how much each session belongs to the engaged behaviour 
    r = type_probabilities(params, skips, mask)
    # M-step: Beta updates where each session counts with weight r or 1 - r
    # p0: "successes" are engaged seshs, "trials" are all sessions
    # theta: "successes" are skips, "trials" are tracks, in sessions of that type
    new = {
        "theta_e": beta_mode((r * s).sum(), (r * n).sum(), *prior["theta_e"]),
        "theta_b": beta_mode(((1 - r) * s).sum(), ((1 - r) * n).sum(), *prior["theta_b"]),
        "p0": beta_mode(r.sum(), len(r), *prior["p0"]),
    }
    return relabel(new)


def em(params, skips, mask, prior, n_iter=200):
    """Run n_iter EM steps and return the last point and the log-posterior at each step"""
    history = [log_posterior_probability_scale(params, skips, mask, prior)]
    for _ in range(n_iter):
        params = em_step(params, skips, mask, prior)
        history.append(log_posterior_probability_scale(params, skips, mask, prior))
    return params, np.array(history)


def random_start(rng, prior=PRIOR):
    """Draw a random starting point from the prior with theta_e < theta_b"""
    start = {name: rng.beta(alpha, beta) for name, (alpha, beta) in prior.items()}
    return relabel(start)


def log_posterior(phi, skips, mask, prior=PRIOR):
    """Return the log-posterior at the log-odds vector phi"""
    params = to_params(phi, NAMES)
    # The constraint theta_e < theta_b fixes the labels
    # so if values are outside we set them at -inf as they have proba 0
    if params["theta_e"] >= params["theta_b"]:
        return -np.inf
    return log_likelihood(params, skips, mask) + log_beta_prior_log_odds(phi, prior, NAMES)


def skip_probabilities(params, skips, mask):
    """Return P(skip at track t | tracks before t) for every track of every session"""
    seen = skips * mask
    s_before = np.cumsum(seen, axis=1) - seen
    n_before = np.cumsum(mask, axis=1) - mask
    log_engaged, log_browsing = log_terms_by_type(params, n_before, s_before)
    p_engaged = np.exp(log_engaged - np.logaddexp(log_engaged, log_browsing))
    # Next track: skipped with theta_e if Engaged, theta_b if Browsing
    return p_engaged * params["theta_e"] + (1 - p_engaged) * params["theta_b"]
