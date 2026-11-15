"""Layer 3 model: two moods (Engaged, Browsing) that can switch inside a session -> each track has its own hidden mood, the mood of a track only depends on the mood of the track before (Markov chain), and given its mood, a track is skipped with rate theta_e or theta_b
"""

import numpy as np
from scipy.special import logsumexp

from posterior import beta_mode, log_beta_prior_log_odds, to_params
# Same prior density as Layer 2, so we reuse it
from session_types import log_prior_probability_scale

# Order of the parameters in the log-odds vector
# -> theta_e: proba a track is skipped when the listener is engaged
# -> theta_b: proba a track is skipped when the listener is browsing
# -> p0: proba the first track of a session is engaged
# -> q_eb: proba to switch from engaged to browsing between two tracks
# -> q_be: proba to switch from browsing to engaged between two tracks
NAMES = ["theta_e", "theta_b", "p0", "q_eb", "q_be"]

# Beta(2, 2) for every parameter, as in Layer 2
PRIOR = {name: (2, 2) for name in NAMES}

# Moods are stored in this order in every array: index 0 = Engaged, index 1 = Browsing


def log_initial(params):
    """Return log P(mood of the first track) for Engaged and Browsing"""
    return np.log(np.array([params["p0"], 1 - params["p0"]]))


def log_transition_matrix(params):
    """Return the log of the 2x2 matrix A, where A[i, j] = P(mood j at t | mood i at t-1)"""
    q_eb, q_be = params["q_eb"], params["q_be"]
    matrix = np.array([[1 - q_eb, q_eb],
                       [q_be, 1 - q_be]])
    # A switch probability of exactly 0 gives log 0 = -inf, which is allowed (used to compare with Layer 2)
    with np.errstate(divide="ignore"):
        return np.log(matrix)


def emission_log_probs(params, skips, mask):
    """Return log P(y_t | mood) for each track and each mood, shape (sessions, tracks, 2)"""
    theta = np.array([params["theta_e"], params["theta_b"]])
    log_probs = np.where(skips[..., None] == 1, np.log(theta), np.log(1 - theta))
    # Cells after the end of a session get log 1 = 0, so they change nothing
    return np.where(mask[..., None], log_probs, 0.0)


def forward(params, skips, mask):
    """Return the forward messages log P(y_1..y_t, mood at t) and log p(y_i) of each session"""
    log_emission = emission_log_probs(params, skips, mask)
    log_A = log_transition_matrix(params)
    log_alpha = np.zeros(log_emission.shape)
    log_alpha[:, 0] = log_initial(params) + log_emission[:, 0]
    # One loop over the positions (at most 20), every session at once
    for t in range(1, skips.shape[1]):
        # alpha_t(j) = sum over i of alpha_{t-1}(i) * A(i, j), times P(y_t | j), written in logs
        log_alpha[:, t] = (logsumexp(log_alpha[:, t - 1, :, None] + log_A, axis=1)
                           + log_emission[:, t])
    # After the end of a session the emission is 1 and the rows of A sum to 1, so the total stays p(y_i)
    return log_alpha, logsumexp(log_alpha[:, -1], axis=1)


def backward(params, skips, mask):
    """Return the backward messages log P(y_{t+1}..y_n | mood at t) of each session"""
    log_emission = emission_log_probs(params, skips, mask)
    log_A = log_transition_matrix(params)
    log_beta = np.zeros(log_emission.shape)   # log 1 = 0 at the last position
    for t in range(skips.shape[1] - 2, -1, -1):
        # beta_t(i) = sum over j of A(i, j) * P(y_{t+1} | j) * beta_{t+1}(j), written in logs
        log_beta[:, t] = logsumexp(
            log_A + (log_emission[:, t + 1] + log_beta[:, t + 1])[:, None, :], axis=2)
    return log_beta


def log_likelihood(params, skips, mask):
    """Return the log-likelihood of all sessions"""
    _, log_p = forward(params, skips, mask)
    return log_p.sum()


def state_probabilities(params, skips, mask):
    """E-step: return P(mood at t | session) and P(moods at t and t+1 | session)

    gamma has shape (sessions, tracks, 2) and xi has shape (sessions, tracks - 1, 2, 2)
    Both are 0 where a track (or the next track) does not exist
    """
    log_emission = emission_log_probs(params, skips, mask)
    log_A = log_transition_matrix(params)
    log_alpha, log_p = forward(params, skips, mask)
    log_beta = backward(params, skips, mask)

    # Bayes' rule: P(mood at t | y) = alpha_t * beta_t / p(y)
    gamma = np.exp(log_alpha + log_beta - log_p[:, None, None])
    # P(mood i at t, mood j at t+1 | y) = alpha_t(i) A(i, j) P(y_{t+1} | j) beta_{t+1}(j) / p(y)
    log_xi = (log_alpha[:, :-1, :, None] + log_A
              + (log_emission[:, 1:] + log_beta[:, 1:])[:, :, None, :]
              - log_p[:, None, None, None])
    gamma = gamma * mask[..., None]
    xi = np.exp(log_xi) * mask[:, 1:, None, None]   # only pairs where both tracks exist
    return gamma, xi


def simulate(params, lengths, rng):
    """Simulate sessions with the given lengths, return skips, mask and moods (True = Engaged)"""
    lengths = np.asarray(lengths)
    n_columns = lengths.max()
    mask = np.arange(n_columns) < lengths[:, None]
    engaged = np.zeros(mask.shape, dtype=bool)
    # Discrete inverse transform: a uniform below p0 means the first track is Engaged
    engaged[:, 0] = rng.random(len(lengths)) < params["p0"]
    for t in range(1, n_columns):
        # An Engaged track stays Engaged with proba 1 - q_eb, a Browsing track becomes Engaged with proba q_be
        p_engaged = np.where(engaged[:, t - 1], 1 - params["q_eb"], params["q_be"])
        engaged[:, t] = rng.random(len(lengths)) < p_engaged
    rate = np.where(engaged, params["theta_e"], params["theta_b"])
    # Same idea for each track: a uniform below the track's rate is a skip
    # Cells after the end of a session stay 0 (similar to checks.to_matrix)
    skips = ((rng.random(mask.shape) < rate) & mask).astype(int)
    return skips, mask, engaged & mask


def relabel(params):
    """Check we have theta_e < theta_b, or else we rename"""
    if params["theta_e"] <= params["theta_b"]:
        return params
    # Renaming the two moods also swaps the two switch probabilities
    return {"theta_e": params["theta_b"], "theta_b": params["theta_e"],
            "p0": 1 - params["p0"], "q_eb": params["q_be"], "q_be": params["q_eb"]}


def log_posterior_probability_scale(params, skips, mask, prior):
    """Return log-likelihood + log prior on the 0-1 scale"""
    return log_likelihood(params, skips, mask) + log_prior_probability_scale(params, prior)


def em_step(params, skips, mask, prior):
    """Do one E-step and one M-step (Baum-Welch), then relabel if needed"""
    # E-step: how much each track belongs to each mood, and each pair of tracks to each switch
    gamma, xi = state_probabilities(params, skips, mask)
    engaged, browsing = gamma[..., 0], gamma[..., 1]
    seen = skips * mask
    # M-step: Beta mode (Ch. 4, Example 4.1), with expected counts instead of counts
    # theta: "successes" are skips, "trials" are tracks, in that mood
    # p0: "successes" are engaged first tracks, "trials" are all sessions
    # q: "successes" are switches, "trials" are tracks of that mood that have a next track
    new = {
        "theta_e": beta_mode((engaged * seen).sum(), engaged.sum(), *prior["theta_e"]),
        "theta_b": beta_mode((browsing * seen).sum(), browsing.sum(), *prior["theta_b"]),
        "p0": beta_mode(engaged[:, 0].sum(), len(skips), *prior["p0"]),
        "q_eb": beta_mode(xi[:, :, 0, 1].sum(), xi[:, :, 0, :].sum(), *prior["q_eb"]),
        "q_be": beta_mode(xi[:, :, 1, 0].sum(), xi[:, :, 1, :].sum(), *prior["q_be"]),
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
    # Swapping the labels changes nothing so the constraint theta_e < theta_b fixes the labels
    # If theta_e >= theta_b, the posterior is 0 there, so its log is -inf
    if params["theta_e"] >= params["theta_b"]:
        return -np.inf
    return log_likelihood(params, skips, mask) + log_beta_prior_log_odds(phi, prior, NAMES)


def skip_probabilities(params, skips, mask):
    """Return P(skip at track t | tracks before t) for every track of every session"""
    log_alpha, _ = forward(params, skips, mask)
    log_A = log_transition_matrix(params)
    # log P(mood at t, y before t): the first track uses p0, the others move alpha_{t-1} one step with A
    log_predicted = np.empty(log_alpha.shape)
    log_predicted[:, 0] = log_initial(params)
    log_predicted[:, 1:] = logsumexp(log_alpha[:, :-1, :, None] + log_A, axis=2)
    # Bayes' rule: P(Engaged at t | y before t)
    p_engaged = np.exp(log_predicted[..., 0] - logsumexp(log_predicted, axis=-1))
    # Next track: skipped with theta_e if engaged, theta_b if browsing
    return p_engaged * params["theta_e"] + (1 - p_engaged) * params["theta_b"]
