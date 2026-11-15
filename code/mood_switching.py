"""Layer 3 model: two moods (Engaged, Browsing) that can switch inside a session -> each track has its own hidden mood, the mood of a track only depends on the mood of the track before (Markov chain), and given its mood, a track is skipped with rate theta_e or theta_b
"""

import numpy as np

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


def emission_probs(params, skips, mask):
    """Return P(y_t | Engaged) and P(y_t | Browsing) for every track, two arrays of shape (sessions, tracks)"""
    engaged = np.where(skips == 1, params["theta_e"], 1 - params["theta_e"])
    browsing = np.where(skips == 1, params["theta_b"], 1 - params["theta_b"])
    # Cells after the end of a session get probability 1, so they change nothing
    return np.where(mask, engaged, 1.0), np.where(mask, browsing, 1.0)


# Forward algorithm with scaling:
# the forward messages alpha_t(j) = P(y_1..y_t, mood j at t) become tiny after many tracks, so after each
# track we divide them by their sum c_t = P(y_t | y_1..y_{t-1}), which keeps them between 0 and 1
# The scaled message is then P(mood j at t | y_1..y_t), and log p(y_i) = sum over t of log c_t
# This gives exactly the same numbers as working in logs, but much faster (no log or exp inside the loop)
# With only two moods we write the sums over moods by hand, as in Layer 2

def forward(params, skips, mask):
    """Return the scaled forward messages P(Engaged at t | y_1..y_t), P(Browsing at t | y_1..y_t) and c_t"""
    emission_e, emission_b = emission_probs(params, skips, mask)
    q_eb, q_be = params["q_eb"], params["q_be"]
    alpha_e = np.empty(skips.shape)
    alpha_b = np.empty(skips.shape)
    scale = np.empty(skips.shape)
    # One loop over the positions (at most 20), every session at once
    for t in range(skips.shape[1]):
        if t == 0:
            new_e = params["p0"] * emission_e[:, 0]
            new_b = (1 - params["p0"]) * emission_b[:, 0]
        else:
            # Stay Engaged or come from Browsing, then see track t (and the same for Browsing)
            new_e = (alpha_e[:, t - 1] * (1 - q_eb) + alpha_b[:, t - 1] * q_be) * emission_e[:, t]
            new_b = (alpha_e[:, t - 1] * q_eb + alpha_b[:, t - 1] * (1 - q_be)) * emission_b[:, t]
        scale[:, t] = new_e + new_b
        alpha_e[:, t] = new_e / scale[:, t]
        alpha_b[:, t] = new_b / scale[:, t]
    # After the end of a session the emission is 1 and the probabilities to stay or switch sum to 1,
    # so c_t = 1 and log c_t = 0
    return alpha_e, alpha_b, scale


def backward(params, skips, mask, scale):
    """Return the scaled backward messages P(y_{t+1}..y_n | mood at t) / (c_{t+1} * ... * c_n), for each mood"""
    emission_e, emission_b = emission_probs(params, skips, mask)
    q_eb, q_be = params["q_eb"], params["q_be"]
    beta_e = np.ones(skips.shape)   # 1 at the last position
    beta_b = np.ones(skips.shape)
    for t in range(skips.shape[1] - 2, -1, -1):
        # Track t+1 seen in each mood, divided by the same c_{t+1} as in forward
        next_e = emission_e[:, t + 1] * beta_e[:, t + 1] / scale[:, t + 1]
        next_b = emission_b[:, t + 1] * beta_b[:, t + 1] / scale[:, t + 1]
        # From Engaged: stay Engaged or switch to Browsing (and the same from Browsing)
        beta_e[:, t] = (1 - q_eb) * next_e + q_eb * next_b
        beta_b[:, t] = q_be * next_e + (1 - q_be) * next_b
    return beta_e, beta_b


def log_likelihood(params, skips, mask):
    """Return the log-likelihood of all sessions"""
    _, _, scale = forward(params, skips, mask)
    # log p(y_i) = sum over t of log c_t
    return np.log(scale).sum()


def state_probabilities(params, skips, mask):
    """E-step: return P(mood at t | session) and P(moods at t and t+1 | session)

    gamma has shape (sessions, tracks, 2) and xi has shape (sessions, tracks - 1, 2, 2), index 0 = Engaged
    Both are 0 where a track (or the next track) does not exist
    """
    emission_e, emission_b = emission_probs(params, skips, mask)
    q_eb, q_be = params["q_eb"], params["q_be"]
    alpha_e, alpha_b, scale = forward(params, skips, mask)
    beta_e, beta_b = backward(params, skips, mask, scale)

    # Bayes' rule: P(mood at t | y) = alpha_t * beta_t / p(y), and with the scaling p(y) is already divided out
    gamma = np.stack([alpha_e * beta_e, alpha_b * beta_b], axis=-1)
    # P(mood i at t, mood j at t+1 | y) = alpha_t(i) * P(i -> j) * P(y_{t+1} | j) * beta_{t+1}(j) / c_{t+1}
    next_e = emission_e[:, 1:] * beta_e[:, 1:] / scale[:, 1:]
    next_b = emission_b[:, 1:] * beta_b[:, 1:] / scale[:, 1:]
    from_e = np.stack([alpha_e[:, :-1] * (1 - q_eb) * next_e, alpha_e[:, :-1] * q_eb * next_b], axis=-1)
    from_b = np.stack([alpha_b[:, :-1] * q_be * next_e, alpha_b[:, :-1] * (1 - q_be) * next_b], axis=-1)
    xi = np.stack([from_e, from_b], axis=-2)
    gamma = gamma * mask[..., None]
    xi = xi * mask[:, 1:, None, None]   # only pairs where both tracks exist
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


def em(params, skips, mask, prior, n_iter=2000, tol=1e-8):
    """Run EM steps until the log-posterior gains less than tol (at most n_iter steps)
    and return the last point and the log-posterior at each step"""
    # Unlike Layer 2, a fixed number of steps is not enough: when the two moods are hard to tell apart,
    # EM climbs very slowly, so we stop when one more step changes almost nothing
    history = [log_posterior_probability_scale(params, skips, mask, prior)]
    for _ in range(n_iter):
        params = em_step(params, skips, mask, prior)
        history.append(log_posterior_probability_scale(params, skips, mask, prior))
        if history[-1] - history[-2] < tol:
            break
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
    alpha_e, alpha_b, _ = forward(params, skips, mask)
    # P(Engaged at t | y before t): p0 for the first track, then stay Engaged or come from Browsing
    p_engaged = np.empty(skips.shape)
    p_engaged[:, 0] = params["p0"]
    p_engaged[:, 1:] = alpha_e[:, :-1] * (1 - params["q_eb"]) + alpha_b[:, :-1] * params["q_be"]
    # Next track: skipped with theta_e if engaged, theta_b if browsing
    return p_engaged * params["theta_e"] + (1 - p_engaged) * params["theta_b"]
