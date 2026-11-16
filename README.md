# Engaged or Browsing? Listener moods in music sessions

Do listeners have a hidden mood that drives their skips? We model listening sessions from the Yandex **Yambda** dataset with three Bayesian models of growing complexity. Each layer is kept only if it explains the real data better than the one before.

Team: Aliénor Sabourdin, Anastasia Bouev-Dombre, Elise Deyris. Course: Bayesian Networks and Markov Models, CentraleSupélec & ESSEC.

- **Engaged**: few skips.
- **Browsing**: many skips.

## The three layers

| Layer | Model | Inference | Test score (additional analysis)* |
|---|---|---|---|
| 1 | One skip rate for every track (Beta-Bernoulli) | Exact (conjugate) | −0.678 |
| 2 | One hidden type per session, no switching (mixture of two Bernoullis) | EM + importance sampling | −0.479 |
| 3 | One hidden mood per track, can switch between tracks (hidden Markov model) | EM (Baum-Welch) + importance sampling | −0.446 |

\* Average log-probability given to the tracks of new listeners. Higher is better; a fair coin scores −0.693.

**Main findings**
- Sessions differ a lot: Layer 2 beats Layer 1 by far.
- Moods do switch within sessions, but rarely: an Engaged stretch lasts about 27 tracks, a Browsing stretch about 22.
- Layer 3 reproduces runs of skips best, but no layer passes all predictive checks. The sessions with no skip or only skips point to differences between listeners, which none of the layers models.

## Method

Each notebook follows the same steps:

1. Validate the model on simulated data with known parameters.
2. Find the posterior mode with EM, from several random starts.
3. Get the full posterior by importance sampling (Student-t proposal at the mode, on the log-odds scale).
4. Run posterior predictive checks with six statistics shared by all layers.
5. Check sensitivity to the prior.

**Additional analysis**: score predictions on held-out users (20% of users, same split for all layers). This is not a check of the model.

## Structure

```
run_all.sh                      # Runs the whole project (see below)
requirements.txt                # Packages, exact versions
code/
├── prepare_yambda.py           # Download Yambda, build sessions and skip labels → data/sessions.csv
├── checks.py                   # Shared: loading, user split, six check statistics, log score
├── posterior.py                # Shared: EM driver, Hessian, importance sampling, predictive checks
├── session_types.py            # Layer 2 model
├── mood_switching.py           # Layer 3 model (forward-backward with scaling)
├── layer1_baseline.ipynb
├── layer2_session_types.ipynb
├── layer3_mood_switching.ipynb
├── data/                       # sessions.csv, split.csv
├── results/                    # layer1.json, layer2.json, layer3.json
└── figures/                    # PDF figures of each layer
```

**Data**: Yambda-5B, Ploshkin et al. (2025), arXiv 2505.22238, Apache 2.0. A skip is a play of 50% of the track or less. A session ends after 60 s of silence or a change of context (organic / recommendation), and has 10 to 20 tracks. We keep one session per user and context.

## Reproduce the results

Python 3.13 or later. From the project folder, install the packages (exact versions in `requirements.txt`), then run everything with one command:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
```

`run_all.sh` runs the three notebooks in order, each from the top in a fresh kernel, and saves them with their outputs. Each layer reads the results of the layers before it, so the script stops at the first error. It needs no manual intervention.

| Notebook | Run time |
|---|---|
| `layer1_baseline.ipynb` | 1.5 min |
| `layer2_session_types.ipynb` | 3.5 min |
| `layer3_mood_switching.ipynb` | 19 min |
| **Total** | **about 23 min** |

**Data.** `code/data/sessions.csv` is included in the repo, so no download is needed. To rebuild it from Yambda (a large download of 50M listens), run `bash run_all.sh --download`.

**Seeds.** All random numbers use the seed 2026. Running `run_all.sh` reproduces exactly the numbers in `code/results/` and in the notebooks.
