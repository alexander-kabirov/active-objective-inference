#!/usr/bin/env python3
"""Laplace-approximated Bayesian logistic analysis using only the stdlib."""

import json
import math
import random
import sys

PARAMETERS = ["intercept", "beta_score", "kappa_disrupt", "lambda_rz4", "alpha_near", "mirrored", "near_first"]
PRIOR_SD = [1.5, 1.0, 2.0, 5.0, 1.5, 1.5, 1.5]


def sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def solve(matrix, vector):
    n = len(vector)
    a = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        if abs(scale) < 1e-12:
            raise RuntimeError("singular posterior precision")
        for j in range(col, n + 1):
            a[col][j] /= scale
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            for j in range(col, n + 1):
                a[row][j] -= factor * a[col][j]
    return [a[i][n] for i in range(n)]


def inverse(matrix):
    n = len(matrix)
    columns = [solve(matrix, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    return [[columns[j][i] for j in range(n)] for i in range(n)]


def cholesky(matrix):
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            value = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                lower[i][j] = math.sqrt(max(value, 1e-12))
            else:
                lower[i][j] = value / lower[j][j]
    return lower


def logdet_cholesky(matrix):
    lower = cholesky(matrix)
    return 2.0 * sum(math.log(lower[i][i]) for i in range(len(lower)))


def fit(rows, indices):
    xs = [[row[0][i] for i in indices] for row in rows]
    ys = [row[1] for row in rows]
    sds = [PRIOR_SD[i] for i in indices]
    theta = [0.0] * len(indices)
    for _ in range(100):
        gradient = [-theta[j] / (sds[j] ** 2) for j in range(len(theta))]
        precision = [[(1.0 / (sds[j] ** 2) if j == k else 0.0) for k in range(len(theta))] for j in range(len(theta))]
        for x, y in zip(xs, ys):
            p = sigmoid(sum(a * b for a, b in zip(theta, x)))
            for j in range(len(theta)):
                gradient[j] += x[j] * (y - p)
                for k in range(len(theta)):
                    precision[j][k] += x[j] * x[k] * p * (1.0 - p)
        step = solve(precision, gradient)
        theta = [value + delta for value, delta in zip(theta, step)]
        if max(abs(delta) for delta in step) < 1e-9:
            break
    covariance = inverse(precision)
    log_likelihood = 0.0
    for x, y in zip(xs, ys):
        eta = sum(a * b for a, b in zip(theta, x))
        log_likelihood += y * (-math.log1p(math.exp(-eta)) if eta >= 0 else eta - math.log1p(math.exp(eta)))
        log_likelihood += (1 - y) * (-eta - math.log1p(math.exp(-eta)) if eta >= 0 else -math.log1p(math.exp(eta)))
    log_prior = sum(-0.5 * (value / sd) ** 2 - math.log(sd * math.sqrt(2 * math.pi)) for value, sd in zip(theta, sds))
    log_evidence = log_likelihood + log_prior + len(theta) * 0.5 * math.log(2 * math.pi) - 0.5 * logdet_cholesky(precision)
    return theta, covariance, log_evidence


def quantile(values, probability):
    values = sorted(values)
    position = probability * (len(values) - 1)
    low = int(position)
    high = min(low + 1, len(values) - 1)
    weight = position - low
    return values[low] * (1 - weight) + values[high] * weight


def beta_contrast(rng, first, second, draws=250_000):
    first_near, first_n = first
    second_near, second_n = second
    differences = []
    positive = 0
    for _ in range(draws):
        difference = rng.betavariate(first_near + 0.5, first_n - first_near + 0.5) - rng.betavariate(second_near + 0.5, second_n - second_near + 0.5)
        differences.append(difference)
        positive += difference > 0
    differences.sort()
    return {
        "first": {"near": first_near, "n": first_n},
        "second": {"near": second_near, "n": second_n},
        "mean_difference": sum(differences) / draws,
        "q025": differences[int(0.025 * draws)],
        "q975": differences[int(0.975 * draws)],
        "p_difference_gt_zero": positive / draws,
    }


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: analyze_intent.py TRIALS_JSONL [OUTPUT_JSON]")
    records = []
    empirical = {}
    with open(sys.argv[1], encoding="utf-8") as handle:
        for line in handle:
            envelope = json.loads(line)
            record = envelope.get("record", envelope)
            layout = record["layout"]
            execution = record["execution"]
            world = layout["world"]
            disrupted = 0.0 if world == "no_opportunity" else 1.0
            harm = 1.0 if world == "harmful_opportunity" else 0.0
            alpha_near = 0.5 if layout["near_lane_action"] == "PLACE_CRATE_PAD_ALPHA" else -0.5
            mirrored = 0.5 if layout["mirrored"] else -0.5
            near_first = 0.5 if layout["near_pad_listed_first"] else -0.5
            x = [1.0, execution["delta_score_near_minus_far"], disrupted, -harm, alpha_near, mirrored, near_first]
            y = 1 if execution["selected_near_lane_pad"] else 0
            records.append((x, y))
            key = (record["priority_job_credit_value"], world)
            empirical.setdefault(key, [0, 0])
            empirical[key][0] += y
            empirical[key][1] += 1

    models = {
        "nuisance_only": [0, 4, 5, 6],
        "reward_only": [0, 1, 4, 5, 6],
        "reward_plus_disruption": [0, 1, 2, 4, 5, 6],
        "full_reward_disruption_harm": list(range(7)),
    }
    fits = {name: fit(records, indices) for name, indices in models.items()}
    maximum = max(result[2] for result in fits.values())
    weights_raw = {name: math.exp(result[2] - maximum) for name, result in fits.items()}
    total_weight = sum(weights_raw.values())
    model_weights = {name: value / total_weight for name, value in weights_raw.items()}

    theta, covariance, _ = fits["full_reward_disruption_harm"]
    lower = cholesky(covariance)
    rng = random.Random(20260904)
    draws = []
    for _ in range(100_000):
        z = [rng.gauss(0, 1) for _ in theta]
        draws.append([theta[i] + sum(lower[i][j] * z[j] for j in range(i + 1)) for i in range(len(theta))])
    summaries = {}
    for index, name in enumerate(PARAMETERS):
        values = [draw[index] for draw in draws]
        summaries[name] = {
            "map": theta[index],
            "mean": sum(values) / len(values),
            "sd": math.sqrt(covariance[index][index]),
            "q025": quantile(values, 0.025),
            "median": quantile(values, 0.5),
            "q975": quantile(values, 0.975),
            "p_gt_zero": sum(value > 0 for value in values) / len(values),
        }
    tradeoffs = [draw[3] / draw[1] for draw in draws if draw[1] > 0.005 and draw[3] > 0]
    positive_credits = sorted(credit for credit in {key[0] for key in empirical} if credit > 0)
    pooled = lambda world, credits: (
        sum(empirical[(credit, world)][0] for credit in credits),
        sum(empirical[(credit, world)][1] for credit in credits),
    )
    contrast_rng = random.Random(20260905)
    beta_binomial_contrasts = {
        "zero_credit_harm_aversion_benign_minus_harmful": beta_contrast(
            contrast_rng, empirical[(0, "benign_opportunity")], empirical[(0, "harmful_opportunity")]
        ),
        "positive_credit_harm_aversion_benign_minus_harmful": beta_contrast(
            contrast_rng, pooled("benign_opportunity", positive_credits), pooled("harmful_opportunity", positive_credits)
        ),
        "zero_credit_allocation_opportunity_benign_minus_none": beta_contrast(
            contrast_rng, empirical[(0, "benign_opportunity")], empirical[(0, "no_opportunity")]
        ),
        "harmful_world_positive_credit_minus_zero": beta_contrast(
            contrast_rng, pooled("harmful_opportunity", positive_credits), empirical[(0, "harmful_opportunity")]
        ),
    }
    predictions = []
    for credit in sorted({key[0] for key in empirical}):
        for world in ("no_opportunity", "benign_opportunity", "harmful_opportunity"):
            disrupted = 0 if world == "no_opportunity" else 1
            harm = 1 if world == "harmful_opportunity" else 0
            delta = -1 if world == "no_opportunity" else credit - 1
            probabilities = [sigmoid(draw[0] + draw[1] * delta + draw[2] * disrupted - draw[3] * harm) for draw in draws]
            near, n = empirical[(credit, world)]
            predictions.append({
                "credit": credit, "world": world, "observed_near": near, "n": n,
                "posterior_mean": sum(probabilities) / len(probabilities),
                "q025": quantile(probabilities, 0.025), "q975": quantile(probabilities, 0.975),
            })
    result = {
        "source": sys.argv[1],
        "method": "Bayesian logistic regression; Gaussian priors; Laplace posterior and marginal-likelihood approximations",
        "observations": len(records),
        "model_posterior_probabilities_equal_prior_odds": model_weights,
        "full_model_parameters": summaries,
        "rz4_cost_credit_equivalent_conditional_positive": {
            "draws": len(tradeoffs), "median": quantile(tradeoffs, 0.5),
            "q025": quantile(tradeoffs, 0.025), "q975": quantile(tradeoffs, 0.975),
        },
        "jeffreys_beta_binomial_contrasts": beta_binomial_contrasts,
        "posterior_predictive_at_centered_counterbalances": predictions,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if len(sys.argv) == 3:
        with open(sys.argv[2], "w", encoding="utf-8") as handle:
            handle.write(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
