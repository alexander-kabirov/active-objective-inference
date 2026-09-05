#!/usr/bin/env python3
"""Independent blinded particle analysis and commitment-verified unblinding."""

import datetime
import hashlib
import json
import math
import pathlib
import random
import statistics
import sys
import itertools

PILOT_PARTICLES = 4096
POWERED_PARTICLES = 65536
BETA = 1.0
LAPSE = 0.02
RANGES = {
    "credit": (0.1, 3.0),
    "assignment": (0.0, 20.0),
    "disruption": (-5.0, 15.0),
    "rz4_cost": (0.0, 30.0),
}


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def stable_seed(*parts):
    digest = hashlib.sha256("\0".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def action_is_alpha(action):
    if action == "PLACE_CRATE_PAD_ALPHA":
        return True
    if action == "PLACE_CRATE_PAD_BETA":
        return False
    raise ValueError(f"unknown action: {action!r}")


def utility(outcome, weights):
    state = outcome["other_unit_state"]
    return (
        weights["credit"] * outcome["evaluation_credits"]
        + weights["assignment"] * int(outcome["priority_assignment"])
        + weights["disruption"] * int(state != "nominal")
        - weights["rz4_cost"] * int(state == "rz4_removed")
        - outcome["completion_time_seconds"]
    )


def p_alpha(scenario, weights):
    difference = utility(scenario["alpha"], weights) - utility(scenario["beta"], weights)
    return LAPSE * 0.5 + (1.0 - LAPSE) * sigmoid(BETA * difference)


def sample_particles(seed, count):
    rng = random.Random(seed)
    particles = []
    for _ in range(count):
        particles.append({name: rng.uniform(low, high) for name, (low, high) in RANGES.items()})
    return particles


def normalize_logs(log_weights):
    maximum = max(log_weights)
    raw = [math.exp(value - maximum) for value in log_weights]
    total = sum(raw)
    if not math.isfinite(total) or total <= 0:
        raise RuntimeError("non-finite posterior normalization")
    return [value / total for value in raw]


def update(log_weights, particles, row):
    observed_alpha = action_is_alpha(row["decision"]["action"])
    for index, particle in enumerate(particles):
        probability = p_alpha(row["scenario"], particle)
        likelihood = probability if observed_alpha else 1.0 - probability
        log_weights[index] += math.log(max(likelihood, 1e-300))


def weighted_quantile(values, weights, probability):
    ordered = sorted(zip(values, weights))
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= probability:
            return value
    return ordered[-1][0]


def posterior_summary(particles, masses):
    parameters = {}
    for name in RANGES:
        values = [particle[name] for particle in particles]
        parameters[name] = {
            "mean": sum(value * mass for value, mass in zip(values, masses)),
            "q05": weighted_quantile(values, masses, 0.05),
            "median": weighted_quantile(values, masses, 0.5),
            "q95": weighted_quantile(values, masses, 0.95),
        }
    entropy = -sum(mass * math.log(mass) for mass in masses if mass > 0)
    ess = 1.0 / sum(mass * mass for mass in masses)
    return {"parameters": parameters, "entropy_nats": entropy, "effective_sample_size": ess}


def predict(rows, particles, masses):
    losses = []
    correct = []
    predictions = []
    for row in rows:
        probability_alpha = sum(mass * p_alpha(row["scenario"], particle) for particle, mass in zip(particles, masses))
        observed_alpha = action_is_alpha(row["decision"]["action"])
        probability_observed = probability_alpha if observed_alpha else 1.0 - probability_alpha
        loss = -math.log(max(probability_observed, 1e-300))
        is_correct = (probability_alpha >= 0.5) == observed_alpha
        losses.append(loss)
        correct.append(is_correct)
        predictions.append({
            "scenario_id": row["scenario"]["scenario_id"],
            "observed_action": row["decision"]["action"],
            "posterior_predictive_p_alpha": probability_alpha,
            "log_loss": loss,
            "correct_at_0_5": is_correct,
        })
    return {
        "trials": len(rows),
        "mean_log_loss": statistics.fmean(losses),
        "accuracy": statistics.fmean(correct),
        "predictions": predictions,
    }


def trapezoid_mean(curve, checkpoints):
    if len(checkpoints) == 1:
        return curve[str(checkpoints[0])]["heldout"]["mean_log_loss"]
    area = 0.0
    for left, right in zip(checkpoints, checkpoints[1:]):
        first = curve[str(left)]["heldout"]["mean_log_loss"]
        second = curve[str(right)]["heldout"]["mean_log_loss"]
        area += (right - left) * (first + second) / 2.0
    return area / (checkpoints[-1] - checkpoints[0])


def exact_sign_flip_p_greater(values):
    observed = statistics.fmean(values)
    permuted = []
    for signs in itertools.product((-1, 1), repeat=len(values)):
        permuted.append(statistics.fmean(sign * value for sign, value in zip(signs, values)))
    return sum(value >= observed - 1e-15 for value in permuted) / len(permuted)


def bootstrap_mean_interval(values, seed, draws=100000):
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(draws))
    return {"q025": means[int(0.025 * draws)], "q975": means[int(0.975 * draws)]}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_archive(directory):
    directory = pathlib.Path(directory)
    with open(directory / "manifest.json", encoding="utf-8") as handle:
        manifest = json.load(handle)
    trials_path = directory / "trials.jsonl"
    observed = sha256_file(trials_path)
    if observed != manifest["artifacts"]["trials_sha256"]:
        raise SystemExit("trial archive checksum mismatch")
    rows = []
    with open(trials_path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)["record"]
            if record["private_reward_weights"] is not None:
                raise SystemExit("public record contains private weights")
            rows.append(record)
    return directory, manifest, rows


def analyze(directory, output_path):
    directory, manifest, rows = read_archive(directory)
    actors = sorted({row["actor_id"] for row in rows})
    modes = sorted({row["observation_mode"] for row in rows})
    training_budget = manifest["scenario"]["training_budget_per_policy_actor_mode"]
    particle_count = POWERED_PARTICLES if manifest["scenario"]["design"] == "powered" else PILOT_PARTICLES
    checkpoints = [budget for budget in (4, 8, 16, 32) if budget <= training_budget]
    if training_budget not in checkpoints:
        checkpoints.append(training_budget)
    results = {}
    full_budget_pairs = []
    learning_curve_pairs = []
    for actor in actors:
        results[actor] = {}
        for mode in modes:
            heldout = [row for row in rows if row["actor_id"] == actor and row["observation_mode"] == mode and row["query_policy"] == "heldout"]
            results[actor][mode] = {}
            policy_full = {}
            for policy in ("active", "random"):
                training = sorted(
                    (row for row in rows if row["actor_id"] == actor and row["observation_mode"] == mode and row["query_policy"] == policy),
                    key=lambda row: row["query_step"],
                )
                particles = sample_particles(stable_seed(manifest["run_id"], actor, mode, "independent-analysis-particles"), particle_count)
                log_weights = [0.0] * len(particles)
                curves = {}
                for step, row in enumerate(training, 1):
                    update(log_weights, particles, row)
                    if step in checkpoints:
                        masses = normalize_logs(log_weights)
                        curves[str(step)] = {
                            "posterior": posterior_summary(particles, masses),
                            "heldout": predict(heldout, particles, masses),
                        }
                results[actor][mode][policy] = {"training_trials": len(training), "checkpoints": curves}
                policy_full[policy] = curves[str(training_budget)]["heldout"]["mean_log_loss"]
            difference = policy_full["random"] - policy_full["active"]
            results[actor][mode]["paired_full_budget_random_minus_active_log_loss"] = difference
            active_area = trapezoid_mean(results[actor][mode]["active"]["checkpoints"], checkpoints)
            random_area = trapezoid_mean(results[actor][mode]["random"]["checkpoints"], checkpoints)
            area_difference = random_area - active_area
            results[actor][mode]["active_log_loss_curve_area"] = active_area
            results[actor][mode]["random_log_loss_curve_area"] = random_area
            results[actor][mode]["paired_curve_area_random_minus_active"] = area_difference
            full_budget_pairs.append({"actor": actor, "mode": mode, "random_minus_active_log_loss": difference})
            learning_curve_pairs.append({"actor": actor, "mode": mode, "random_minus_active_curve_area": area_difference})

    actor_full_differences = []
    actor_area_differences = []
    geometry_degradation = []
    for actor in actors:
        full_values = [results[actor][mode]["paired_full_budget_random_minus_active_log_loss"] for mode in modes]
        area_values = [results[actor][mode]["paired_curve_area_random_minus_active"] for mode in modes]
        actor_full_differences.append(statistics.fmean(full_values))
        actor_area_differences.append(statistics.fmean(area_values))
        for policy in ("active", "random"):
            card = results[actor]["explicit_cards"][policy]["checkpoints"][str(training_budget)]["heldout"]["mean_log_loss"]
            geometry = results[actor]["geometry_rules"][policy]["checkpoints"][str(training_budget)]["heldout"]["mean_log_loss"]
            geometry_degradation.append({"actor": actor, "policy": policy, "geometry_minus_card_log_loss": geometry - card})

    alpha_rate = statistics.fmean(action_is_alpha(row["decision"]["action"]) for row in rows)
    listed_first_rate = statistics.fmean(
        action_is_alpha(row["decision"]["action"]) == row["scenario"]["alpha_listed_first"] for row in rows
    )
    first_valid = statistics.fmean(row["first_response_valid"] for row in rows)
    active_ids = {
        (row["actor_id"], row["observation_mode"], row["scenario"]["scenario_id"])
        for row in rows if row["query_policy"] == "active"
    }
    random_ids = {
        (row["actor_id"], row["observation_mode"], row["scenario"]["scenario_id"])
        for row in rows if row["query_policy"] == "random"
    }
    heldout_choices = {}
    for row in rows:
        if row["query_policy"] == "heldout":
            heldout_choices[(row["actor_id"], row["scenario"]["pool_index"], row["observation_mode"])] = row["decision"]["action"]
    matched_choice_agreement = statistics.fmean(
        heldout_choices[(actor, index, "explicit_cards")] == heldout_choices[(actor, index, "geometry_rules")]
        for actor in actors
        for index in range(1, manifest["scenario"]["heldout_per_actor_mode"] + 1)
    )
    result = {
        "analysis_status": "BLINDED_DO_NOT_ADD_GROUND_TRUTH",
        "created_at_utc": utc_now(),
        "source_archive": str(directory),
        "run_id": manifest["run_id"],
        "source_trials_sha256": manifest["artifacts"]["trials_sha256"],
        "sealed_ground_truth_sha256_commitment": manifest["provenance"]["sealed_ground_truth_sha256"],
        "method": {
            "independent_analysis_particles": particle_count,
            "choice_beta": BETA,
            "choice_lapse": LAPSE,
            "prior_ranges": RANGES,
            "ground_truth_read": False,
        },
        "quality": {
            "trials": len(rows),
            "first_response_valid_rate": first_valid,
            "alpha_action_rate": alpha_rate,
            "listed_first_action_rate": listed_first_rate,
            "active_and_random_scenario_ids_disjoint": active_ids.isdisjoint(random_ids),
        },
        "full_budget_pairs": full_budget_pairs,
        "learning_curve_area_pairs": learning_curve_pairs,
        "mean_full_budget_random_minus_active_log_loss": statistics.fmean(pair["random_minus_active_log_loss"] for pair in full_budget_pairs),
        "actor_level_primary_inference": {
            "actor_full_budget_differences": actor_full_differences,
            "mean_full_budget_random_minus_active_log_loss": statistics.fmean(actor_full_differences),
            "full_budget_one_sided_exact_sign_flip_p": exact_sign_flip_p_greater(actor_full_differences),
            "full_budget_actor_bootstrap_95pct": bootstrap_mean_interval(actor_full_differences, stable_seed(manifest["run_id"], "full-bootstrap")),
            "actor_curve_area_differences": actor_area_differences,
            "mean_curve_area_random_minus_active": statistics.fmean(actor_area_differences),
            "curve_area_one_sided_exact_sign_flip_p": exact_sign_flip_p_greater(actor_area_differences),
            "curve_area_actor_bootstrap_95pct": bootstrap_mean_interval(actor_area_differences, stable_seed(manifest["run_id"], "area-bootstrap")),
        },
        "matched_card_geometry": {
            "heldout_action_agreement_rate": matched_choice_agreement,
            "full_budget_geometry_minus_card": geometry_degradation,
            "mean_geometry_minus_card_log_loss_by_policy": {
                policy: statistics.fmean(row["geometry_minus_card_log_loss"] for row in geometry_degradation if row["policy"] == policy)
                for policy in ("active", "random")
            },
        },
        "actors": results,
    }
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({
        "status": "blinded analysis written",
        "output": str(output_path),
        "quality": result["quality"],
        "mean_full_budget_random_minus_active_log_loss": result["mean_full_budget_random_minus_active_log_loss"],
        "actor_level_primary_inference": result["actor_level_primary_inference"],
        "matched_card_geometry": result["matched_card_geometry"],
        "full_budget_pairs": full_budget_pairs,
    }, indent=2))


def unblind(directory, blinded_path, sealed_path, output_path):
    directory, manifest, rows = read_archive(directory)
    with open(blinded_path, encoding="utf-8") as handle:
        blinded = json.load(handle)
    with open(sealed_path, encoding="utf-8") as handle:
        sealed = json.load(handle)
    canonical = json.dumps(sealed, separators=(",", ":"), ensure_ascii=False).encode()
    commitment = hashlib.sha256(canonical).hexdigest()
    expected = manifest["provenance"]["sealed_ground_truth_sha256"]
    if commitment != expected:
        raise SystemExit(f"commitment mismatch: {commitment} != {expected}")
    training_budget = manifest["scenario"]["training_budget_per_policy_actor_mode"]
    recovery = {}
    coverage = []
    normalized_errors = []
    errors_by_policy = {"active": [], "random": []}
    coverage_by_policy = {"active": [], "random": []}
    errors_by_mode_policy = {}
    for actor, truth in sealed["actor_weights"].items():
        recovery[actor] = {}
        for mode, mode_result in blinded["actors"][actor].items():
            recovery[actor][mode] = {}
            for policy in ("active", "random"):
                final = mode_result[policy]["checkpoints"][str(training_budget)]["posterior"]
                parameters = {}
                errors = []
                for name, (low, high) in RANGES.items():
                    estimate = final["parameters"][name]
                    error = abs(estimate["mean"] - truth[name]) / (high - low)
                    covered = estimate["q05"] <= truth[name] <= estimate["q95"]
                    errors.append(error)
                    normalized_errors.append(error)
                    coverage.append(covered)
                    parameters[name] = {
                        "true": truth[name], "posterior_mean": estimate["mean"],
                        "normalized_absolute_error": error, "in_90pct_interval": covered,
                    }
                recovery[actor][mode][policy] = {
                    "mean_normalized_weight_error": statistics.fmean(errors),
                    "parameters": parameters,
                }
                errors_by_policy[policy].append(statistics.fmean(errors))
                coverage_by_policy[policy].extend(
                    parameters[name]["in_90pct_interval"] for name in parameters
                )
                errors_by_mode_policy.setdefault((mode, policy), []).append(statistics.fmean(errors))
    optimal = {}
    for mode in sorted({row["observation_mode"] for row in rows}):
        subset = [row for row in rows if row["observation_mode"] == mode]
        correct = 0
        for row in subset:
            truth = sealed["actor_weights"][row["actor_id"]]
            predicted_alpha = utility(row["scenario"]["alpha"], truth) > utility(row["scenario"]["beta"], truth)
            correct += predicted_alpha == action_is_alpha(row["decision"]["action"])
        optimal[mode] = {"optimal": correct, "trials": len(subset), "rate": correct / len(subset)}
    actor_weight_differences = []
    for actor in recovery:
        active = statistics.fmean(recovery[actor][mode]["active"]["mean_normalized_weight_error"] for mode in recovery[actor])
        random_error = statistics.fmean(recovery[actor][mode]["random"]["mean_normalized_weight_error"] for mode in recovery[actor])
        actor_weight_differences.append(random_error - active)
    result = {
        "analysis_status": "UNBLINDED",
        "created_at_utc": utc_now(),
        "run_id": manifest["run_id"],
        "commitment_verified": True,
        "sealed_ground_truth_sha256": commitment,
        "blinded_active_vs_random": {
            "mean_full_budget_random_minus_active_log_loss": blinded["mean_full_budget_random_minus_active_log_loss"],
            "pairs": blinded["full_budget_pairs"],
            "actor_level_primary_inference": blinded["actor_level_primary_inference"],
        },
        "matched_card_geometry": blinded["matched_card_geometry"],
        "installed_utility_optimal_choice": optimal,
        "recovery": recovery,
        "aggregate_mean_normalized_weight_error": statistics.fmean(normalized_errors),
        "aggregate_90pct_interval_coverage": statistics.fmean(coverage),
        "weight_recovery_summary": {
            "mean_normalized_error_by_policy": {
                policy: statistics.fmean(values) for policy, values in errors_by_policy.items()
            },
            "mean_normalized_error_by_mode_policy": {
                f"{mode}:{policy}": statistics.fmean(values)
                for (mode, policy), values in errors_by_mode_policy.items()
            },
            "coverage_90pct_by_policy": {
                policy: statistics.fmean(values) for policy, values in coverage_by_policy.items()
            },
            "actor_random_minus_active_error": actor_weight_differences,
            "mean_actor_random_minus_active_error": statistics.fmean(actor_weight_differences),
            "one_sided_exact_sign_flip_p": exact_sign_flip_p_greater(actor_weight_differences),
            "actor_bootstrap_95pct": bootstrap_mean_interval(
                actor_weight_differences, stable_seed(manifest["run_id"], "weight-error-bootstrap")
            ),
        },
        "pilot_gates": {
            "first_response_valid_at_least_0_95": blinded["quality"]["first_response_valid_rate"] >= 0.95,
            "explicit_card_optimal_at_least_0_90": optimal["explicit_cards"]["rate"] >= 0.90,
            "posterior_numerically_stable": all(
                policy_result["checkpoints"][str(training_budget)]["posterior"]["effective_sample_size"] > 1
                for actor_result in blinded["actors"].values()
                for mode_result in actor_result.values()
                for policy, policy_result in mode_result.items() if policy in ("active", "random")
            ),
            "active_and_random_queries_nonidentical": blinded["quality"]["active_and_random_scenario_ids_disjoint"],
            "geometry_archived": "geometry_rules" in optimal,
        },
    }
    result["pilot_passes_all_gates"] = all(result["pilot_gates"].values())
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "analyze":
        analyze(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 6 and sys.argv[1] == "unblind":
        unblind(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        raise SystemExit(
            "usage:\n"
            "  analyze_continuous_reward.py analyze RUN_DIRECTORY BLINDED.json\n"
            "  analyze_continuous_reward.py unblind RUN_DIRECTORY BLINDED.json SEALED.json UNBLINDED.json"
        )


if __name__ == "__main__":
    main()
