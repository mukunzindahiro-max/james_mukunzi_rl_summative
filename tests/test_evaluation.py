from training.evaluation import EVAL_SEEDS, evaluate, nearest_policy, random_policy, run_episode


def test_baselines_never_take_invalid_actions():
    for policy in (random_policy(0), nearest_policy()):
        episode = run_episode(policy, EVAL_SEEDS[0])
        assert episode["invalid_actions"] == 0
        assert episode["completed"] + episode["unresolved"] == 50


def test_nearest_beats_random_on_held_out_seeds():
    seeds = EVAL_SEEDS[:5]
    nearest = evaluate(nearest_policy(), seeds=seeds)["summary"]
    random = evaluate(random_policy(0), seeds=seeds)["summary"]
    assert nearest["return_mean"] > random["return_mean"]
    assert nearest["total_km_mean"] < random["total_km_mean"]


def test_rainy_season_scenario_is_harder():
    seeds = EVAL_SEEDS[:5]
    standard = evaluate(nearest_policy(), "standard", seeds)["summary"]
    rainy = evaluate(nearest_policy(), "rainy_season", seeds)["summary"]
    assert rainy["mean_referral_minutes_mean"] > standard["mean_referral_minutes_mean"]
