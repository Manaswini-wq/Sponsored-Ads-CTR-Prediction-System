from src.ab_testing.experiment import (
    ExperimentManager, StatisticalAnalyzer,
)


def test_deterministic_assignment():
    mgr = ExperimentManager()
    exp = mgr.create_experiment("test", "model_a", "model_b", traffic_split=0.5)

    # Same user always gets the same variant
    v1 = mgr.assign_variant("user_123", exp.experiment_id)
    v2 = mgr.assign_variant("user_123", exp.experiment_id)
    assert v1 == v2
    assert v1 in ("control", "treatment")


def test_experiment_results():
    mgr = ExperimentManager()
    exp = mgr.create_experiment("test", "model_a", "model_b", traffic_split=0.5)

    # Simulate outcomes
    for i in range(200):
        uid = f"user_{i}"
        variant = mgr.assign_variant(uid, exp.experiment_id)
        clicked = i % 3 == 0  # ~33% CTR for both
        mgr.record_outcome(exp.experiment_id, uid, variant, clicked)

    results = mgr.get_results(exp.experiment_id)
    assert results.control_samples > 0
    assert results.treatment_samples > 0
    assert 0 <= results.p_value <= 1


def test_minimum_sample_size():
    n = StatisticalAnalyzer.compute_minimum_sample_size(
        baseline_ctr=0.03, mde=0.005
    )
    assert n > 0
    assert isinstance(n, int)
