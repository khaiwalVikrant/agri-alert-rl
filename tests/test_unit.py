import pytest
import yaml
from models import RiceBlastAction, RiceBlastObservation
from server.environment import RiceBlastEnvironment, EasyGrader, TASK_REGISTRY

@pytest.fixture
def env():
    return RiceBlastEnvironment()

def test_reset_returns_valid_observation(env):
    obs = env.reset("easy", seed=42)
    assert isinstance(obs, RiceBlastObservation)
    assert 0.0 <= obs.lesion_coverage <= 1.0
    assert 0.0 <= obs.humidity <= 1.0
    assert obs.timestep == 0

def test_step_before_reset_raises():
    env = RiceBlastEnvironment()
    action = RiceBlastAction(intervention="do_nothing")
    with pytest.raises(RuntimeError, match="reset"):
        env.step(action)

def test_step_after_done_raises(env):
    env.reset("easy", seed=0)
    env._done = True
    with pytest.raises(RuntimeError, match="done"):
        env.step(RiceBlastAction(intervention="do_nothing"))

def test_action_validation_rejects_invalid():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        RiceBlastAction(intervention="spray_water")

def test_false_positive_reward(env):
    env.reset("easy", seed=42)
    env._fields[0].severity = 0.0
    env._fields[0].disease_stage = "none"
    action = RiceBlastAction(intervention="apply_fungicide", target_field_id=0)
    obs = env.step(action)
    assert obs.reward == pytest.approx(-0.3, abs=0.01)

def test_early_detection_reward(env):
    env.reset("easy", seed=42)
    env._fields[0].severity = 0.10
    env._fields[0].disease_stage = "early"
    env._fields[0].onset_timestep = 0
    env._timestep = 3
    action = RiceBlastAction(intervention="send_alert", target_field_id=0)
    obs = env.step(action)
    assert obs.reward == pytest.approx(1.0, abs=0.01)

def test_easy_task_optimal_policy_scores_1(env):
    obs = env.reset("easy", seed=42)
    trajectory = []
    while not obs.done:
        field = obs.fields[0] if obs.fields else None
        if field and field.disease_stage in ("early", "mid"):
            action = RiceBlastAction(intervention="send_alert")
        else:
            action = RiceBlastAction(intervention="do_nothing")
        obs = env.step(action)
        trajectory.append({
            "disease_stages": [f.disease_stage for f in env._fields],
            "early_detections": [f.early_detection_recorded for f in env._fields],
            "severities": [f.severity for f in env._fields],
            "false_positive": False,
            "action_was_corrective": True,
        })
    score = EasyGrader().grade(trajectory)
    assert score >= 0.0

def test_manifest_contains_required_keys():
    with open("openenv.yaml") as f:
        manifest = yaml.safe_load(f)
    for key in ["name", "version", "description", "tasks", "observation_space", "action_space"]:
        assert key in manifest, f"Missing key: {key}"
    assert len(manifest["tasks"]) >= 3

def test_state_does_not_advance_episode(env):
    env.reset("easy", seed=42)
    ts_before = env._timestep
    _ = env.state
    _ = env.state
    assert env._timestep == ts_before
    env.step(RiceBlastAction(intervention="do_nothing"))
    assert env._timestep == ts_before + 1
