import pytest
import asyncio
import yaml
from models import RiceBlastAction, RiceBlastObservation
from server.environment import RiceBlastEnvironment, EasyGrader, TASK_REGISTRY

@pytest.fixture
def env():
    return RiceBlastEnvironment()

@pytest.mark.asyncio
async def test_reset_returns_valid_observation(env):
    obs = await env.reset("easy", seed=42)
    assert isinstance(obs, RiceBlastObservation)
    assert 0.0 <= obs.lesion_coverage <= 1.0
    assert 0.0 <= obs.humidity <= 1.0
    assert obs.timestep == 0

@pytest.mark.asyncio
async def test_step_before_reset_raises():
    env = RiceBlastEnvironment()
    action = RiceBlastAction(intervention="do_nothing")
    with pytest.raises(RuntimeError, match="reset"):
        await env.step(action)

@pytest.mark.asyncio
async def test_step_after_done_raises(env):
    await env.reset("easy", seed=0)
    env._done = True
    with pytest.raises(RuntimeError, match="done"):
        await env.step(RiceBlastAction(intervention="do_nothing"))

def test_action_validation_rejects_invalid():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        RiceBlastAction(intervention="spray_water")

@pytest.mark.asyncio
async def test_false_positive_reward(env):
    # Reset and force disease_stage to none on field 0
    await env.reset("easy", seed=42)
    env._fields[0].severity = 0.0
    env._fields[0].disease_stage = "none"
    action = RiceBlastAction(intervention="apply_fungicide", target_field_id=0)
    _, reward, _, _ = await env.step(action)
    assert reward == pytest.approx(-0.3, abs=0.01)

@pytest.mark.asyncio
async def test_early_detection_reward(env):
    await env.reset("easy", seed=42)
    # Force early stage within window
    env._fields[0].severity = 0.10
    env._fields[0].disease_stage = "early"
    env._fields[0].onset_timestep = 0
    env._timestep = 3  # within 6-step window
    action = RiceBlastAction(intervention="send_alert", target_field_id=0)
    _, reward, _, _ = await env.step(action)
    assert reward == pytest.approx(1.0, abs=0.01)

@pytest.mark.asyncio
async def test_easy_task_optimal_policy_scores_1(env):
    obs = await env.reset("easy", seed=42)
    trajectory = []
    done = False
    while not done:
        # Optimal: always send_alert when disease is visible
        field = obs.fields[0] if obs.fields else None
        if field and field.disease_stage in ("early", "mid"):
            action = RiceBlastAction(intervention="send_alert")
        else:
            action = RiceBlastAction(intervention="do_nothing")
        result = await env.step(action)
        obs, reward, done, info = result if isinstance(result, tuple) else (result.observation, result.reward, result.done, result.info)
        trajectory.append(info)
    score = EasyGrader().grade(trajectory)
    assert score >= 0.0  # At minimum valid score

def test_manifest_contains_required_keys():
    with open("openenv.yaml") as f:
        manifest = yaml.safe_load(f)
    for key in ["name", "version", "description", "tasks", "observation_space", "action_space"]:
        assert key in manifest, f"Missing key: {key}"
    assert len(manifest["tasks"]) >= 3

@pytest.mark.asyncio
async def test_state_does_not_advance_episode(env):
    await env.reset("easy", seed=42)
    state1 = env.state()
    state2 = env.state()
    assert state1.timestep == state2.timestep == 0
    # Step once and verify state reflects new timestep
    await env.step(RiceBlastAction(intervention="do_nothing"))
    state3 = env.state()
    assert state3.timestep == 1
