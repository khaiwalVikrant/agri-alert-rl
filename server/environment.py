"""All simulation logic: WeatherState, FieldState, TaskConfig, TASK_REGISTRY,
DiseaseSimulator, Graders, and RiceBlastEnvironment (async)."""

from __future__ import annotations

import copy
import sys
import os
from dataclasses import dataclass
from typing import Literal

import numpy as np

# Ensure root is on path so models.py can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import RiceBlastAction, RiceBlastObservation, FieldObservation

# Try to import openenv-core base classes
try:
    from openenv.core.environment import Environment as _BaseEnvironment
    from openenv.core.models import StepResult
    OPENENV_AVAILABLE = True
except ImportError:
    _BaseEnvironment = object
    StepResult = None
    OPENENV_AVAILABLE = False


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WeatherState:
    temperature: float   # 15–35 Celsius
    humidity: float      # 0–1
    rainfall: float      # mm/hr >= 0
    wind_speed: float    # m/s >= 0


@dataclass
class FieldState:
    field_id: int
    severity: float                  # [0.0, 1.0]
    disease_stage: str               # "none" | "early" | "mid" | "late"
    onset_timestep: int              # when disease begins spreading (-1 if not yet)
    days_since_treatment: int
    field_size_ha: float
    crop_stage: str
    last_treatment_timestep: int     # -999 if never treated
    early_detection_recorded: bool


@dataclass
class TaskConfig:
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    num_fields: int
    max_timesteps: int
    onset_timestep_range: tuple[int, int]
    base_humidity_range: tuple[float, float]
    humidity_noise_std: float
    initial_lesion_coverage: float
    force_lesion_pattern: str | None
    grader_class: str


TASK_REGISTRY: dict[str, TaskConfig] = {
    "easy": TaskConfig(
        name="easy", difficulty="easy", num_fields=1, max_timesteps=48,
        onset_timestep_range=(3, 5), base_humidity_range=(0.3, 0.5),
        humidity_noise_std=0.02, initial_lesion_coverage=0.08,
        force_lesion_pattern="diamond", grader_class="EasyGrader",
    ),
    "medium": TaskConfig(
        name="medium", difficulty="medium", num_fields=1, max_timesteps=48,
        onset_timestep_range=(3, 8), base_humidity_range=(0.75, 0.90),
        humidity_noise_std=0.10, initial_lesion_coverage=0.06,
        force_lesion_pattern=None, grader_class="MediumGrader",
    ),
    "hard": TaskConfig(
        name="hard", difficulty="hard", num_fields=3, max_timesteps=72,
        onset_timestep_range=(2, 15), base_humidity_range=(0.5, 0.8),
        humidity_noise_std=0.08, initial_lesion_coverage=0.05,
        force_lesion_pattern=None, grader_class="HardGrader",
    ),
}


# ---------------------------------------------------------------------------
# Disease stage helpers
# ---------------------------------------------------------------------------

def _severity_to_stage(severity: float) -> str:
    if severity < 0.05:
        return "none"
    elif severity < 0.20:
        return "early"
    elif severity < 0.50:
        return "mid"
    else:
        return "late"


# ---------------------------------------------------------------------------
# DiseaseSimulator
# ---------------------------------------------------------------------------

class DiseaseSimulator:
    BASE_SPREAD_RATE = 0.04

    def __init__(self, rng: np.random.Generator | None = None,
                 task_config: TaskConfig | None = None):
        self._rng = rng or np.random.default_rng()
        self._task_config = task_config

    def compute_weather_modifier(self, weather: WeatherState) -> float:
        modifier = (
            1.0
            + 0.02 * (weather.temperature - 25)
            + 0.5 * weather.humidity
            + 0.01 * weather.wind_speed
        )
        return float(np.clip(modifier, 0.5, 2.5))

    def advance(self, field: FieldState, timestep: int, weather: WeatherState) -> FieldState:
        """Advance disease spread for one timestep."""
        if field.onset_timestep <= timestep:
            weather_modifier = self.compute_weather_modifier(weather)
            spread = self.BASE_SPREAD_RATE * weather_modifier
            field.severity = min(1.0, field.severity + spread)
            field.days_since_treatment += 1
        field.disease_stage = _severity_to_stage(field.severity)
        return field

    def apply_intervention(self, field: FieldState, action: RiceBlastAction, timestep: int) -> FieldState:
        """Apply the chosen intervention to a field."""
        intervention = action.intervention

        if intervention == "apply_fungicide":
            base_eff = 0.4
            if field.days_since_treatment < 3:
                base_eff *= 0.5  # resistance modifier
            if field.disease_stage == "late":
                base_eff *= 0.2
            field.severity = max(0.0, field.severity * (1 - base_eff))
            field.days_since_treatment = 0
            field.last_treatment_timestep = timestep

        # Record early detection for alert/agronomist actions
        if intervention in {"send_alert", "call_agronomist", "apply_fungicide"}:
            if field.disease_stage == "early" and timestep - field.onset_timestep <= 6:
                field.early_detection_recorded = True

        field.disease_stage = _severity_to_stage(field.severity)
        return field

    def get_lesion_pattern(self, field: FieldState, weather: WeatherState) -> str:
        """Determine the visible lesion pattern for a field."""
        if field.disease_stage == "none":
            # Environmental stress can appear even without disease when humidity is high
            if weather.humidity > 0.7:
                return "environmental_stress"
            return "none"
        # Forced pattern from task config (e.g. easy task always shows diamond)
        if self._task_config and self._task_config.force_lesion_pattern:
            return self._task_config.force_lesion_pattern
        # Stochastic: high humidity can mask early blast as environmental stress
        if weather.humidity > 0.7 and field.disease_stage == "early":
            if self._rng.random() < 0.5:
                return "environmental_stress"
        if field.disease_stage == "early":
            return "diamond"
        if field.disease_stage == "mid":
            return "gray_center"
        return "brown_border"

    def get_leaf_color_index(self, field: FieldState) -> float:
        """Healthy = 1.0, fully diseased approaches 0.2."""
        return max(0.0, 1.0 - field.severity * 0.8)


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

class BaseGrader:
    def grade(self, trajectory: list[dict]) -> float:
        raise NotImplementedError


class EasyGrader(BaseGrader):
    """Score 1.0 for early detection within window, 0.5 for mid-stage intervention, else 0.0."""

    def grade(self, trajectory: list[dict]) -> float:
        # Check for early detection within window
        for step in trajectory:
            early_dets = step.get("early_detections", [False])
            if early_dets and early_dets[0]:
                return 0.999
        # Partial credit: mid-stage corrective intervention
        for step in trajectory:
            stages = step.get("disease_stages", ["none"])
            if stages and stages[0] == "mid" and step.get("action_was_corrective", False):
                return 0.5
        return 0.001


class MediumGrader(BaseGrader):
    """Score = correct_detections/1 - 0.3*false_positives, clamped (0, 1)."""

    def grade(self, trajectory: list[dict]) -> float:
        early_detected = any(
            step.get("early_detections", [False])[0]
            for step in trajectory
            if step.get("early_detections")
        )
        correct = 1.0 if early_detected else 0.0
        false_positives = sum(1 for step in trajectory if step.get("false_positive", False))
        penalty = min(1.0, 0.3 * false_positives)
        score = max(0.0, correct - penalty)
        return max(0.001, min(0.999, score))


class HardGrader(BaseGrader):
    """Score = 1.0 - mean(final_severities across 3 fields), clamped (0, 1)."""

    def grade(self, trajectory: list[dict]) -> float:
        if not trajectory:
            return 0.001
        last_info = trajectory[-1]
        severities = last_info.get("severities", [0.0])
        mean_severity = sum(severities) / len(severities) if severities else 0.0
        score = 1.0 - mean_severity
        return max(0.001, min(0.999, score))


# ---------------------------------------------------------------------------
# Helper maps
# ---------------------------------------------------------------------------

_CROP_STAGES = ["seedling", "tillering", "booting", "heading", "ripening"]
_GRADER_MAP: dict[str, type[BaseGrader]] = {
    "EasyGrader": EasyGrader,
    "MediumGrader": MediumGrader,
    "HardGrader": HardGrader,
}


# ---------------------------------------------------------------------------
# RiceBlastEnvironment (async, openenv-core compatible)
# ---------------------------------------------------------------------------

class RiceBlastEnvironment(_BaseEnvironment):
    """Async rice blast disease simulation environment."""

    def __init__(self) -> None:
        self._fields: list[FieldState] | None = None
        self._task_config: TaskConfig | None = None
        self._timestep: int = 0
        self._done: bool = False
        self._trajectory: list[dict] = []
        self._rng: np.random.Generator | None = None
        self._weather: WeatherState | None = None
        self._simulator: DiseaseSimulator | None = None

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def reset(self, task: str = "easy", seed: int | None = None) -> RiceBlastObservation:
        """Initialize a new episode and return the first observation."""
        self._rng = np.random.default_rng(seed)
        config = TASK_REGISTRY[task]
        self._task_config = config
        self._timestep = 0
        self._done = False
        self._trajectory = []
        self._simulator = DiseaseSimulator(rng=self._rng, task_config=config)

        # Initialize weather
        base_hum = float(self._rng.uniform(*config.base_humidity_range))
        self._weather = WeatherState(
            temperature=float(self._rng.uniform(20.0, 32.0)),
            humidity=float(np.clip(base_hum + self._rng.normal(0, config.humidity_noise_std), 0.0, 1.0)),
            rainfall=float(max(0.0, self._rng.normal(2.0, 1.0))),
            wind_speed=float(max(0.0, self._rng.normal(3.0, 1.0))),
        )

        # Initialize fields
        self._fields = []
        for i in range(config.num_fields):
            onset_ts = int(self._rng.integers(
                config.onset_timestep_range[0],
                config.onset_timestep_range[1] + 1,
            ))
            field_size = float(self._rng.uniform(0.5, 5.0))
            crop_stage = _CROP_STAGES[int(self._rng.integers(0, len(_CROP_STAGES)))]
            severity = config.initial_lesion_coverage if onset_ts == 0 else 0.01
            disease_stage = "early" if severity >= 0.05 else "none"
            self._fields.append(FieldState(
                field_id=i,
                severity=severity,
                disease_stage=disease_stage,
                onset_timestep=onset_ts,
                days_since_treatment=0,
                field_size_ha=field_size,
                crop_stage=crop_stage,
                last_treatment_timestep=-999,
                early_detection_recorded=False,
            ))

        return self._build_observation()

    async def step(self, action: RiceBlastAction):
        """Advance the simulation by one timestep and return result."""
        if self._fields is None:
            raise RuntimeError("Environment must be reset before stepping")
        if self._done:
            raise RuntimeError("Episode is done; call reset() to start a new episode")
        if action.target_field_id >= len(self._fields):
            # Clamp to valid range instead of crashing — handles UI defaults sending field_id=1 on single-field tasks
            action = RiceBlastAction(
                intervention=action.intervention,
                target_field_id=len(self._fields) - 1,
            )

        prev_fields = copy.deepcopy(self._fields)

        # Advance all fields
        for f in self._fields:
            self._simulator.advance(f, self._timestep, self._weather)

        # Apply intervention to target field
        target = self._fields[action.target_field_id]
        self._simulator.apply_intervention(target, action, self._timestep)

        # Update weather with small noise
        std = self._task_config.humidity_noise_std
        self._weather = WeatherState(
            temperature=float(np.clip(
                self._weather.temperature + self._rng.normal(0, 0.5), 15.0, 35.0
            )),
            humidity=float(np.clip(
                self._weather.humidity + self._rng.normal(0, std), 0.0, 1.0
            )),
            rainfall=float(max(0.0, self._weather.rainfall + self._rng.normal(0, 0.5))),
            wind_speed=float(max(0.0, self._weather.wind_speed + self._rng.normal(0, 0.3))),
        )

        self._timestep += 1

        reward = self._compute_reward(action, prev_fields, self._fields)
        self._done = self._is_terminal()

        # Determine false positive: fungicide applied when target field had no disease
        target_prev = next(f for f in prev_fields if f.field_id == action.target_field_id)
        false_positive = (
            action.intervention == "apply_fungicide"
            and target_prev.disease_stage == "none"
        )

        # Determine if action was corrective (for grader)
        action_was_corrective = action.intervention in {
            "send_alert", "apply_fungicide", "call_agronomist"
        }

        done_reason: str | None = None
        if self._timestep >= self._task_config.max_timesteps:
            done_reason = "max_steps"
        elif any(f.severity >= 1.0 for f in self._fields):
            done_reason = "crop_loss"

        info = {
            "timestep": self._timestep,
            "disease_stages": [f.disease_stage for f in self._fields],
            "early_detections": [f.early_detection_recorded for f in self._fields],
            "severities": [f.severity for f in self._fields],
            "false_positive": false_positive,
            "action_was_corrective": action_was_corrective,
            "episode_done_reason": done_reason,
        }
        self._trajectory.append(info)

        obs = self._build_observation()
        obs.reward = reward
        obs.done = self._done

        if OPENENV_AVAILABLE and StepResult is not None:
            return StepResult(observation=obs, reward=reward, done=self._done, info=info)
        return obs, reward, self._done, info

    def state(self) -> RiceBlastObservation:
        """Return current observation without advancing the episode (sync, required by openenv-core)."""
        if self._fields is None:
            raise RuntimeError("Environment must be reset before calling state()")
        return self._build_observation()

    async def async_state(self) -> RiceBlastObservation:
        """Async version for direct async usage."""
        return self.state()

    async def reset_async(self, task: str = "easy", seed: int | None = None) -> RiceBlastObservation:
        """Async alias for reset — used by openenv-core web interface."""
        return await self.reset(task=task, seed=seed)

    async def step_async(self, action: RiceBlastAction):
        """Async alias for step — used by openenv-core web interface."""
        return await self.step(action)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        action: RiceBlastAction,
        prev_fields: list[FieldState],
        new_fields: list[FieldState],
    ) -> float:
        rewards: list[float] = []

        for prev_f in prev_fields:
            is_target = (prev_f.field_id == action.target_field_id)

            if is_target:
                r = self._reward_for_target(action, prev_f)
            else:
                # Non-target fields: penalise if late stage with no prior intervention
                new_f = next(f for f in new_fields if f.field_id == prev_f.field_id)
                if new_f.disease_stage == "late" and not new_f.early_detection_recorded:
                    r = -0.5
                else:
                    r = 0.0
            rewards.append(r)

        raw = sum(rewards) / len(rewards) if rewards else 0.0
        return float(max(-1.0, min(1.0, raw)))

    def _reward_for_target(self, action: RiceBlastAction, prev_f: FieldState) -> float:
        """Compute reward for the targeted field based on the 5 reward cases."""
        intervention = action.intervention

        # Case 1: early detection within window
        if (intervention in {"send_alert", "apply_fungicide", "call_agronomist"}
                and prev_f.disease_stage == "early"
                and self._timestep - prev_f.onset_timestep <= 6):
            return 1.0

        # Case 2: mid-stage corrective intervention
        if (intervention in {"send_alert", "apply_fungicide", "call_agronomist"}
                and prev_f.disease_stage == "mid"):
            return 0.5

        # Case 3: false positive fungicide
        if intervention == "apply_fungicide" and prev_f.disease_stage == "none":
            return -0.3

        # Case 4: late stage with no prior corrective intervention
        if prev_f.disease_stage == "late" and not prev_f.early_detection_recorded:
            return -0.5

        # Case 5: inaction during active disease
        if intervention == "do_nothing" and prev_f.disease_stage in {"early", "mid"}:
            return -0.1

        return 0.0

    def _build_observation(self) -> RiceBlastObservation:
        primary = self._fields[0]
        weather = self._weather
        lesion_pattern = self._simulator.get_lesion_pattern(primary, weather)
        leaf_color = self._simulator.get_leaf_color_index(primary)

        field_obs = [
            FieldObservation(
                field_id=f.field_id,
                lesion_coverage=min(1.0, f.severity),
                leaf_color_index=self._simulator.get_leaf_color_index(f),
                lesion_pattern=self._simulator.get_lesion_pattern(f, weather),
                crop_stage=f.crop_stage,
                days_since_last_treatment=f.days_since_treatment,
                field_size_ha=f.field_size_ha,
                disease_stage=f.disease_stage,
            )
            for f in self._fields
        ]

        return RiceBlastObservation(
            lesion_coverage=min(1.0, primary.severity),
            leaf_color_index=leaf_color,
            lesion_pattern=lesion_pattern,
            temperature=weather.temperature,
            humidity=min(1.0, max(0.0, weather.humidity)),
            rainfall=max(0.0, weather.rainfall),
            wind_speed=max(0.0, weather.wind_speed),
            crop_stage=primary.crop_stage,
            days_since_last_treatment=primary.days_since_treatment,
            field_size_ha=primary.field_size_ha,
            fields=field_obs,
            timestep=self._timestep,
        )

    def _is_terminal(self) -> bool:
        return (
            self._timestep >= self._task_config.max_timesteps
            or any(f.severity >= 1.0 for f in self._fields)
        )

    def close(self) -> None:
        """Clean up episode state. Required by openenv-core (sync)."""
        self._fields = None
        self._task_config = None
        self._done = False
        self._trajectory = []


# ---------------------------------------------------------------------------
# Backwards-compatible sync alias (used by existing tests and app.py)
# ---------------------------------------------------------------------------

class RiceBlastEnv:
    """Synchronous wrapper around RiceBlastEnvironment for backwards compatibility."""

    def __init__(self) -> None:
        import asyncio
        self._env = RiceBlastEnvironment()
        self._loop = asyncio.new_event_loop()

    def reset(self, task: str = "easy", seed: int | None = None):
        return self._loop.run_until_complete(self._env.reset(task=task, seed=seed))

    def step(self, action):
        result = self._loop.run_until_complete(self._env.step(action))
        # Normalise to tuple regardless of StepResult vs tuple
        if OPENENV_AVAILABLE and StepResult is not None and hasattr(result, "observation"):
            return result.observation, result.reward, result.done, result.info
        return result

    def state(self):
        return self._env.state()

    def __del__(self):
        try:
            self._loop.close()
        except Exception:
            pass
