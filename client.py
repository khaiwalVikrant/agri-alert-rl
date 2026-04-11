"""RiceBlastEnv client — what training code imports."""

from __future__ import annotations

import sys
import os
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import RiceBlastAction, RiceBlastObservation

try:
    from openenv.core import EnvClient
    from openenv.core.client_types import StepResult
    from openenv.core.env_server.types import State
    OPENENV_AVAILABLE = True
except ImportError:
    OPENENV_AVAILABLE = False


if OPENENV_AVAILABLE:
    class RiceBlastEnv(EnvClient[RiceBlastAction, RiceBlastObservation, State]):
        """OpenEnv-compliant client backed by openenv-core WebSocket transport."""

        def _step_payload(self, action: RiceBlastAction) -> Dict:
            """Convert RiceBlastAction to JSON payload for step message."""
            return {
                "intervention": action.intervention,
                "target_field_id": action.target_field_id,
            }

        def _parse_result(self, payload: Dict) -> StepResult[RiceBlastObservation]:
            """Parse server response into StepResult[RiceBlastObservation]."""
            obs_data = payload.get("observation", payload)
            observation = RiceBlastObservation(**obs_data)
            return StepResult(
                observation=observation,
                reward=payload.get("reward", 0.0),
                done=payload.get("done", False),
            )

        def _parse_state(self, payload: Dict) -> State:
            """Parse server response into State object."""
            return State(
                episode_id=payload.get("episode_id"),
                step_count=payload.get("step_count", 0),
            )

else:
    import httpx

    class RiceBlastEnv:  # type: ignore[no-redef]
        """Sync HTTP client fallback when openenv-core is absent."""

        def __init__(self, base_url: str = "https://khaiwal009-agri-alert-rl.hf.space"):
            self.base_url = base_url.rstrip("/")
            self._client = httpx.Client(timeout=30.0)

        def reset(self, task: str = "easy", seed: int | None = None) -> RiceBlastObservation:
            params: dict = {"task": task}
            if seed is not None:
                params["seed"] = seed
            r = self._client.post(f"{self.base_url}/reset", params=params)
            r.raise_for_status()
            return RiceBlastObservation(**r.json())

        def step(self, action: RiceBlastAction) -> tuple[RiceBlastObservation, float, bool, dict]:
            r = self._client.post(f"{self.base_url}/step", json=action.model_dump())
            r.raise_for_status()
            data = r.json()
            return (
                RiceBlastObservation(**data["observation"]),
                data["reward"],
                data["done"],
                data["info"],
            )

        def state(self) -> RiceBlastObservation:
            r = self._client.get(f"{self.base_url}/state")
            r.raise_for_status()
            return RiceBlastObservation(**r.json())

        def sync(self) -> "RiceBlastEnv":
            return self

        def close(self) -> None:
            self._client.close()

        def __enter__(self) -> "RiceBlastEnv":
            return self

        def __exit__(self, *_) -> None:
            self.close()
