"""RiceBlastEnv client — what training code imports."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import RiceBlastAction, RiceBlastObservation

try:
    from openenv.core.env_client import EnvClient as _BaseEnvClient
    OPENENV_AVAILABLE = True
except ImportError:
    _BaseEnvClient = object
    OPENENV_AVAILABLE = False


if OPENENV_AVAILABLE:
    class RiceBlastEnv(_BaseEnvClient):
        """OpenEnv-compliant async client backed by openenv-core WebSocket transport."""
        action_type = RiceBlastAction
        observation_type = RiceBlastObservation

else:
    import httpx

    class RiceBlastEnv:  # type: ignore[no-redef]
        """Sync HTTP client for the Rice Blast FastAPI server (fallback when openenv-core is absent)."""

        def __init__(self, base_url: str = "https://khaiwal009-agri-alert-rl.hf.space"):
            self.base_url = base_url.rstrip("/")
            self._client = httpx.Client(timeout=30.0)

        # ------------------------------------------------------------------
        # Public API
        # ------------------------------------------------------------------

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
            """Return self for compatibility with openenv-core sync() pattern."""
            return self

        def close(self) -> None:
            self._client.close()

        def __enter__(self) -> "RiceBlastEnv":
            return self

        def __exit__(self, *_) -> None:
            self.close()
