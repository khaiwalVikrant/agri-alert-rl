import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core"
    ) from e

from models import RiceBlastAction, RiceBlastObservation
from server.environment import RiceBlastEnvironment

# create_app wires up /reset, /step, /state, /health, /ws, /schema, /web, /docs
app = create_app(
    RiceBlastEnvironment,
    RiceBlastAction,
    RiceBlastObservation,
    env_name="rice-blast-env",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)
