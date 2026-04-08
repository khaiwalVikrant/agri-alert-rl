"""
Inference Script for Rice Blast RL Environment
================================================
MANDATORY env vars:
  API_BASE_URL  - The API endpoint for the LLM (default: https://router.huggingface.co/v1)
  MODEL_NAME    - The model identifier to use for inference
  HF_TOKEN      - Your Hugging Face / API key (also checked as API_KEY)
"""

import os
import sys
import re
import json
import textwrap
from typing import Optional

from openai import OpenAI

# Read env vars at module level
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# Validate required env vars — deferred to main() so imports work cleanly

# Constants
NUM_EPISODES = 3
TEMPERATURE = 0.0
MAX_TOKENS = 50
FALLBACK_ACTION = "do_nothing"
VALID_INTERVENTIONS = [
    "do_nothing", "send_alert", "apply_fungicide",
    "call_agronomist", "increase_monitoring_frequency"
]
TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = textwrap.dedent("""
You are an agricultural AI agent managing rice blast disease in rice fields.
Given a field observation as JSON, respond with EXACTLY ONE of these action strings:
  do_nothing
  send_alert
  apply_fungicide
  call_agronomist
  increase_monitoring_frequency

Rules:
- If lesion_coverage > 0.05 and lesion_pattern is diamond or gray_center: use send_alert or apply_fungicide
- If lesion_pattern is environmental_stress and humidity > 0.7: use increase_monitoring_frequency
- If disease_stage is late: use apply_fungicide
- Otherwise: use do_nothing

Respond with ONLY the action string. No explanation.
""").strip()

def parse_action(response_text: str) -> str:
    """Parse model response into a valid intervention string."""
    if not response_text:
        return FALLBACK_ACTION
    text = response_text.strip().lower()
    for action in VALID_INTERVENTIONS:
        if action in text:
            return action
    return FALLBACK_ACTION

def run_episode(env, task: str, grader_cls, client: OpenAI, model_name: str) -> float:
    """Run one episode and return the grader score."""
    obs = env.reset(task=task)
    trajectory = []
    done = False

    while not done:
        # Build prompt from observation
        obs_dict = obs.model_dump() if hasattr(obs, 'model_dump') else dict(obs)
        # Simplify for token efficiency
        prompt = json.dumps({
            "lesion_coverage": round(obs_dict.get("lesion_coverage", 0), 3),
            "lesion_pattern": obs_dict.get("lesion_pattern", "none"),
            "disease_stage": obs_dict.get("fields", [{}])[0].get("disease_stage", "none") if obs_dict.get("fields") else "none",
            "humidity": round(obs_dict.get("humidity", 0), 2),
            "timestep": obs_dict.get("timestep", 0),
        })

        try:
            completion = client.chat.completions.create(
                model=model_name,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  Model request failed: {exc}. Using fallback.", file=sys.stderr)
            response_text = FALLBACK_ACTION

        intervention = parse_action(response_text)

        # Import here to avoid circular imports
        from models import RiceBlastAction
        action = RiceBlastAction(intervention=intervention)

        result = env.step(action)
        if isinstance(result, tuple):
            obs, reward, done, info = result
        else:
            obs, reward, done, info = result.observation, result.reward, result.done, result.info

        trajectory.append(info)

    return grader_cls().grade(trajectory)


def main() -> None:
    # Re-read env vars at runtime (validator injects them before calling main)
    api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    model_name = os.getenv("MODEL_NAME")

    if not api_key:
        print("Error: HF_TOKEN (or API_KEY) environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    if not model_name:
        print("Error: MODEL_NAME environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=api_base_url, api_key=api_key)

    # Use the sync environment directly (no server needed for inference)
    from server.environment import RiceBlastEnv, EasyGrader, MediumGrader, HardGrader

    grader_map = {
        "easy": EasyGrader,
        "medium": MediumGrader,
        "hard": HardGrader,
    }

    env = RiceBlastEnv()
    scores = {}

    for task in TASKS:
        task_scores = []
        for ep in range(NUM_EPISODES):
            score = run_episode(env, task, grader_map[task], client, model_name)
            task_scores.append(score)
            print(f"  {task} episode {ep+1}/{NUM_EPISODES}: {score:.3f}")
        scores[task] = sum(task_scores) / len(task_scores)
        print(f"{task}: {scores[task]:.3f}")

    aggregate = sum(scores.values()) / len(scores)
    print(f"aggregate: {aggregate:.3f}")


if __name__ == "__main__":
    main()
