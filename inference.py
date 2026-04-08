"""
Inference Script — Agri Alert RL (Rice Blast Detection)
=========================================================
MANDATORY env vars:
  API_BASE_URL  - The API endpoint for the LLM
  MODEL_NAME    - The model identifier to use for inference
  HF_TOKEN      - Your Hugging Face / API key (also checked as API_KEY)

STDOUT FORMAT:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
import textwrap
from typing import List, Optional

from openai import OpenAI

# Env vars — read at module level with defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "agri-alert-rl"
NUM_EPISODES = 1  # 1 episode per task to stay within 20 min
SUCCESS_SCORE_THRESHOLD = 0.5
TEMPERATURE = 0.0
MAX_TOKENS = 50
FALLBACK_ACTION = "do_nothing"
VALID_INTERVENTIONS = [
    "do_nothing", "send_alert", "apply_fungicide",
    "call_agronomist", "increase_monitoring_frequency",
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


# ---------------------------------------------------------------------------
# Structured log helpers — MUST match exact format
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def parse_action(response_text: str) -> str:
    if not response_text:
        return FALLBACK_ACTION
    text = response_text.strip().lower()
    for action in VALID_INTERVENTIONS:
        if action in text:
            return action
    return FALLBACK_ACTION


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(env, task: str, grader_cls, client: OpenAI, model_name: str) -> float:
    from models import RiceBlastAction

    obs = env.reset(task=task)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=model_name)

    try:
        done = False
        trajectory = []
        while not done:
            obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)
            prompt = json.dumps({
                "lesion_coverage": round(obs_dict.get("lesion_coverage", 0), 3),
                "lesion_pattern": obs_dict.get("lesion_pattern", "none"),
                "disease_stage": obs_dict.get("fields", [{}])[0].get("disease_stage", "none") if obs_dict.get("fields") else "none",
                "humidity": round(obs_dict.get("humidity", 0), 2),
                "timestep": obs_dict.get("timestep", 0),
            })

            error = None
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
                error = str(exc)[:100]
                response_text = FALLBACK_ACTION

            intervention = parse_action(response_text)
            action = RiceBlastAction(intervention=intervention)

            result = env.step(action)
            if isinstance(result, tuple):
                obs, reward, done, info = result
            else:
                obs, reward, done, info = result.observation, result.reward, result.done, result.info

            trajectory.append(info)
            steps_taken += 1
            rewards.append(reward)
            log_step(step=steps_taken, action=intervention, reward=reward, done=done, error=error)

        score = grader_cls().grade(trajectory)
        score = min(max(score, 0.001), 0.999)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            env.close() if hasattr(env, 'close') else None
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Re-read at runtime so validator-injected env vars are picked up
    api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

    if not api_key:
        print("Error: HF_TOKEN (or API_KEY) environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=api_base_url, api_key=api_key)

    from server.environment import RiceBlastEnv, EasyGrader, MediumGrader, HardGrader

    grader_map = {"easy": EasyGrader, "medium": MediumGrader, "hard": HardGrader}
    env = RiceBlastEnv()
    all_scores = {}

    for task in TASKS:
        score = run_episode(env, task, grader_map[task], client, model_name)
        all_scores[task] = score

    aggregate = sum(all_scores.values()) / len(all_scores)
    print(f"aggregate: {aggregate:.3f}", flush=True)


if __name__ == "__main__":
    main()
