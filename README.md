---
title: Agri Alert RL
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
base_path: /web
tags:
  - openenv
  - agriculture
  - disease-detection
  - reinforcement-learning
---

# Rice Blast RL Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant reinforcement learning environment simulating **rice blast disease** (*Magnaporthe oryzae*) early detection and treatment decision-making.

## Domain Description

Rice blast is one of the most destructive rice diseases worldwide. Once onset occurs, it can spread from early symptoms to total crop loss within **24 simulated hours** without intervention. Farmers need timely, actionable alerts to intervene before the disease spreads.

An agent observes simulated crop image features, weather data, and field metadata each timestep, then selects one of five interventions. The challenge: detect disease early enough to act, without triggering costly false alarms.

## Observation Space

| Field | Type | Range |
|-------|------|-------|
| `lesion_coverage` | float | [0.0, 1.0] |
| `leaf_color_index` | float | [0.0, 1.0] |
| `lesion_pattern` | categorical | none, diamond, gray_center, brown_border, environmental_stress |
| `temperature` | float | 15.0–35.0 °C |
| `humidity` | float | [0.0, 1.0] |
| `rainfall` | float | ≥ 0.0 mm/hr |
| `wind_speed` | float | ≥ 0.0 m/s |
| `crop_stage` | categorical | seedling, tillering, booting, heading, ripening |
| `days_since_last_treatment` | int | ≥ 0 |
| `field_size_ha` | float | > 0.0 ha |
| `timestep` | int | ≥ 0 |
| `fields` | list[FieldObservation] | per-field detail (1 or 3 entries) |

## Action Space

| Action | Description |
|--------|-------------|
| `do_nothing` | No intervention |
| `send_alert` | Send disease alert (low cost, early detection signal) |
| `apply_fungicide` | Apply chemical treatment (effective in early/mid stage) |
| `call_agronomist` | Request expert consultation |
| `increase_monitoring_frequency` | Increase observation fidelity next step |

## Tasks

| Task | Difficulty | Fields | Description |
|------|-----------|--------|-------------|
| `easy` | Easy | 1 | Clear diamond lesion pattern, low humidity noise. Detect within 6-step window. |
| `medium` | Medium | 1 | High humidity (0.75–0.90) causes environmental_stress patterns. Distinguish real blast from noise. |
| `hard` | Hard | 3 | Three fields with staggered disease onset. One treatment per timestep. Minimize total crop loss. |

## Reward Function

```
reward = clamp(raw_reward, 0.0, 1.0)

+1.0  early detection: action ∈ {send_alert, apply_fungicide, call_agronomist}
       AND disease_stage == "early" AND timestep - onset ≤ 6
+0.5  mid-stage intervention: same actions AND disease_stage == "mid"
+0.5  neutral: correct action on healthy field
+0.2  inaction penalty: do_nothing AND disease_stage ∈ {early, mid}
+0.1  false positive: apply_fungicide AND disease_stage == "none"
 0.0  missed disease: disease_stage == "late" AND no prior intervention
```

## Baseline Scores

Scores produced by `llama3.2` (via Ollama) with `NUM_EPISODES=3`:

| Task | Score |
|------|-------|
| easy | 0.500 |
| medium | 0.333 |
| hard | 0.078 |
| aggregate | 0.304 |

## Setup

### Requirements
- Python 3.10+
- Docker (for deployment)
- Hugging Face CLI (for HF Spaces deployment)

### Local Installation

```bash
pip install -e .
# or
pip install -r requirements.txt
```

### Run Server Locally

```bash
cd server
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Minimal Usage Example

```python
from server.environment import RiceBlastEnv
from models import RiceBlastAction

env = RiceBlastEnv()
obs = env.reset(task="easy", seed=42)
print(f"Initial observation: timestep={obs.timestep}, lesion_coverage={obs.lesion_coverage:.3f}")

action = RiceBlastAction(intervention="send_alert")
obs, reward, done, info = env.step(action)
print(f"Reward: {reward}, Done: {done}")
```

### Run Inference Script

```bash
export HF_TOKEN=your_token
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

### Docker Build & Run

```bash
docker build -f server/Dockerfile -t rice-blast-env .
docker run -p 7860:7860 rice-blast-env
```

### Deploy to Hugging Face Spaces

```bash
pip install huggingface_hub
huggingface-cli login
openenv push --repo-id your-username/rice-blast-env
```
