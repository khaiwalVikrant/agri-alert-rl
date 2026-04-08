import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from typing import Optional
import uvicorn

from models import RiceBlastAction, RiceBlastObservation
from server.environment import RiceBlastEnvironment

app = FastAPI(title="Agri Alert RL — Rice Blast Detection Environment")
env = RiceBlastEnvironment()

WEB_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agri Alert RL — Rice Blast Detection</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #f8faf8; color: #1a1a1a; }
  h1 { color: #2d6a2d; } h2 { color: #3a7a3a; border-bottom: 2px solid #c8e6c8; padding-bottom: 6px; }
  .card { background: white; border-radius: 10px; padding: 20px; margin: 16px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
  select, input, button { padding: 8px 14px; border-radius: 6px; border: 1px solid #ccc; font-size: 14px; margin: 4px; }
  button { background: #2d6a2d; color: white; border: none; cursor: pointer; font-weight: 600; }
  button:hover { background: #1e4d1e; }
  button.secondary { background: #5a8a5a; }
  pre { background: #f0f7f0; border-radius: 6px; padding: 14px; overflow-x: auto; font-size: 13px; white-space: pre-wrap; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .easy { background: #c8e6c8; color: #1b5e20; }
  .medium { background: #fff9c4; color: #f57f17; }
  .hard { background: #ffcdd2; color: #b71c1c; }
  .reward-pos { color: #2d6a2d; font-weight: bold; }
  .reward-neg { color: #c62828; font-weight: bold; }
  #log { max-height: 300px; overflow-y: auto; }
</style>
</head>
<body>
<h1>🌾 Agri Alert RL</h1>
<p>Rice Blast Disease Detection — OpenEnv-compliant RL Environment</p>

<div class="card">
  <h2>Episode Control</h2>
  <label>Task:
    <select id="task">
      <option value="easy">Easy <span class="badge easy">easy</span></option>
      <option value="medium">Medium</option>
      <option value="hard">Hard</option>
    </select>
  </label>
  <label>Seed: <input type="number" id="seed" value="42" style="width:80px"></label>
  <button onclick="resetEnv()">▶ Reset Episode</button>
  <span id="episode-status" style="margin-left:12px;color:#666;font-size:13px;"></span>
</div>

<div class="card" id="obs-card" style="display:none">
  <h2>Current Observation</h2>
  <div id="obs-summary" style="margin-bottom:10px;font-size:14px;"></div>
  <pre id="obs-json"></pre>
</div>

<div class="card" id="action-card" style="display:none">
  <h2>Take Action</h2>
  <label>Intervention:
    <select id="intervention">
      <option value="do_nothing">do_nothing</option>
      <option value="send_alert">send_alert</option>
      <option value="apply_fungicide">apply_fungicide</option>
      <option value="call_agronomist">call_agronomist</option>
      <option value="increase_monitoring_frequency">increase_monitoring_frequency</option>
    </select>
  </label>
  <label>Field ID: <input type="number" id="field-id" value="0" style="width:60px" min="0" max="2"></label>
  <button onclick="stepEnv()">⚡ Step</button>
</div>

<div class="card" id="log-card" style="display:none">
  <h2>Episode Log</h2>
  <button class="secondary" onclick="document.getElementById('log').innerHTML=''">Clear</button>
  <div id="log"></div>
</div>

<script>
const BASE = '';
let stepCount = 0;

async function resetEnv() {
  const task = document.getElementById('task').value;
  const seed = document.getElementById('seed').value;
  const r = await fetch(`${BASE}/reset?task=${task}&seed=${seed}`, {method:'POST'});
  const obs = await r.json();
  stepCount = 0;
  document.getElementById('episode-status').textContent = `Task: ${task} | Seed: ${seed}`;
  document.getElementById('obs-card').style.display = 'block';
  document.getElementById('action-card').style.display = 'block';
  document.getElementById('log-card').style.display = 'block';
  showObs(obs);
  addLog(`🔄 Episode reset — task: <b>${task}</b>, seed: <b>${seed}</b>`);
}

async function stepEnv() {
  const intervention = document.getElementById('intervention').value;
  const target_field_id = parseInt(document.getElementById('field-id').value);
  const r = await fetch(`${BASE}/step`, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({intervention, target_field_id})
  });
  const data = await r.json();
  if (r.status === 400) { addLog(`⚠️ ${data.detail}`); return; }
  stepCount++;
  showObs(data.observation);
  const rewardClass = data.reward >= 0 ? 'reward-pos' : 'reward-neg';
  const rewardSign = data.reward >= 0 ? '+' : '';
  addLog(`Step ${stepCount}: <b>${intervention}</b> → reward <span class="${rewardClass}">${rewardSign}${data.reward.toFixed(3)}</span> | done: ${data.done} | stage: ${data.info.disease_stages?.[0] ?? '?'}`);
  if (data.done) {
    addLog(`🏁 Episode complete — reason: ${data.info.episode_done_reason}`);
    document.getElementById('action-card').style.display = 'none';
  }
}

function showObs(obs) {
  const field = obs.fields?.[0] ?? {};
  document.getElementById('obs-summary').innerHTML =
    `Timestep: <b>${obs.timestep}</b> | Lesion: <b>${(obs.lesion_coverage*100).toFixed(1)}%</b> | Pattern: <b>${obs.lesion_pattern}</b> | Stage: <b>${field.disease_stage ?? '?'}</b> | Humidity: <b>${(obs.humidity*100).toFixed(0)}%</b>`;
  document.getElementById('obs-json').textContent = JSON.stringify(obs, null, 2);
}

function addLog(msg) {
  const log = document.getElementById('log');
  const div = document.createElement('div');
  div.style.cssText = 'padding:4px 0;border-bottom:1px solid #e8f5e8;font-size:13px;';
  div.innerHTML = msg;
  log.prepend(div);
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return WEB_UI


@app.get("/web", response_class=HTMLResponse)
async def web():
    return WEB_UI


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/reset")
async def reset(task: str = "easy", seed: Optional[int] = None):
    obs = await env.reset(task=task, seed=seed)
    return obs.model_dump()


@app.post("/step")
async def step(action: RiceBlastAction):
    try:
        result = await env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if hasattr(result, "observation"):
        return {
            "observation": result.observation.model_dump(),
            "reward": result.reward,
            "done": result.done,
            "info": result.info,
        }
    obs, reward, done, info = result
    return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}


@app.get("/state")
async def state():
    try:
        obs = env.state()
    except RuntimeError:
        raise HTTPException(status_code=400, detail="No active episode. Call POST /reset first.")
    return obs.model_dump()


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
