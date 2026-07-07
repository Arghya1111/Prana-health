# PRĀNA AI Triage System

Multi-modal clinical intelligence platform combining ECG, EEG, PPG, SpO₂, thermal imaging, and fusion-based patient triage.

## Quick start (local)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard.py
```

**Entry point:** `dashboard.py` (root of the repository)

## Deploy to Render

The repository includes a [Render Blueprint](https://render.com/docs/blueprint-spec) (`render.yaml`) for one-click deployment.

### Architecture

| Component | Path | Notes |
|-----------|------|-------|
| Dashboard | `dashboard.py` | Streamlit web UI |
| Data directory | `PRANA_DATA_DIR` (default: `database/`) | SQLite + ECG pickle |
| Model weights | `modules/*/model/*.pth`, `database/ecg_model.pkl` | Must be present in deploy artifact |
| Config | `.streamlit/config.toml` | Production Streamlit settings |

### One-click deploy

1. Push this repository to GitHub.
2. In [Render Dashboard](https://dashboard.render.com/) → **New** → **Blueprint**.
3. Connect the repo — Render reads `render.yaml` automatically.
4. Attach model weight files to the repository or upload them to the persistent disk after first deploy (see below).

### Manual web service setup

| Setting | Value |
|---------|-------|
| **Runtime** | Python 3.11 |
| **Build command** | `pip install --upgrade pip && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt` |
| **Start command** | `streamlit run dashboard.py --server.port=$PORT --server.address=0.0.0.0` |
| **Health check path** | `/` |

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `PRANA_DATA_DIR` | Recommended on Render | Writable path for SQLite (e.g. `/var/data` with a persistent disk) |
| `NATS_URL` | Optional | NATS server URL if running the messaging pipeline |
| `NATS_HOST` / `NATS_PORT` | Optional | Fallback NATS health-check target |

### Persistent storage (ephemeral filesystem)

Render web services use an **ephemeral filesystem** — files written outside a persistent disk are lost on redeploy.

`render.yaml` mounts a 1 GB disk at `/var/data` and sets `PRANA_DATA_DIR=/var/data` so that:

- `prana_records.db` survives redeploys
- `ecg_model.pkl` can be stored on the disk

PyTorch checkpoints under `modules/*/model/` are read from the **deploy slug** (commit them, use Git LFS, or copy to the disk after deploy).

### Model weights

The dashboard checks for these files at startup:

| Modality | Expected path |
|----------|---------------|
| ECG | `database/ecg_model.pkl` (or `$PRANA_DATA_DIR/ecg_model.pkl`) |
| EEG | `modules/eeg/model/eeg_model.pth` |
| PPG | `modules/ppg/model/ppg_model.pth` |
| SpO₂ | `modules/spo2/model/spo2_model.pth` |
| Thermal | `modules/thermal/model/thermal_model.pth` |

If weights are missing, the dashboard **still loads** in demo mode and shows a warning banner. Full load verification is available under **Settings → Model Weights**.

Train locally, then either:

- Commit weights with [Git LFS](https://git-lfs.github.com/), or
- Upload to the Render persistent disk via shell

### Verify deployment locally

```bash
python scripts/verify_deployment.py
```

### Production Streamlit settings

Configured in `.streamlit/config.toml`:

- `headless = true`
- `enableCORS = false`
- `enableXsrfProtection = false`

Render overrides the port via the start command (`--server.port=$PORT`).

---

## Deployment checklist

Use this before marking a Render deploy as complete:

- [ ] **Repository** pushed to GitHub with `render.yaml`, `runtime.txt`, and `requirements.txt`
- [ ] **Python 3.11** selected (via `runtime.txt` or Render env)
- [ ] **Build succeeds** — CPU PyTorch wheels install without CUDA errors
- [ ] **Start command** uses `$PORT` and `--server.address=0.0.0.0`
- [ ] **Persistent disk** attached and `PRANA_DATA_DIR=/var/data` set (for SQLite durability)
- [ ] **Model weights** uploaded or committed for ECG, EEG, PPG, SpO₂, Thermal
- [ ] **Dashboard loads** at the Render URL without import errors
- [ ] **Settings page** → Model Weights shows all modalities OK (or expected demo-mode warnings)
- [ ] **SQLite** creates `prana_records.db` on first request (empty queue is normal)
- [ ] **No hardcoded Windows paths** — all paths resolve via `app.paths`
- [ ] **NATS** marked offline unless a NATS service is provisioned (expected for dashboard-only deploy)

---

## Project layout

```
dashboard.py              ← Streamlit entry point (Render start target)
app/
  paths.py                ← Central path + env configuration
  model_health.py         ← Model file / load verification
  clinical_dashboard/     ← Dashboard package namespace
  api.py                    ← Optional FastAPI service
database/
  prana_database.py       ← SQLite schema + ECG persistence
modules/                  ← Per-modality training & inference
fusion/                   ← Multi-modal fusion engine
```

## License

See repository license file.
