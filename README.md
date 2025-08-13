# House Price Prediction — Flask · scikit‑learn/XGBoost · Docker · Heroku

This is my first end‑to‑end ML project. I trained a model to predict house prices (Ames dataset), wrapped it in a Flask app, containerized it with Docker, and set up automatic deploys to Heroku using GitHub Actions. I’m keeping this README short and practical so anyone can run the same setup I’m using.

---

## What this repo is

- A Flask app that serves a trained house‑price predictor.
- Everything runs inside a Docker container (Python 3.11), so it’s reproducible.
- GitHub Actions builds the image and releases it to Heroku whenever I push to `main`.

---

## How to run it (exact commands I use)

### Option A — Docker (recommended)

```bash
# from the repo root
# 1) build the image using the Dockerfile in this repo
docker build -t houseprice-api .

# 2) run the container locally
#    Heroku sets $PORT in prod; locally I pass 5000
docker run -p 5000:5000 -e PORT=5000 houseprice-api
```

Now open `http://localhost:5000`.

---

## Deploys (GitHub Actions → Heroku)
Live app: https://housepredectionpipeline-0a04d51ed8a0.herokuapp.com
- On every push to `main`, the workflow at `.github/workflows/deploy-heroku.yml` runs.
- It installs the Heroku CLI on the runner, switches the app to the **container** stack if needed, logs in to the container registry, builds the Docker image, pushes it to `registry.heroku.com/<app>/web`, and releases it.
- You need two repo **secrets**:
  - `HEROKU_API_KEY` – from Heroku Account Settings
  - `HEROKU_APP_NAME` – the exact Heroku app name

That’s it—no Procfile needed for container deploys. The `CMD` in the Dockerfile starts Gunicorn with `app:app`.

---

## The model (what I trained and how I combine predictions)

I didn’t rely on just one algorithm. I trained two separate regressors and blended them:

- **ElasticNet** (linear model with L1/L2 regularization). Good at handling multicollinearity and gives stable, interpretable effects.
- **XGBoost** (gradient‑boosted trees). Good at capturing non‑linear interactions and handling mixed feature types.

I tuned each model separately (regularization for ElasticNet; learning rate, depth, estimators, etc. for XGBoost) using cross‑validation. After that, I **combined their predictions (averaged them) **. Blending reduces variance and usually beats either model alone.

**Artifacts**

- I save both models using pickle
- At app startup, both models are loaded, a feature preprocessing step is applied in the same way as during training, and the final prediction is computed.
---

## Why I’m using Docker + Actions

- Docker gives me the same environment locally, in CI, and on Heroku.
- GitHub Actions acts as my CI/CD: push → build container in a clean VM → release to Heroku automatically.
- This keeps deploys predictable and removes “works on my machine” surprises.

---

## Notes I picked up while shipping this

- With Docker, `.python-version` and Procfile aren’t required by Heroku—the Dockerfile controls the runtime and start command.
- Heroku’s **container** stack is required for `heroku container:release` (the workflow sets it if needed).
- Keep the repo focused on what’s needed at inference time for faster builds.

If you spot anything off or want more details (training, features, or exact CV settings), open an issue or ping me and I’ll add it.

