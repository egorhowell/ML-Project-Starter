# ML Project Starter

A production-grade scaffold for any end-to-end machine learning project. Clone it, drop in your data, and you have a working training pipeline, inference pipeline, and live dashboard — without having to wire anything together from scratch.

This is the starter template that goes with the **Get Hired With the Right Project** workbook. The workbook helps you pick a project worth building. This repo helps you ship it.

---

## What you get out of the box

| Concern | Tool | Why it's in here |
|---|---|---|
| Python version pinning | **pyenv** | Avoids "works on my machine" — everyone runs the same Python |
| Dependency management | **Poetry** | Lock-file based, sane defaults, what most ML teams use |
| ML & data science | **scikit-learn, XGBoost, LightGBM, scipy** | Core libraries for classical ML; swap any estimator in one line |
| Experiment tracking | **MLflow** | Log params, metrics, and artefacts across runs |
| Hyperparameter tuning | **optuna** | Plug in when you're ready to tune |
| Code style | **black** + **ruff** | Auto-formats and lints — no bikeshedding on style |
| Type checking | **mypy** | Catches a class of bugs before runtime |
| Testing | **pytest** | Industry standard, friendly syntax |
| CI on every push | **CircleCI** | Runs lint + type-check + tests automatically |
| Scheduled pipeline runs | **GitHub Actions** | Runs inference on a schedule without you doing anything |
| Database | **Supabase** | Postgres + auth + dashboard with a generous free tier |
| Web dashboard | **Streamlit** + **Plotly** | Build a usable UI in pure Python |
| Hosting | **Streamlit Community Cloud** | Free, one-click deploy, public URL — no server needed |

You don't need to know every one of these on day one. They're already wired together — you can swap in your model and ignore the rest until you're ready.

---

## The pipeline shape

Two modes, one command each:

```
  make train
  ┌───────────┐     ┌───────────┐     ┌───────┐
  │ extractor │ ──► │ processor │ ──► │ model │ ──► model saved to disk
  └───────────┘     └───────────┘     └───────┘
    load_training     preprocess()       fit(X, y)
    _data()           → (X, y)           → metrics

  make predict
  ┌───────────┐     ┌───────────┐     ┌───────┐     ┌────────┐     ┌──────────┐
  │ extractor │ ──► │ processor │ ──► │ model │ ──► │ output │ ──► │ database │
  └───────────┘     └───────────┘     └───────┘     └────────┘     └──────────┘
    load_inference    preprocess        predict(X)    format          save to
    _data()           _features()       → preds       predictions     Supabase
                      → X                                                 │
                                                                          ▼
                                                                 ┌──────────────┐
                                                                 │  streamlit   │
                                                                 │  dashboard   │
                                                                 └──────────────┘
```

Each box is one file under `src/`. `src/main.py` orchestrates both modes and is what runs in production.

---

## Quickstart (5 minutes)

### 1. Prerequisites

You'll need [pyenv](https://github.com/pyenv/pyenv#installation) and [Poetry](https://python-poetry.org/docs/#installation). On macOS:

```bash
brew install pyenv
pip install poetry
```

On Linux/Windows, follow the linked install guides.

### 2. Get the code and install dependencies

```bash
cd ml-project-starter
pyenv install 3.12.12        # if you don't have it already
pyenv local 3.12.12          # tells this folder which Python to use
make install-dev             # creates the virtual env + installs everything
```

### 3. Run the demo

The repo ships with a synthetic dataset so you can prove the pipeline works before touching your own data.

```bash
make train      # fits the model on data/train.csv, saves to data/model.joblib
make predict    # loads the model, runs inference on data/inference.csv, saves predictions
make dashboard  # opens the dashboard at http://localhost:8501
```

> **No external account needed.** If you haven't set up Supabase yet, predictions are written to `data/predictions.json` and the dashboard reads from there. Configure Supabase when you're ready to deploy (see [Set up Supabase](#optional-set-up-supabase) below).

---

## Making it your own

Four steps. Everything else can stay as-is until you want to change it.

### Step 1 — Prepare your data

The pipeline expects two CSV files (configured in `settings.py`):

| File | Contains |
|---|---|
| `data/train.csv` | Labeled data — features + target column |
| `data/inference.csv` | Unlabeled data — features only (no target required) |

Any column layout is fine. The only requirement is that your training CSV includes the column named in `settings.TARGET_COLUMN`.

### Step 2 — Configure `settings.py`

Open [`src/settings.py`](src/settings.py) and set:

```python
DATA_PATH = "data/train.csv"           # path to your labeled training data
INFERENCE_DATA_PATH = "data/inference.csv"  # path to your inference data
TARGET_COLUMN = "your_target_column"   # the column the model should predict
FEATURE_COLUMNS = []                   # [] = use every column except the target
```

That's the minimum. Everything downstream reads from these settings.

### Step 3 — Plug in your data source (`src/extractor.py`)

If your data lives somewhere other than a CSV (a database, an API, S3, …), replace `_load_csv()` in [`src/extractor.py`](src/extractor.py) with your own fetch logic. The return type just needs to be a `pd.DataFrame` — the rest of the pipeline adapts automatically.

```python
# Replace this:
def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# With your own source, e.g.:
def _load_from_db(query: str) -> pd.DataFrame:
    engine = sqlalchemy.create_engine(os.environ["DB_URL"])
    return pd.read_sql(query, engine)
```

### Step 4 — Swap the model (`src/model.py`)

The default is a scikit-learn `LinearRegression`. To change the algorithm, edit one line in `_build_estimator()`:

```python
def _build_estimator():
    return LinearRegression()          # default

    # ── Regression ───────────────────────────────────────────────────
    # return Ridge(alpha=1.0)
    # return RandomForestRegressor(n_estimators=100, random_state=42)
    # return XGBRegressor(n_estimators=100, learning_rate=0.1)
    # return LGBMRegressor(n_estimators=100, learning_rate=0.1)

    # ── Classification ───────────────────────────────────────────────
    # return LogisticRegression(max_iter=1000)
    # return RandomForestClassifier(n_estimators=100, random_state=42)
    # return XGBClassifier(n_estimators=100, learning_rate=0.1)
```

The model interface (`fit(X, y)` / `predict(X)`) stays the same regardless of which estimator you use, so the rest of the pipeline just works.

Numeric targets → regression metrics (MAE, RMSE, R²) are logged automatically.
String/categorical targets → classification metrics (accuracy, F1) are logged automatically.

If you need new packages: `poetry add <package-name>`.

---

## Optional: experiment tracking with MLflow

Switch on experiment tracking by replacing `Model()` with `train_with_tracking()` in `src/main.py`:

```python
# Instead of:
model = Model()
metrics = model.fit(X, y)

# Use:
from src.model import train_with_tracking
model, metrics = train_with_tracking(X, y)
```

Runs are logged to `mlruns/` by default. Open the MLflow UI with:

```bash
poetry run mlflow ui
```

To use a remote tracking server, set `MLFLOW_TRACKING_URI` in `settings.py`.

---

## Optional: hyperparameter tuning with Optuna

Optuna is installed and ready to use. Create a study in your training script:

```python
import optuna
from sklearn.ensemble import RandomForestRegressor

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    return mean_absolute_error(y_val, model.predict(X_val))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
```

---

## Optional: set up Supabase

Required for production deployment (the local JSON fallback is for development only).

1. Sign up at [supabase.com](https://supabase.com) — the free tier is enough.
2. Create a new project.
3. Go to **SQL Editor** and create your predictions table. The minimal schema:

   ```sql
   CREATE TABLE predictions (
       id          BIGSERIAL PRIMARY KEY,
       prediction  FLOAT8,
       predicted_at TIMESTAMPTZ
   );
   ```

   Add any extra columns that match the fields your `output.format_predictions()` produces (feature values, IDs, labels, etc.).

4. Go to **Project Settings → API** and copy:
   - **Project URL** → `SUPABASE_URL`
   - **`service_role` secret** (the bottom key, NOT the anon key) → `SUPABASE_KEY`

5. Create a `.env` file at the repo root:

   ```bash
   cp .env.example .env
   # edit .env and paste your values
   ```

6. Run `make predict`. You should see `Inserted N rows into Supabase table 'predictions'` in the logs.

---

## Optional: set up CircleCI

Adds automatic lint + type-check + test on every push and pull request.

1. Sign up at [circleci.com](https://circleci.com) with your GitHub account.
2. From your dashboard, click **Set Up Project** next to this repo.
3. Choose **"Fastest"** and select the `.circleci/config.yml` already in the repo.

To require checks to pass before merging: **GitHub → Settings → Branches → Add rule** for `main`, tick *"Require status checks to pass before merging"*.

---

## Optional: deploy to Streamlit Community Cloud

Gives your project a free public URL. Streamlit watches your GitHub repo and re-deploys automatically on every push to `main`.

1. Push your repo to GitHub (public or private — both work).
2. Sign in at [share.streamlit.io](https://share.streamlit.io).
3. Click **New app** and fill in:
   - **Repository** — your repo
   - **Branch** — `main`
   - **Main file path** — `src/streamlit_app.py`
4. Open **Advanced settings → Secrets** and paste your Supabase credentials:
   ```toml
   SUPABASE_URL = "https://your-project-ref.supabase.co"
   SUPABASE_KEY = "your-service-role-key"
   ```
5. Click **Deploy**. You'll get a `*.streamlit.app` URL to share.

---

## Common questions

**What kind of ML tasks does this support?**
Any scikit-learn-compatible task: regression, binary classification, multiclass classification. Swap the estimator in `_build_estimator()` and the metrics update automatically. For clustering or anomaly detection, call the estimator directly rather than through the `Model` wrapper.

**My data has both numeric and categorical columns. Do I need to encode them manually?**
No. The `Model` uses a `ColumnTransformer` that auto-detects column types: `StandardScaler` for numerics, `OneHotEncoder` for strings. You don't need to preprocess column types yourself.

**Can I use PyTorch or TensorFlow?**
Yes. Replace the contents of `model.py` with your own training loop. Keep the `fit(X, y)` / `predict(X)` / `save()` / `load()` signatures and `main.py` won't need to change.

**I want to tune hyperparameters. Where do I start?**
See the [Optuna section](#optional-hyperparameter-tuning-with-optuna) above. Optuna is already installed.

**I want to log experiments. Where do I start?**
See the [MLflow section](#optional-experiment-tracking-with-mlflow) above. MLflow is already installed.

**My features need domain-specific engineering before training.**
Add it to `processor.py`. There are two helper stubs already in there — `_add_interaction_terms()` and `_add_date_features()` — showing the pattern. Uncomment and adapt whichever you need.

**How do I add a new dependency?**
`poetry add <package-name>`. Commit the updated `pyproject.toml` and `poetry.lock`.

**How do I run `make` on Windows?**
Use WSL, or run the commands directly: `poetry run python -m src.main train`, `poetry run python -m src.main predict`, `poetry run streamlit run src/streamlit_app.py`.

**The pipeline is slow.**
Profile with `python -m cProfile -s cumulative -m src.main predict`. The bottleneck is usually model training or data loading. Smaller feature sets, a simpler estimator, or caching your data fetch will usually fix it.

---

## The companion workbook

Picking the right project matters as much as building it well — a great scaffold plus a generic project still gives you a generic result.

The **Get Hired With the Right Project** workbook is the companion to this repo: a short fill-in exercise that helps you land on a project that's genuinely yours. Five things you're interested in, five questions for each, then you narrow down to the one that's exciting and just slightly beyond your current reach.

If you haven't done it yet, start there — then come back here to build.

---

## Getting unstuck

Once you have a project picked and the scaffold cloned, the hard part begins: choosing the right model, figuring out features, evaluating performance honestly, and being able to talk about all of it in a job interview.

That's where most people stall. If you want hands-on coaching to get all the way from "I have a scaffold" to "I landed the job," see [**Code to Careers**](https://coaching.egorhowell.com/) — a 3-6 month programme designed for data professionals who want to dramatically level up their next role.

