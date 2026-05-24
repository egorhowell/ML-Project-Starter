# ML Project Starter

A working, production-grade scaffold for an end-to-end machine learning portfolio project. Clone it, run two commands, and you have a live dashboard predicting tomorrow's weather. Replace three files and it's predicting whatever **you** care about.

This is the starter template that goes with the **Get Hired With the Right Project** workbook. The workbook helps you pick a project worth building. This repo helps you ship it.

---

## What you get out of the box

| Concern | Tool | Why it's in here |
|---|---|---|
| Python version pinning | **pyenv** | Avoids "works on my machine" — everyone runs the same Python |
| Dependency management | **Poetry** | Lock-file based, sane defaults, what most ML teams use |
| Code style | **black** + **ruff** | Auto-formats and lints — no bikeshedding on style |
| Type checking | **mypy** | Catches a class of bugs before runtime |
| Testing | **pytest** | Industry standard, friendly syntax |
| CI on every push | **CircleCI** | Runs lint + type-check + tests automatically |
| Scheduled pipeline runs | **GitHub Actions** | Runs your model daily without you doing anything |
| Auto-deploy on merge to main | **GitHub Actions** | Push to main → dashboard updates within a minute |
| Database | **Supabase** | Postgres + auth + dashboard with a generous free tier |
| Web dashboard | **Streamlit** + **Plotly** | Build a usable UI in pure Python |
| Hosting | **Hostinger VPS** (optional) | Cheap, simple, your own URL |

You don't need to know every one of these on day one. They're already wired together — you can swap in your model and ignore the rest until you're ready.

---

## The pipeline shape

```
   ┌───────────┐     ┌────────────┐     ┌───────┐     ┌───────────┐     ┌──────────┐
   │ extractor │ ──► │ processor  │ ──► │ model │ ──► │  output   │ ──► │ database │
   └───────────┘     └────────────┘     └───────┘     └───────────┘     └──────────┘
        Pull              Clean            Fit +          Post-              Save to
        raw data          and prep        predict        process            Supabase
                                                                                │
                                                                                ▼
                                                                       ┌──────────────┐
                                                                       │  streamlit   │
                                                                       │  dashboard   │
                                                                       └──────────────┘
```

Each box maps to one file under `src/`. The whole thing is orchestrated by `src/main.py`, which is what runs once a day in production.

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

```bash
make run          # runs the pipeline once — predicts tomorrow's weather for 3 cities
make dashboard    # opens the dashboard at http://localhost:8501
```

You should see a dashboard with three cards (London, New York, Tokyo) showing predicted temperatures, plus a chart of actual vs predicted history.

The predictions are obviously fake — the placeholder model is a random walk. The point of the demo is to prove every piece of the system is plumbed together correctly. Now you replace the parts that matter.

> **No external account needed for first run.** If you haven't set up Supabase yet, predictions are saved to a local JSON file at `data/predictions.json`. The dashboard reads from the same place. When you set up Supabase later (see [Set up Supabase](#optional-set-up-supabase) below), the pipeline switches automatically.

---

## Making it your own

You're going to replace three files. Everything else can stay as-is until you want to change it.

### Step 1 — Pick your project

Work through the **Get Hired With the Right Project** workbook, the companion to this repo. It takes you through five things you're interested in, five questions for each, and helps you narrow down to one concrete project idea — something like *"predict my fantasy football lineup,"* *"forecast Bitcoin price,"* or *"score my dating-app matches."*

### Step 2 — Replace the data source (`src/extractor.py`)

Whatever your project predicts, you need a way to pull historical data for it. The default extractor pulls weather data from Open-Meteo. Replace `_fetch_open_meteo()` with a function that pulls **your** data.

The contract is simple. Your replacement should return a dict shaped like this:

```python
{
    "item_name_1": pd.DataFrame({"date": [...], "value": [...]}),
    "item_name_2": pd.DataFrame({"date": [...], "value": [...]}),
    ...
}
```

If your raw data has more than one "value" column (open/close/volume for stocks, say), pick the one you're actually predicting and put it in `value`. Stash the rest in extra columns if you need them later — the model and dashboard only need `date` and `value`.

Also update `settings.ITEMS_TO_PREDICT` to list the things you care about. Each entry is `(name, metadata_dict)` — the metadata dict is whatever your extractor needs (a lat/long, a ticker symbol, a player ID).

### Step 3 — Replace the model (`src/model.py`)

The default `Model` class is a random walk — it predicts tomorrow as today plus noise. Replace it with anything that has the same two methods:

```python
class Model:
    def fit(self, series: pd.Series) -> None: ...
    def predict_next(self) -> float: ...
```

Some starting points:

| Project type | Try |
|---|---|
| Time-series forecasting | Facebook Prophet, statsmodels, sktime |
| Tabular regression / classification | scikit-learn, XGBoost, LightGBM |
| Image / text data | PyTorch, HuggingFace Transformers |
| Quick baseline | sklearn `LinearRegression` or a moving average |

If you need to add packages: `poetry add <package-name>`.

### Step 4 — Make the dashboard yours (`src/streamlit_app.py`)

Update `DASHBOARD_TITLE` and `DASHBOARD_SUBTITLE` in `settings.py`. If your project benefits from different visualisations than the default summary-cards + line-chart, edit `_render_summary_cards` and `_render_history_chart`. Streamlit is just Python — you can put whatever you want there.

That's it. You now have your own end-to-end ML project. Push it to GitHub.

---

## Optional: set up Supabase

Required to deploy to production (the local JSON fallback is just for development).

1. Sign up at [supabase.com](https://supabase.com) — free tier is enough.
2. Create a new project. Pick a strong database password and save it.
3. Once the project is provisioned, go to **SQL Editor** in the left sidebar and paste:

   ```sql
   CREATE TABLE predictions (
       id BIGSERIAL PRIMARY KEY,
       created_at TIMESTAMPTZ DEFAULT NOW(),
       as_of_date DATE NOT NULL,
       item TEXT NOT NULL,
       prediction FLOAT8,
       last_value FLOAT8,
       predicted_change_pct FLOAT8
   );
   ```

   Click **Run**.
4. Go to **Project Settings → API**. Copy two values:
   - **Project URL** → goes into the `SUPABASE_URL` env var
   - **`service_role` secret** (the bottom one, NOT the anon key) → goes into `SUPABASE_KEY`

5. Locally, create a `.env` file in the repo root by copying `.env.example`:

   ```bash
   cp .env.example .env
   # then edit .env and paste your real values
   ```

6. Run `make run` again. You should see `Inserted N rows into Supabase table 'predictions'` in the logs, and the rows should appear in the **Table Editor** in Supabase.

---

## Optional: set up CircleCI

Adds automatic lint + type-check + test on every push and pull request. Catches mistakes before they hit main.

1. Sign up at [circleci.com](https://circleci.com) with your GitHub account.
2. From your CircleCI dashboard, click **Set Up Project** next to this repo.
3. Choose **"Fastest"** and pick the `.circleci/config.yml` already in the repo.
4. CircleCI now runs on every push. Open a pull request and you'll see the checks appear in GitHub.

To require checks to pass before merging to main: in GitHub, go to **Settings → Branches → Add branch protection rule** for `main` and tick *"Require status checks to pass before merging"*.

---

## Optional: deploy to a VPS

Gives your project a real public URL someone can visit. The cheapest decent option is a [Hostinger VPS](https://www.hostinger.com/vps-hosting). The KVM 2 plan is plenty.

You'll need:

1. A VPS with Ubuntu, accessible via SSH.
2. Add four secrets to your GitHub repo (**Settings → Secrets and variables → Actions**):
   - `VPS_HOST` — the IP address of your VPS
   - `VPS_USERNAME` — usually `root` on a fresh Hostinger VPS
   - `SSH_PRIVATE_KEY` — the private half of an SSH keypair whose public key you've added to `~/.ssh/authorized_keys` on the VPS
   - `VPS_REPO_PATH` — where this repo will live on the VPS (e.g. `/root/ml-project-starter`)
3. SSH into your VPS and:

   ```bash
   # Install pyenv, Poetry, and the same Python version as locally.
   curl https://pyenv.run | bash
   # ...follow the post-install instructions to update your shell rc...
   pyenv install 3.12.12
   pyenv global 3.12.12
   pip install poetry

   # Clone your repo to the path you set in VPS_REPO_PATH.
   git clone https://github.com/<your-user>/<your-repo>.git /root/ml-project-starter
   cd /root/ml-project-starter
   make install

   # Create .env on the VPS with your real Supabase values.
   cp .env.example .env
   nano .env  # paste values
   ```

4. Create a systemd service so Streamlit runs continuously and restarts on failure. Save the following to `/etc/systemd/system/streamlit-app.service`:

   ```ini
   [Unit]
   Description=Streamlit dashboard
   After=network.target

   [Service]
   Type=simple
   User=root
   WorkingDirectory=/root/ml-project-starter
   EnvironmentFile=/root/ml-project-starter/.env
   ExecStart=/root/.local/bin/poetry run streamlit run src/streamlit_app.py --server.port 80 --server.address 0.0.0.0
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

   Then:

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable streamlit-app
   sudo systemctl start streamlit-app
   ```

5. Visit `http://<your-vps-ip>` — your dashboard should be live.

From now on, every push to `main` will auto-deploy. (See `.github/workflows/deploy.yml`.)

---

## Common questions

**My model needs more than one feature column. Can I do that?**
Yes. Keep `value` as the thing you're predicting, but add extra columns to the DataFrame your extractor returns. Then in `model.py`, change `fit(series)` to accept the whole DataFrame and use whatever columns you want.

**I don't want to write the model per-item. Can I train one shared model?**
Yes — change `main.py` to call `model.fit(all_cleaned_data)` once instead of looping. You'll also want to add an `item` feature so the model knows which thing it's predicting for.

**The pipeline is slow.**
Profile with `python -m cProfile -s cumulative -m src.main`. Usually it's the model training. Smaller features or a simpler model will fix it.

**How do I add a new dependency?**
`poetry add <package-name>`. Commit the updated `pyproject.toml` and `poetry.lock`.

**How do I run `make` on Windows?**
Use WSL, or run the underlying commands directly (`poetry run python -m src.main`, `poetry run streamlit run src/streamlit_app.py`).

---

## The companion workbook

Picking the right project matters as much as building it well — a great scaffold plus a generic project still gives you a generic result.

The **Get Hired With the Right Project** workbook is the companion to this repo: a short fill-in exercise that helps you land on a project that's genuinely yours. Five things you're interested in, five questions for each, then you narrow down to the one that's exciting and just slightly beyond your current reach.

If you haven't done it yet, start there — then come back here to build.

---

## Getting unstuck

Once you have a project picked and the scaffold cloned, the hard part begins: choosing the right model, figuring out features, evaluating performance honestly, and being able to talk about all of it in a job interview.

That's where most people stall. If you want hands-on coaching to get all the way from "I have a scaffold" to "I landed the job," see [**Code to Careers**](https://smartertechies.io) — a 3-month 1:1 coaching program designed for data professionals who want to dramatically level up their next role.

---

## Credits

Built on top of [@egorhowell](https://github.com/egorhowell)'s [Prophet-Forecasting-For-Portfolio-Optimisation](https://github.com/egorhowell/Prophet-Forecasting-For-Portfolio-Optimisation) — the worked example from his [end-to-end ML project video](https://www.youtube.com/watch?v=2BvLAJwvfgo). If you want to see this exact scaffolding used on a real project (stock forecasting + Markowitz portfolio optimisation), that repo and that video are the gold standard.
