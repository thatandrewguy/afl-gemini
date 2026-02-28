# 🏉 AFL Gemini Forecaster

Institutional-grade AFL match forecasting web app built with Streamlit.  
Served at `/afl-gemini` — e.g. `https://andrewsayer.me/afl-gemini`

---

## Features

- **HistGradientBoosting** bivariate regressors for margin & total score
- **ELO ratings** with dynamic home-ground advantage (neutralised for co-tenant matchups)
- **Continuity-corrected** win/draw/loss probabilities via Normal CDF
- Rolling EMA team stats, form, rest days, travel distance, venue familiarity
- Modern dark-green UI with probability bar visualisation

---

## Quick Start

### Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) package manager

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/afl-gemini.git
cd afl-gemini
```

### 2. Add the data file

Place `ReadyFor2026.csv` in the root of the repository (same directory as `app.py`).

> The CSV must contain AFL historical match data with columns including:
> `match.venueLocalStartTime`, `match.homeTeam.name`, `match.awayTeam.name`,
> `homeTeamScore.matchScore.totalScore`, `awayTeamScore.matchScore.totalScore`,
> `venue.name`, `venue.state`, `weather.tempInCelsius`, and per-team stat columns.

### 3. Create virtual environment & install dependencies

```bash
uv venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will be available at:  
`http://localhost:8501/afl-gemini`

---

## Deployment (reverse proxy)

To serve at `https://andrewsayer.me/afl-gemini`, configure your reverse proxy (nginx/caddy) to forward requests from `/afl-gemini` to the Streamlit server. The `baseUrlPath` is already set in [`.streamlit/config.toml`](.streamlit/config.toml).

Example nginx snippet:

```nginx
location /afl-gemini {
    proxy_pass http://127.0.0.1:8501;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_read_timeout 86400;
}
```

---

## Input Fields

| Field | Description |
|---|---|
| **Home Team** | Dropdown — select the home team |
| **Away Team** | Dropdown — select the away team |
| **Venue** | Dropdown of known venues, or enter manually |
| **Match Date** | Date picker |
| **Local Start Time** | Time picker (supports e.g. 20:05) |
| **Temperature (°C)** | Integer forecast temperature |

---

## Model Architecture

```
ReadyFor2026.csv
      │
      ▼
Feature Engineering
  ├─ ELO ratings (K=28, season regression)
  ├─ Rolling EMA stats (N=8, α=0.35)
  ├─ Form: win rate, percentage, avg margin (last 6)
  ├─ Rest days, short-rest flags
  ├─ Travel distance (state-based lookup)
  └─ Venue familiarity
      │
      ▼
HistGradientBoostingRegressor × 2
  ├─ Margin regressor  → pred_margin
  └─ Total regressor   → pred_total
      │
      ▼
Score reconstruction
  home = (total + margin) / 2
  away = (total − margin) / 2
      │
      ▼
Win probability (Normal CDF + continuity correction)
  P(home win) = 1 − Φ((0.5 − margin) / RMSE)
  P(draw)     = Φ((0.5 − margin) / RMSE) − Φ((−0.5 − margin) / RMSE)
  P(away win) = Φ((−0.5 − margin) / RMSE)
```

---

## File Structure

```
afl-gemini/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .streamlit/
│   └── config.toml         # Streamlit server config (baseUrlPath, theme)
└── ReadyFor2026.csv        # ← You provide this (not in repo)
```

---

## License

MIT
