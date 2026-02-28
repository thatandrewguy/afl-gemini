import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import warnings
import os

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Page config – must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AFL Gemini Forecaster",
    page_icon="🏉",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Custom CSS – modern dark-green AFL aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* ── Background ── */
  .stApp {
    background: linear-gradient(135deg, #0a0f0a 0%, #0d1a0d 50%, #0a1020 100%);
    min-height: 100vh;
  }

  /* ── Hero banner ── */
  .hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
  }
  .hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00e676, #69f0ae, #40c4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
  }
  .hero p {
    color: #90a4ae;
    font-size: 1.05rem;
    font-weight: 400;
  }

  /* ── Card wrapper ── */
  .card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,230,118,0.15);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(8px);
  }
  .card-title {
    color: #00e676;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1rem;
  }

  /* ── Streamlit widget overrides ── */
  .stSelectbox > div > div,
  .stNumberInput > div > div > input,
  .stTextInput > div > div > input,
  .stDateInput > div > div > input,
  .stTimeInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(0,230,118,0.25) !important;
    border-radius: 10px !important;
    color: #e0e0e0 !important;
  }
  .stSelectbox > div > div:hover,
  .stNumberInput > div > div > input:focus,
  .stTextInput > div > div > input:focus {
    border-color: #00e676 !important;
    box-shadow: 0 0 0 2px rgba(0,230,118,0.15) !important;
  }
  label, .stSelectbox label, .stNumberInput label {
    color: #b0bec5 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
  }

  /* ── Predict button ── */
  .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00c853, #00e676);
    color: #000 !important;
    font-weight: 700;
    font-size: 1.05rem;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    cursor: pointer;
    transition: all 0.2s ease;
    letter-spacing: 0.04em;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #00e676, #69f0ae);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(0,230,118,0.35);
  }

  /* ── Result cards ── */
  .result-winner {
    background: linear-gradient(135deg, rgba(0,200,83,0.15), rgba(0,230,118,0.08));
    border: 1px solid rgba(0,230,118,0.4);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    margin-bottom: 1rem;
  }
  .result-winner .team-name {
    font-size: 2rem;
    font-weight: 800;
    color: #00e676;
  }
  .result-winner .confidence {
    font-size: 1.1rem;
    color: #b2dfdb;
    margin-top: 0.3rem;
  }

  .score-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(64,196,255,0.2);
    border-radius: 14px;
    padding: 1.4rem;
    text-align: center;
  }
  .score-box .score-label {
    font-size: 0.78rem;
    color: #78909c;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
  }
  .score-box .score-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #40c4ff;
  }
  .score-box .score-sub {
    font-size: 0.85rem;
    color: #90a4ae;
    margin-top: 0.2rem;
  }

  .stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.55rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    color: #cfd8dc;
    font-size: 0.92rem;
  }
  .stat-row:last-child { border-bottom: none; }
  .stat-row .stat-label { color: #78909c; }
  .stat-row .stat-val { font-weight: 600; color: #e0e0e0; }

  /* ── Warning / info ── */
  .stAlert { border-radius: 10px !important; }

  /* ── Divider ── */
  hr { border-color: rgba(0,230,118,0.1) !important; }

  /* ── Spinner ── */
  .stSpinner > div { border-top-color: #00e676 !important; }

  /* ── Hide Streamlit branding ── */
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants (extracted exactly from source)
# ─────────────────────────────────────────────
KEY_STATS = ['kicksTotal', 'marksTotal', 'handballsTotal', 'disposalsTotal', 'inside50sTotal', 'clearancesTotal']

ELO_K = 28
HOME_BOOST_MAX = 55
ROLLING_N = 8
EMA_ALPHA = 0.35

STATE_DIST = {
    ('VIC', 'VIC'): 0, ('NSW', 'NSW'): 0, ('QLD', 'QLD'): 0, ('SA', 'SA'): 0, ('WA', 'WA'): 0, ('TAS', 'TAS'): 0,
    ('VIC', 'NSW'): 700, ('VIC', 'SA'): 650, ('VIC', 'TAS'): 400, ('VIC', 'QLD'): 1400, ('VIC', 'WA'): 2700,
    ('NSW', 'SA'): 1100, ('NSW', 'QLD'): 750, ('NSW', 'WA'): 3300, ('NSW', 'TAS'): 1000,
    ('SA', 'QLD'): 1600, ('SA', 'WA'): 2100, ('SA', 'TAS'): 1000,
    ('QLD', 'WA'): 3600, ('QLD', 'TAS'): 1800, ('WA', 'TAS'): 3000
}

TEAM_STATES = {
    'Collingwood': 'VIC', 'Essendon': 'VIC', 'Carlton': 'VIC', 'Geelong Cats': 'VIC',
    'Hawthorn': 'VIC', 'North Melbourne': 'VIC', 'Richmond': 'VIC', 'St Kilda': 'VIC',
    'Western Bulldogs': 'VIC', 'Melbourne': 'VIC', 'Sydney Swans': 'NSW', 'GWS GIANTS': 'NSW',
    'Brisbane Lions': 'QLD', 'Gold Coast SUNS': 'QLD', 'Adelaide Crows': 'SA', 'Port Adelaide': 'SA',
    'West Coast Eagles': 'WA', 'Fremantle': 'WA',
}

TEAM_ALIASES = {
    'collingwood': 'Collingwood', 'pies': 'Collingwood',
    'essendon': 'Essendon', 'bombers': 'Essendon',
    'carlton': 'Carlton', 'blues': 'Carlton',
    'geelong': 'Geelong Cats', 'cats': 'Geelong Cats', 'geelong cats': 'Geelong Cats',
    'hawthorn': 'Hawthorn', 'hawks': 'Hawthorn',
    'north melbourne': 'North Melbourne', 'kangaroos': 'North Melbourne', 'roos': 'North Melbourne',
    'richmond': 'Richmond', 'tigers': 'Richmond',
    'st kilda': 'St Kilda', 'saints': 'St Kilda',
    'western bulldogs': 'Western Bulldogs', 'bulldogs': 'Western Bulldogs', 'dogs': 'Western Bulldogs',
    'melbourne': 'Melbourne', 'demons': 'Melbourne', 'dees': 'Melbourne',
    'sydney': 'Sydney Swans', 'swans': 'Sydney Swans', 'sydney swans': 'Sydney Swans',
    'gws': 'GWS GIANTS', 'giants': 'GWS GIANTS', 'gws giants': 'GWS GIANTS',
    'brisbane': 'Brisbane Lions', 'lions': 'Brisbane Lions', 'brisbane lions': 'Brisbane Lions',
    'gold coast': 'Gold Coast SUNS', 'suns': 'Gold Coast SUNS', 'gold coast suns': 'Gold Coast SUNS',
    'adelaide': 'Adelaide Crows', 'crows': 'Adelaide Crows', 'adelaide crows': 'Adelaide Crows',
    'port adelaide': 'Port Adelaide', 'power': 'Port Adelaide', 'port': 'Port Adelaide',
    'west coast': 'West Coast Eagles', 'eagles': 'West Coast Eagles', 'west coast eagles': 'West Coast Eagles',
    'fremantle': 'Fremantle', 'dockers': 'Fremantle', 'freo': 'Fremantle'
}

# Canonical team list for dropdowns (sorted display name → alias key)
TEAM_DISPLAY = sorted(set(TEAM_ALIASES.values()))

# ─────────────────────────────────────────────
# Model loading (cached so it only runs once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    csv_path = os.path.join(os.path.dirname(__file__), 'ReadyFor2026.csv')
    if not os.path.exists(csv_path):
        return None, "ReadyFor2026.csv not found. Please place it in the same directory as app.py."

    df = pd.read_csv(csv_path)

    df['matchDate'] = pd.to_datetime(df['match.venueLocalStartTime'], errors='coerce')
    df = df.dropna(subset=['matchDate']).sort_values('matchDate').reset_index(drop=True)

    df['home_total'] = pd.to_numeric(df['homeTeamScore.matchScore.totalScore'], errors='coerce')
    df['away_total'] = pd.to_numeric(df['awayTeamScore.matchScore.totalScore'], errors='coerce')
    df = df.dropna(subset=['home_total', 'away_total'])

    df['home_win'] = (df['home_total'] > df['away_total']).astype(int)
    df['margin'] = df['home_total'] - df['away_total']
    df['match_total'] = df['home_total'] + df['away_total']

    teams = sorted(set(df['match.homeTeam.name'].dropna()) | set(df['match.awayTeam.name'].dropna()))
    team_overall_history = {t: [] for t in teams}
    team_results_history = {t: [] for t in teams}
    team_elo = {t: 1500.0 for t in teams}
    team_last_played = {}
    team_venue_games = {t: {} for t in teams}

    avg_temp = df['weather.tempInCelsius'].mean()
    if pd.isna(avg_temp):
        avg_temp = 18.0

    def get_travel_km(state1, state2):
        if not state1 or not state2:
            return 0
        return STATE_DIST.get((state1, state2), STATE_DIST.get((state2, state1), 0))

    def get_rolling(team, n=ROLLING_N):
        history = team_overall_history[team]
        if not history:
            return {s: 0.0 for s in KEY_STATS}
        recent = history[-n:]
        result = {}
        for stat in KEY_STATS:
            vals = [g.get(stat, 0) for g in recent]
            ema_val = vals[0]
            for v in vals[1:]:
                ema_val = EMA_ALPHA * v + (1 - EMA_ALPHA) * ema_val
            result[stat] = ema_val
        return result

    def get_form(team, n=6):
        history = team_results_history[team]
        if not history:
            return 0.5, 1.0, 85.0, 85.0, 0.0
        recent = history[-n:]
        wins = sum(w for _, _, w in recent)
        sf = np.mean([s for s, _, _ in recent])
        sa = np.mean([s for _, s, _ in recent])
        return wins / len(recent), sf / (sa + 1e-6), sf, sa, sf - sa

    feature_rows, labels_margin, labels_total, labels_class = [], [], [], []
    last_year = None

    for _, row in df.iterrows():
        ht, at = row['match.homeTeam.name'], row['match.awayTeam.name']
        if pd.isna(ht) or pd.isna(at):
            continue

        mdt = row['matchDate']
        if last_year and mdt.year != last_year:
            for t in team_elo:
                team_elo[t] = 0.65 * team_elo[t] + 0.35 * 1500.0
        last_year = mdt.year

        h_rest = min(max((mdt - team_last_played.get(ht, mdt - pd.Timedelta(days=8))).days, 0), 30)
        a_rest = min(max((mdt - team_last_played.get(at, mdt - pd.Timedelta(days=8))).days, 0), 30)

        vstate = row.get('venue.state', 'VIC')
        h_state = TEAM_STATES.get(ht, 'VIC')
        a_state = TEAM_STATES.get(at, 'VIC')
        h_travel_km = get_travel_km(h_state, vstate)
        a_travel_km = get_travel_km(a_state, vstate)

        h_roll, a_roll = get_rolling(ht), get_rolling(at)
        h_wr, h_pct, h_sf, h_sa, h_mg = get_form(ht)
        a_wr, a_pct, a_sf, a_sa, a_mg = get_form(at)

        dynamic_home_boost = 0 if h_state == a_state else HOME_BOOST_MAX
        h_elo_adj = team_elo[ht] + dynamic_home_boost
        elo_exp = 1.0 / (1.0 + 10 ** ((team_elo[at] - h_elo_adj) / 400))

        h_games = len(team_venue_games[ht].get(row.get('venue.name', ''), []))
        a_games = len(team_venue_games[at].get(row.get('venue.name', ''), []))
        venue_fam = (h_games - a_games) / (h_games + a_games + 1)

        feat = {
            'elo_diff': h_elo_adj - team_elo[at],
            'elo_expected': elo_exp,
            'home_win_rate': h_wr, 'away_win_rate': a_wr,
            'percentage_diff': h_pct - a_pct,
            'margin_diff': h_mg - a_mg,
            'rest_diff': h_rest - a_rest,
            'h_short_rest': int(h_rest < 7), 'a_short_rest': int(a_rest < 7),
            'travel_km_diff': a_travel_km - h_travel_km,
            'venue_familiarity': venue_fam,
            'temp': float(row.get('weather.tempInCelsius', avg_temp))
        }

        for stat in KEY_STATS:
            feat[f'diff_{stat}'] = h_roll.get(stat, 0) - a_roll.get(stat, 0)
            feat[f'total_{stat}'] = h_roll.get(stat, 0) + a_roll.get(stat, 0)

        feature_rows.append(feat)
        labels_margin.append(row['margin'])
        labels_total.append(row['match_total'])
        labels_class.append(row['home_win'])

        h_stats = {s: row.get(f"homeTeam.{s}", 0) for s in KEY_STATS}
        a_stats = {s: row.get(f"awayTeam.{s}", 0) for s in KEY_STATS}
        team_overall_history[ht].append(h_stats)
        team_overall_history[at].append(a_stats)

        h_won = row['home_win']
        team_results_history[ht].append((row['home_total'], row['away_total'], h_won))
        team_results_history[at].append((row['away_total'], row['home_total'], not h_won))

        team_venue_games[ht].setdefault(row.get('venue.name', ''), []).append(1)
        team_venue_games[at].setdefault(row.get('venue.name', ''), []).append(1)

        team_last_played[ht] = team_last_played[at] = mdt

        margin_mult = np.log(abs(row['margin']) + 1) / np.log(80)
        k = ELO_K * (0.6 + 0.4 * margin_mult)
        actual = 1.0 if h_won else 0.0
        team_elo[ht] += k * (actual - elo_exp)
        team_elo[at] += k * ((1 - actual) - (1 - elo_exp))

    X = pd.DataFrame(feature_rows).fillna(0)
    MIN_HISTORY = 60
    X_model = X.iloc[MIN_HISTORY:]
    y_marg_model = np.array(labels_margin)[MIN_HISTORY:]
    y_tot_model = np.array(labels_total)[MIN_HISTORY:]
    y_class_model = np.array(labels_class)[MIN_HISTORY:]

    split_idx = int(len(X_model) * 0.8)
    X_train = X_model.iloc[:split_idx]
    X_test = X_model.iloc[split_idx:]
    y_marg_train = y_marg_model[:split_idx]
    y_marg_test = y_marg_model[split_idx:]
    y_tot_train = y_tot_model[:split_idx]
    y_class_test = y_class_model[split_idx:]

    margin_reg = HistGradientBoostingRegressor(
        loss='squared_error', max_iter=350, max_depth=4,
        learning_rate=0.03, min_samples_leaf=40,
        l2_regularization=3.0, random_state=42
    )
    margin_reg.fit(X_train, y_marg_train)

    total_reg = HistGradientBoostingRegressor(
        loss='squared_error', max_iter=350, max_depth=4,
        learning_rate=0.03, min_samples_leaf=40,
        l2_regularization=3.0, random_state=42
    )
    total_reg.fit(X_train, y_tot_train)

    preds_margin = margin_reg.predict(X_test)
    preds_class = (preds_margin > 0).astype(int)
    model_rmse = np.sqrt(mean_squared_error(y_marg_test, preds_margin))

    # Retrain on full dataset
    margin_reg.fit(X_model, y_marg_model)
    total_reg.fit(X_model, y_tot_model)

    venue_map = df.groupby('venue.name').agg({'venue.state': 'first'}).to_dict('index')

    val_accuracy = accuracy_score(y_class_test, preds_class) * 100
    val_mae = mean_absolute_error(y_marg_test, preds_margin)

    state = {
        'margin_reg': margin_reg,
        'total_reg': total_reg,
        'team_elo': team_elo,
        'team_last_played': team_last_played,
        'team_overall_history': team_overall_history,
        'team_results_history': team_results_history,
        'team_venue_games': team_venue_games,
        'venue_map': venue_map,
        'X_model': X_model,
        'avg_temp': avg_temp,
        'model_rmse': model_rmse,
        'val_accuracy': val_accuracy,
        'val_mae': val_mae,
    }
    return state, None


def resolve_team(name):
    return TEAM_ALIASES.get(name.lower().strip())


def predict_match(state, home_input, away_input, venue_name, temp, start_time_str):
    home_team = resolve_team(home_input)
    away_team = resolve_team(away_input)

    if not home_team or home_team not in state['team_overall_history']:
        return f"Home Team '{home_input}' not found or unrecognized."
    if not away_team or away_team not in state['team_overall_history']:
        return f"Away Team '{away_input}' not found or unrecognized."

    team_last_played = state['team_last_played']
    team_elo = state['team_elo']
    team_overall_history = state['team_overall_history']
    team_results_history = state['team_results_history']
    team_venue_games = state['team_venue_games']
    venue_map = state['venue_map']
    avg_temp = state['avg_temp']
    margin_reg = state['margin_reg']
    total_reg = state['total_reg']
    X_model = state['X_model']
    model_rmse = state['model_rmse']

    start_dt = pd.to_datetime(start_time_str, errors='coerce')
    if pd.isna(start_dt):
        start_dt = team_last_played.get(home_team, pd.Timestamp.now()) + pd.Timedelta(days=7)

    h_rest = min(max((start_dt - team_last_played.get(home_team, start_dt - pd.Timedelta(days=7))).days, 0), 30)
    a_rest = min(max((start_dt - team_last_played.get(away_team, start_dt - pd.Timedelta(days=7))).days, 0), 30)

    vstate = venue_map.get(venue_name, {}).get('venue.state', 'VIC')
    h_state = TEAM_STATES.get(home_team, 'VIC')
    a_state = TEAM_STATES.get(away_team, 'VIC')

    def get_travel_km(state1, state2):
        if not state1 or not state2:
            return 0
        return STATE_DIST.get((state1, state2), STATE_DIST.get((state2, state1), 0))

    h_travel_km = get_travel_km(h_state, vstate)
    a_travel_km = get_travel_km(a_state, vstate)

    def get_rolling_local(team, n=ROLLING_N):
        history = team_overall_history[team]
        if not history:
            return {s: 0.0 for s in KEY_STATS}
        recent = history[-n:]
        result = {}
        for stat in KEY_STATS:
            vals = [g.get(stat, 0) for g in recent]
            ema_val = vals[0]
            for v in vals[1:]:
                ema_val = EMA_ALPHA * v + (1 - EMA_ALPHA) * ema_val
            result[stat] = ema_val
        return result

    def get_form_local(team, n=6):
        history = team_results_history[team]
        if not history:
            return 0.5, 1.0, 85.0, 85.0, 0.0
        recent = history[-n:]
        wins = sum(w for _, _, w in recent)
        sf = np.mean([s for s, _, _ in recent])
        sa = np.mean([s for _, s, _ in recent])
        return wins / len(recent), sf / (sa + 1e-6), sf, sa, sf - sa

    h_roll, a_roll = get_rolling_local(home_team), get_rolling_local(away_team)
    h_wr, h_pct, h_sf, h_sa, h_mg = get_form_local(home_team)
    a_wr, a_pct, a_sf, a_sa, a_mg = get_form_local(away_team)

    dynamic_home_boost = 0 if h_state == a_state else HOME_BOOST_MAX
    h_elo_adj = team_elo[home_team] + dynamic_home_boost
    elo_exp = 1.0 / (1.0 + 10 ** ((team_elo[away_team] - h_elo_adj) / 400))

    h_games = len(team_venue_games[home_team].get(venue_name, []))
    a_games = len(team_venue_games[away_team].get(venue_name, []))
    venue_fam = (h_games - a_games) / (h_games + a_games + 1)

    feat = {
        'elo_diff': h_elo_adj - team_elo[away_team],
        'elo_expected': elo_exp,
        'home_win_rate': h_wr, 'away_win_rate': a_wr,
        'percentage_diff': h_pct - a_pct,
        'margin_diff': h_mg - a_mg,
        'rest_diff': h_rest - a_rest,
        'h_short_rest': int(h_rest < 7), 'a_short_rest': int(a_rest < 7),
        'travel_km_diff': a_travel_km - h_travel_km,
        'venue_familiarity': venue_fam,
        'temp': float(temp) if temp is not None else avg_temp
    }

    for stat in KEY_STATS:
        feat[f'diff_{stat}'] = h_roll.get(stat, 0) - a_roll.get(stat, 0)
        feat[f'total_{stat}'] = h_roll.get(stat, 0) + a_roll.get(stat, 0)

    X_new = pd.DataFrame([feat]).reindex(columns=X_model.columns, fill_value=0)

    pred_margin = margin_reg.predict(X_new)[0]
    pred_total = total_reg.predict(X_new)[0]

    home_score_pred = max(0, round((pred_total + pred_margin) / 2))
    away_score_pred = max(0, round((pred_total - pred_margin) / 2))

    home_win_prob = 1 - norm.cdf((0.5 - pred_margin) / model_rmse)
    away_win_prob = norm.cdf((-0.5 - pred_margin) / model_rmse)
    draw_prob = norm.cdf((0.5 - pred_margin) / model_rmse) - norm.cdf((-0.5 - pred_margin) / model_rmse)

    if home_win_prob > away_win_prob and home_win_prob > draw_prob:
        winner = home_team
        win_prob = home_win_prob * 100
    elif away_win_prob > home_win_prob and away_win_prob > draw_prob:
        winner = away_team
        win_prob = away_win_prob * 100
    else:
        winner = "Draw"
        win_prob = draw_prob * 100

    return {
        'home_team': home_team,
        'away_team': away_team,
        'winner': winner,
        'margin': round(abs(pred_margin)),
        'home_score': home_score_pred,
        'away_score': away_score_pred,
        'total': round(pred_total),
        'win_prob': win_prob,
        'draw_prob': draw_prob * 100,
        'home_win_prob': home_win_prob * 100,
        'away_win_prob': away_win_prob * 100,
        'h_elo': round(team_elo[home_team]),
        'a_elo': round(team_elo[away_team]),
        'h_wr': h_wr,
        'a_wr': a_wr,
        'h_rest': h_rest,
        'a_rest': a_rest,
    }


# ─────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🏉 AFL Gemini</h1>
  <p>Institutional-grade match forecasting · Powered by HistGradientBoosting + ELO</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
with st.spinner("Loading model & training on historical data…"):
    model_state, load_error = load_model()

if load_error:
    st.error(f"⚠️ {load_error}")
    st.info("Place `ReadyFor2026.csv` in the same directory as `app.py`, then refresh.")
    st.stop()

# Show validation metrics in a subtle banner
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("Win/Loss Accuracy", f"{model_state['val_accuracy']:.1f}%")
with col_m2:
    st.metric("Margin MAE", f"{model_state['val_mae']:.1f} pts")
with col_m3:
    st.metric("Margin RMSE (σ)", f"{model_state['model_rmse']:.1f} pts")

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Input form
# ─────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">⚙️ Match Details</div>', unsafe_allow_html=True)

with st.form("forecast_form"):
    col1, col2 = st.columns(2)

    with col1:
        home_team_display = st.selectbox(
            "🏠 Home Team",
            options=TEAM_DISPLAY,
            index=TEAM_DISPLAY.index("Sydney Swans") if "Sydney Swans" in TEAM_DISPLAY else 0,
            help="Select the home team"
        )

    with col2:
        away_team_display = st.selectbox(
            "✈️ Away Team",
            options=TEAM_DISPLAY,
            index=TEAM_DISPLAY.index("Carlton") if "Carlton" in TEAM_DISPLAY else 1,
            help="Select the away team"
        )

    # Venue
    known_venues = sorted(model_state['venue_map'].keys())
    venue_options = known_venues + ["Other / Unknown"]
    venue_name = st.selectbox(
        "🏟️ Venue",
        options=venue_options,
        index=0,
        help="Select the match venue. Choose 'Other / Unknown' if not listed."
    )
    if venue_name == "Other / Unknown":
        venue_name = st.text_input("Enter venue name manually", value="MCG")

    col3, col4 = st.columns(2)

    with col3:
        match_date = st.date_input(
            "📅 Match Date",
            value=pd.Timestamp.now().date(),
            help="The date of the match"
        )

    with col4:
        match_time = st.time_input(
            "🕐 Local Start Time",
            value=pd.Timestamp("2026-01-01 14:10").time(),
            step=60,
            help="Local start time of the match (e.g. 20:05)"
        )

    temp_celsius = st.number_input(
        "🌡️ Forecast Temperature (°C)",
        min_value=-10,
        max_value=50,
        value=18,
        step=1,
        help="Forecast temperature in whole degrees Celsius"
    )

    submitted = st.form_submit_button("🔮 Generate Forecast", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Run prediction
# ─────────────────────────────────────────────
if submitted:
    if home_team_display == away_team_display:
        st.error("Home and Away teams must be different.")
    else:
        start_time_str = f"{match_date} {match_time}"

        with st.spinner("Computing forecast…"):
            result = predict_match(
                model_state,
                home_team_display,
                away_team_display,
                venue_name,
                temp_celsius,
                start_time_str
            )

        if isinstance(result, str):
            st.error(f"Prediction error: {result}")
        else:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("### 📊 Forecast Results")

            # ── Winner banner ──
            winner_emoji = "🏆"
            if result['winner'] == "Draw":
                winner_label = "Expected Draw"
                winner_emoji = "🤝"
            else:
                winner_label = result['winner']

            st.markdown(f"""
            <div class="result-winner">
              <div style="font-size:0.8rem;color:#78909c;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.5rem;">Predicted Winner</div>
              <div class="team-name">{winner_emoji} {winner_label}</div>
              <div class="confidence">{result['win_prob']:.1f}% confidence</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Score boxes ──
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.markdown(f"""
                <div class="score-box">
                  <div class="score-label">{result['home_team']}</div>
                  <div class="score-value">{result['home_score']}</div>
                  <div class="score-sub">Home</div>
                </div>
                """, unsafe_allow_html=True)
            with sc2:
                st.markdown(f"""
                <div class="score-box">
                  <div class="score-label">Total Points</div>
                  <div class="score-value">{result['total']}</div>
                  <div class="score-sub">Combined</div>
                </div>
                """, unsafe_allow_html=True)
            with sc3:
                st.markdown(f"""
                <div class="score-box">
                  <div class="score-label">{result['away_team']}</div>
                  <div class="score-value">{result['away_score']}</div>
                  <div class="score-sub">Away</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Probability breakdown ──
            st.markdown("#### Win Probabilities")
            prob_col1, prob_col2, prob_col3 = st.columns(3)
            with prob_col1:
                st.metric(result['home_team'], f"{result['home_win_prob']:.1f}%")
            with prob_col2:
                st.metric("Draw", f"{result['draw_prob']:.1f}%")
            with prob_col3:
                st.metric(result['away_team'], f"{result['away_win_prob']:.1f}%")

            # ── Progress bars for visual probability ──
            st.markdown(f"""
            <div style="margin:0.5rem 0 1.5rem;">
              <div style="display:flex;height:10px;border-radius:8px;overflow:hidden;gap:2px;">
                <div style="width:{result['home_win_prob']:.1f}%;background:linear-gradient(90deg,#00c853,#00e676);border-radius:8px 0 0 8px;"></div>
                <div style="width:{result['draw_prob']:.1f}%;background:#546e7a;"></div>
                <div style="width:{result['away_win_prob']:.1f}%;background:linear-gradient(90deg,#ff6d00,#ff9100);border-radius:0 8px 8px 0;"></div>
              </div>
              <div style="display:flex;justify-content:space-between;margin-top:0.4rem;font-size:0.75rem;color:#78909c;">
                <span>{result['home_team']}</span>
                <span>Draw</span>
                <span>{result['away_team']}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Match stats breakdown ──
            st.markdown("#### Match Context")
            st.markdown(f"""
            <div class="card">
              <div class="stat-row">
                <span class="stat-label">Predicted Margin</span>
                <span class="stat-val">{result['margin']} pts ({result['winner']})</span>
              </div>
              <div class="stat-row">
                <span class="stat-label">Home ELO Rating</span>
                <span class="stat-val">{result['h_elo']}</span>
              </div>
              <div class="stat-row">
                <span class="stat-label">Away ELO Rating</span>
                <span class="stat-val">{result['a_elo']}</span>
              </div>
              <div class="stat-row">
                <span class="stat-label">Home Win Rate (last 6)</span>
                <span class="stat-val">{result['h_wr']*100:.0f}%</span>
              </div>
              <div class="stat-row">
                <span class="stat-label">Away Win Rate (last 6)</span>
                <span class="stat-val">{result['a_wr']*100:.0f}%</span>
              </div>
              <div class="stat-row">
                <span class="stat-label">Home Days Rest</span>
                <span class="stat-val">{result['h_rest']} days{"  ⚠️ Short rest" if result['h_rest'] < 7 else ""}</span>
              </div>
              <div class="stat-row">
                <span class="stat-label">Away Days Rest</span>
                <span class="stat-val">{result['a_rest']} days{"  ⚠️ Short rest" if result['a_rest'] < 7 else ""}</span>
              </div>
              <div class="stat-row">
                <span class="stat-label">Venue</span>
                <span class="stat-val">{venue_name}</span>
              </div>
              <div class="stat-row">
                <span class="stat-label">Temperature</span>
                <span class="stat-val">{temp_celsius}°C</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.caption("Model: HistGradientBoostingRegressor · ELO with dynamic home advantage · Continuity-corrected win probabilities")
