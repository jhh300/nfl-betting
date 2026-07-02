# NFL Betting Pipeline (Odds + Model + Model-vs-Vegas Tracking + Streamlit)

Pulls **NFL odds** from **The Odds API**, builds leakage-free team features from
**nfl-data-py** schedules, trains gradient-boosted models for **moneyline /
spreads / totals**, computes calibrated **p_true**, compares to de-vigged book
probabilities for **edge/EV/Kelly**, and — new — **tracks model performance
against Vegas closing lines** (e.g. model says home by 10, Vegas closed −8,
home won by 11 → model was closer and its side covered). Closing lines come
from nflverse, so tracking needs **no API key**. Bets can be logged to SQLite
with CLV tracking. Includes a Streamlit dashboard.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cp .env.example .env        # then put THE_ODDS_API_KEY=... inside
```

## Usage

```bash
# Dashboard (tabs: Picks, Model vs Vegas, Backtest, Diagnostics)
streamlit run app.py

# CLI: current-week picks (needs THE_ODDS_API_KEY)
python cli.py picks --season 2025

# CLI: score the model against Vegas closing lines (no API key needed)
python cli.py track --seasons 2023,2024 --save_csv data/model_vs_vegas.csv

# Log / settle / CLV-track real bets in SQLite
python bets_cli.py add --game_id <GAME_ID> --market spreads --side HOME \
  --entry_line -2.5 --book_key draftkings --price -110 --stake 150
python bets_cli.py settle --season 2025
```

## Model vs Vegas tracking

For every completed game in a walk-forward backtest (models only ever see
prior seasons), the tracker records the model's predicted margin/total, the
Vegas closing line, and the actual result, then reports:

- **MAE**: is the model's number closer to the final margin/total than the line?
- **ATS / O-U record** when the model disagrees with the line, bucketed by
  disagreement size (you need >52.4% to beat −110 juice)
- **Moneyline Brier score** vs the de-vigged market probability

## Notes

- Features are joined **as-of strictly before each game week** — no game's own
  result leaks into its features, and upcoming games get real features from
  the most recent completed week.
- Team features include EPA/play (off + def), pass EPA per dropback, and
  turnover margin, aggregated automatically from free nflverse player-weekly
  data — no key needed. If that download fails, the model degrades gracefully
  to points-based features.
- Optional team-metric CSVs (DVOA, success rate, etc.) can be uploaded in the
  sidebar; `build_extra_metrics_csvs.py` generates templates.
- `basketball_app.py` / `basketball_cli.py` are a self-contained NBA/NCAAB
  bolt-on (`streamlit run basketball_app.py`).
