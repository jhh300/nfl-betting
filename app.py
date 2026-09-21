from __future__ import annotations
import datetime as dt
import traceback
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import os

import fetch_nfl_data as fn

st.set_page_config(page_title="NFL Picks", layout="wide")

# ---- Utilities --------------------------------------------------------------

def utc_now_year() -> int:
    try:
        from datetime import UTC
        return dt.datetime.now(UTC).year
    except Exception:
        return dt.datetime.utcnow().year

def pick_top(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    order_cols = [c for c in ["ev_per_usd","edge_vs_book_devig","odds_decimal"] if c in df.columns]
    if not order_cols: return df.head(n)
    return df.sort_values(order_cols, ascending=[False]*len(order_cols)).head(n)

def _coalesce_team_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for side in ["home","away"]:
        base = f"{side}_team"; x = f"{base}_x"; y = f"{base}_y"
        series = d.get(base)
        if x in d.columns: series = d[x] if series is None else series.where(series.notna(), d[x])
        if y in d.columns: series = d[y] if series is None else series.where(series.notna(), d[y])
        if series is not None: d[base] = series
        for c in (x, y):
            if c in d.columns: d.drop(columns=c, inplace=True)
    return d

# ---- Team conference/division map + pick filters ------------------------------

NFL_TEAMS = {
    "BUF": ("AFC","East"),  "MIA": ("AFC","East"),  "NE":  ("AFC","East"),  "NYJ": ("AFC","East"),
    "BAL": ("AFC","North"), "CIN": ("AFC","North"), "CLE": ("AFC","North"), "PIT": ("AFC","North"),
    "HOU": ("AFC","South"), "IND": ("AFC","South"), "JAC": ("AFC","South"), "TEN": ("AFC","South"),
    "DEN": ("AFC","West"),  "KC":  ("AFC","West"),  "LV":  ("AFC","West"),  "LAC": ("AFC","West"),
    "DAL": ("NFC","East"),  "NYG": ("NFC","East"),  "PHI": ("NFC","East"),  "WAS": ("NFC","East"),
    "CHI": ("NFC","North"), "DET": ("NFC","North"), "GB":  ("NFC","North"), "MIN": ("NFC","North"),
    "ATL": ("NFC","South"), "CAR": ("NFC","South"), "NO":  ("NFC","South"), "TB":  ("NFC","South"),
    "ARI": ("NFC","West"),  "LAR": ("NFC","West"),  "SEA": ("NFC","West"),  "SF":  ("NFC","West"),
}
DIVISIONS = [f"{c} {d}" for c in ("AFC","NFC") for d in ("East","North","South","West")]
TIME_SLOTS = ["Thursday","Fri/Sat","Sunday early (1 PM)","Sunday late (4 PM)","Sunday night","Monday"]

def _time_slot(ts) -> str:
    """Bucket an Eastern-time kickoff into the standard NFL slate slots."""
    if ts is None or pd.isna(ts): return "TBD"
    day = ts.strftime("%a")
    if day == "Thu": return "Thursday"
    if day in ("Fri","Sat"): return "Fri/Sat"
    if day == "Mon": return "Monday"
    if day == "Sun":
        if ts.hour < 15: return "Sunday early (1 PM)"
        if ts.hour < 19: return "Sunday late (4 PM)"
        return "Sunday night"
    return day

def _team_in(team: str, conf: str = None, div: str = None) -> bool:
    c, d = NFL_TEAMS.get(str(team), ("", ""))
    if conf and c != conf: return False
    if div and f"{c} {d}" != div: return False
    return True

def filter_picks(picks: pd.DataFrame, slot_by_game: dict,
                 week: int | None, conf: str | None, div: str | None, slot: str | None) -> pd.DataFrame:
    """Filter picks by week, conference, division (either team), and ET kickoff slot."""
    f = picks
    if week is not None:
        f = f[pd.to_numeric(f["week"], errors="coerce") == week]
    if conf:
        f = f[f.apply(lambda r: _team_in(r["home_team"], conf=conf) or _team_in(r["away_team"], conf=conf), axis=1)]
    if div:
        f = f[f.apply(lambda r: _team_in(r["home_team"], div=div) or _team_in(r["away_team"], div=div), axis=1)]
    if slot:
        f = f[f["game_id"].map(slot_by_game).eq(slot)]
    return f

# ---- Per-game summary: model's predicted side vs. the best-EV market pick ----

def _model_ml_label(home: str, away: str, p_home) -> str:
    p = pd.to_numeric(pd.Series([p_home]), errors="coerce").iloc[0]
    if pd.isna(p): return "—"
    return f"{home} ({p:.0%})" if p >= 0.5 else f"{away} ({1-p:.0%})"

def _model_spread_label(home: str, away: str, margin) -> str:
    m = pd.to_numeric(pd.Series([margin]), errors="coerce").iloc[0]
    if pd.isna(m): return "—"
    if abs(m) < 0.05: return "PK"
    return f"{home} -{m:.1f}" if m > 0 else f"{away} -{-m:.1f}"

def _best_ev_label(rows: pd.DataFrame) -> str:
    """The single highest-EV candidate for a game+market, formatted for display.
    Shows the best candidate even if it's not positive-EV, labeling it clearly."""
    if rows is None or rows.empty: return "No line"
    r = rows.sort_values("ev_per_usd", ascending=False).iloc[0]
    ev = pd.to_numeric(pd.Series([r.get("ev_per_usd")]), errors="coerce").iloc[0]
    sel  = _pick_selection(r)
    odds = _fmt_odds(r.get("odds_american"))
    book = r.get("book_title") or ""
    if pd.isna(ev) or ev <= 0:
        return f"{sel} ({odds}) {book} — no edge"
    return f"{sel} ({odds}) {book} · {ev:+.1%} EV"

def build_game_table(games: pd.DataFrame, picks_all: pd.DataFrame) -> pd.DataFrame:
    """One row per game: model's predicted ML/spread side next to the best-EV
    market pick for each, so you can see where the model's lean and the
    recommended bet agree or diverge."""
    if games is None or games.empty: return pd.DataFrame()
    rows = []
    for _, g in games.iterrows():
        gid = g["game_id"]
        gp = picks_all[picks_all["game_id"] == gid] if picks_all is not None else pd.DataFrame()
        rows.append({
            "week":              g.get("week"),
            "kickoff":           g.get("kickoff"),
            "matchup":           f"{g.get('away_team')} @ {g.get('home_team')}",
            "model_ml_pick":     _model_ml_label(g.get("home_team"), g.get("away_team"), g.get("p_home")),
            "ev_ml_pick":        _best_ev_label(gp[gp["market_key"].eq("moneyline")]),
            "model_spread_pick": _model_spread_label(g.get("home_team"), g.get("away_team"), g.get("pred_margin")),
            "ev_spread_pick":    _best_ev_label(gp[gp["market_key"].eq("spreads")]),
        })
    return pd.DataFrame(rows)

# ---- ESPN-style pick cards ---------------------------------------------------

_ESPN_LOGO_FIX = {"JAC": "jax", "WAS": "wsh"}

def _logo_url(abbr: str) -> str:
    a = str(abbr or "").upper()
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{_ESPN_LOGO_FIX.get(a, a.lower())}.png"

_ESPN_CSS = """
<style>
.espn-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px;margin-top:4px}
.espn-card{background:#fff;border:1px solid #dcdddf;border-radius:8px;overflow:hidden;
  font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;box-shadow:0 1px 2px rgba(0,0,0,.05)}
.espn-head{display:flex;justify-content:space-between;align-items:center;padding:7px 12px;
  border-bottom:1px solid #edeef0;font-size:10.5px;font-weight:700;color:#6c6d6f;
  text-transform:uppercase;letter-spacing:.6px}
.espn-team{display:flex;align-items:center;gap:9px;padding:7px 12px}
.espn-team img{width:26px;height:26px;flex:none}
.espn-team .abbr{font-weight:700;font-size:15px;color:#1c1d1f}
.espn-team .sub{font-size:11px;color:#6c6d6f;font-weight:500}
.espn-team .proj{margin-left:auto;font-weight:700;font-size:19px;color:#1c1d1f;font-variant-numeric:tabular-nums}
.espn-team.dim .abbr,.espn-team.dim .proj{color:#6c6d6f}
.espn-picks{border-top:1px solid #edeef0;padding:7px 12px 10px;display:flex;flex-direction:column;gap:6px}
.espn-pick{display:flex;align-items:center;gap:8px;font-size:12.5px;color:#1c1d1f;line-height:1.35}
.espn-pick .mkt{flex:none;font-size:9.5px;font-weight:700;color:#6c6d6f;letter-spacing:.6px;
  text-transform:uppercase;width:52px}
.espn-pick .sel{font-weight:700;white-space:nowrap}
.espn-pick .odds{color:#6c6d6f;font-variant-numeric:tabular-nums}
.espn-pick .book{font-size:10.5px;color:#9a9b9d;text-transform:uppercase;letter-spacing:.3px}
.espn-pick .ev{margin-left:auto;flex:none;font-weight:700;font-size:11px;border-radius:4px;
  padding:1px 7px;font-variant-numeric:tabular-nums}
.espn-pick .ev.pos{background:#e6f4ea;color:#0a7d33}
.espn-pick .ev.neg{background:#fdecea;color:#b3261e}
.espn-pick .stake{flex:none;width:56px;text-align:right;font-weight:600;font-variant-numeric:tabular-nums}
.espn-note{font-size:10.5px;color:#9a9b9d;padding:0 12px 2px;margin-top:-3px}
</style>
"""

import html as _html

def _esc(x) -> str:
    """Escape third-party strings (team/book names, times) before HTML interpolation."""
    return _html.escape(str(x)) if x is not None else ""

def _fmt_odds(a) -> str:
    try:
        a = float(a)
        return f"{a:+.0f}"
    except (TypeError, ValueError):
        return "—"

def _pick_selection(r: pd.Series) -> str:
    side, mk = r.get("side"), r.get("market_key")
    home, away = r.get("home_team", ""), r.get("away_team", "")
    line = pd.to_numeric(pd.Series([r.get("line")]), errors="coerce").iloc[0]
    if mk == "moneyline":
        return f"{home if side == 'HOME' else away} ML"
    if mk == "spreads":
        team = home if side == "HOME" else away
        return f"{team} {line:+g}" if pd.notna(line) else team
    return f"{'O' if side == 'OVER' else 'U'} {line:g}" if pd.notna(line) else str(side)

def render_espn_picks(games: pd.DataFrame, bettable: pd.DataFrame, pred: pd.DataFrame | None) -> None:
    """Render one ESPN-scoreboard-style card per game in `games` (every
    scheduled/odds-matched game), best-EV game first. `bettable` supplies the
    pick rows for games that clear the edge/EV filter; a game with none still
    gets a card — with a 'no qualifying edge' note — instead of disappearing,
    so the grid always matches the full slate."""
    info = {}
    if pred is not None and not pred.empty:
        for _, g in pred.iterrows():
            info[g["game_id"]] = g
    ev_by_game = (bettable.groupby("game_id")["ev_per_usd"].max()
                  if bettable is not None and not bettable.empty else pd.Series(dtype=float))
    game_ids = games["game_id"].dropna().unique().tolist()
    order = sorted(game_ids, key=lambda gid: -ev_by_game.get(gid, -np.inf))
    _market_order = {"moneyline": 0, "spreads": 1, "totals": 2}
    cards = []
    for gid in order:
        g = info.get(gid)
        grow = games[games["game_id"] == gid].iloc[0]
        home = g["home_team"] if g is not None else grow.get("home_team", "?")
        away = g["away_team"] if g is not None else grow.get("away_team", "?")
        h_pts = f"{g['home_pts']:.0f}" if g is not None and pd.notna(g.get("home_pts")) else ""
        a_pts = f"{g['away_pts']:.0f}" if g is not None and pd.notna(g.get("away_pts")) else ""
        home_dim = " dim" if (g is not None and g["home_pts"] < g["away_pts"]) else ""
        away_dim = " dim" if (g is not None and g["away_pts"] <= g["home_pts"]) else ""
        kick = str((g.get("kickoff") if g is not None else "") or grow.get("kickoff") or "")
        week = g.get("week") if g is not None else grow.get("week")
        head_l = f"NFL{f' &bull; WEEK {int(week)}' if pd.notna(week) else ''}"

        all_rows = (bettable[bettable["game_id"] == gid].sort_values("ev_per_usd", ascending=False)
                    if bettable is not None and not bettable.empty else pd.DataFrame())
        if all_rows.empty:
            pick_rows_html = ('<div class="espn-note" style="padding:8px 12px 2px">'
                               'No picks clear the edge/EV threshold for this game.</div>')
            note = ""
        else:
            # One line per market (best-priced book), so ML and spread both
            # show instead of one market sweeping all slots on raw EV.
            rows = (all_rows.drop_duplicates("market_key", keep="first")
                            .assign(_ord=lambda d: d["market_key"].map(_market_order).fillna(9))
                            .sort_values("_ord").drop(columns="_ord"))
            n_more = len(all_rows) - len(rows)
            pick_rows = []
            for _, r in rows.iterrows():
                ev = pd.to_numeric(pd.Series([r.get("ev_per_usd")]), errors="coerce").iloc[0]
                ev_cls = "pos" if pd.notna(ev) and ev > 0 else "neg"
                ev_txt = f"{ev:+.1%} EV" if pd.notna(ev) else "—"
                stake = pd.to_numeric(pd.Series([r.get("stake_usd")]), errors="coerce").iloc[0]
                stake_txt = f"${stake:,.0f}" if pd.notna(stake) and stake > 0 else "—"
                mkt = {"moneyline": "ML", "spreads": "Spread", "totals": "Total"}.get(r.get("market_key"), r.get("market_key"))
                pick_rows.append(
                    f'<div class="espn-pick"><span class="mkt">{_esc(mkt)}</span>'
                    f'<span class="sel">{_esc(_pick_selection(r))}</span>'
                    f'<span class="odds">{_fmt_odds(r.get("odds_american"))}</span>'
                    f'<span class="book">{_esc(r.get("book_title") or "")}</span>'
                    f'<span class="ev {ev_cls}">{ev_txt}</span>'
                    f'<span class="stake">{stake_txt}</span></div>'
                )
            pick_rows_html = "".join(pick_rows)
            spread_rows = rows[rows["market_key"].eq("spreads")]
            note_row = spread_rows.iloc[0] if not spread_rows.empty else rows.iloc[0]
            note = ""
            ml_val = pd.to_numeric(pd.Series([note_row.get("model_line")]), errors="coerce").iloc[0]
            ep_val = pd.to_numeric(pd.Series([note_row.get("edge_pts")]), errors="coerce").iloc[0]
            if pd.notna(ml_val) and pd.notna(ep_val):
                note = f'<div class="espn-note">Model line {ml_val:+.1f} &bull; {ep_val:+.1f} pts vs book</div>'
            if n_more > 0:
                note += f'<div class="espn-note">+{n_more} more pick{"s" if n_more > 1 else ""} in table view</div>'
        cards.append(
            f'<div class="espn-card">'
            f'<div class="espn-head"><span>{head_l}</span><span>{_esc(kick)}</span></div>'
            f'<div class="espn-team{away_dim}"><img src="{_esc(_logo_url(away))}"/>'
            f'<span class="abbr">{_esc(away)}</span><span class="sub">Away</span><span class="proj">{a_pts}</span></div>'
            f'<div class="espn-team{home_dim}"><img src="{_esc(_logo_url(home))}"/>'
            f'<span class="abbr">{_esc(home)}</span><span class="sub">Home</span><span class="proj">{h_pts}</span></div>'
            f'<div class="espn-picks">{pick_rows_html}</div>{note}'
            f'</div>'
        )
    st.markdown(_ESPN_CSS + f'<div class="espn-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

# ---- Tweet generation --------------------------------------------------------

_MARKET_EMOJI = {"moneyline": "💰", "spreads": "📊", "totals": "🎯"}

def _tweet_text(r: pd.Series) -> str:
    matchup = f"{_esc_plain(r['away_team'])} @ {_esc_plain(r['home_team'])}"
    sel     = _pick_selection(r)
    odds    = _fmt_odds(r.get("odds_american"))
    ev      = pd.to_numeric(pd.Series([r.get("ev_per_usd")]), errors="coerce").iloc[0]
    book    = r.get("book_title") or ""
    emoji   = _MARKET_EMOJI.get(r.get("market_key"), "🏈")
    week    = r.get("week")
    wk_txt  = f" Wk{int(week)}" if pd.notna(week) else ""

    lines = [f"🏈 NFL{wk_txt}: {matchup}", f"{emoji} {sel} ({odds}){f' — {book}' if book else ''}"]
    if pd.notna(ev):
        lines.append(f"Model edge: {ev:+.1%} EV")
    lines.append("#NFLPicks #NFLBetting")
    text = "\n".join(lines)

    if len(text) > 280:  # trim the book name first, then drop the hashtags
        lines[1] = f"{emoji} {sel} ({odds})"
        text = "\n".join(lines)
    if len(text) > 280:
        text = "\n".join(lines[:-1])
    return text[:280]

def _esc_plain(x) -> str:
    """Plain-text (non-HTML) cleanup for tweet bodies."""
    return str(x) if x is not None else ""

def build_tweets(picks: pd.DataFrame, per_week: int = 5) -> dict:
    """Top-N picks by EV for each week present in `picks`, formatted as tweets."""
    if picks is None or picks.empty: return {}
    out = {}
    for wk, grp in picks.groupby(picks["week"].astype("Int64")):
        if pd.isna(wk): continue
        top = grp.sort_values("ev_per_usd", ascending=False).head(per_week)
        out[int(wk)] = [_tweet_text(r) for _, r in top.iterrows()]
    return dict(sorted(out.items()))

PICKS_COLUMN_CONFIG = {
    "p_true":              st.column_config.NumberColumn("Model win prob", format="percent"),
    "book_p_devig":        st.column_config.NumberColumn("Book prob (devig)", format="percent"),
    "edge_vs_book_devig":  st.column_config.NumberColumn("Edge vs book", format="percent"),
    "kelly_frac":          st.column_config.NumberColumn("Kelly frac", format="percent"),
    "ev_per_usd":          st.column_config.NumberColumn("EV per $1", format="$%.3f"),
    "stake_usd":           st.column_config.NumberColumn("Stake", format="dollar"),
    "line":                st.column_config.NumberColumn("Book line"),
    "model_line":          st.column_config.NumberColumn("Model line", format="%.1f"),
    "edge_pts":            st.column_config.NumberColumn("Edge (pts)", format="%.1f"),
    "odds_american":       st.column_config.NumberColumn("Odds (US)"),
    "odds_decimal":        st.column_config.NumberColumn("Odds (dec)", format="%.2f"),
}

# ---- Cached data/model functions -------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Loading schedule and stats...")
def load_data(season_list_tuple: tuple) -> tuple:
    sched  = fn.load_schedule(list(season_list_tuple))
    weekly = fn.team_weekly_from_schedule(sched)
    return sched, weekly

@st.cache_data(show_spinner="Training models...")
def fit_models(train_df: pd.DataFrame) -> tuple:
    margin_model, s_margin, cv_mae_margin = fn.fit_margin_model(train_df)
    total_model,  s_total,  cv_mae_total  = fn.fit_total_model(train_df)
    return margin_model, s_margin, cv_mae_margin, total_model, s_total, cv_mae_total

@st.cache_data(ttl=300, show_spinner="Fetching live odds...")
def fetch_odds(api_key: str, markets_tuple: tuple, books: str) -> pd.DataFrame:
    api_market_map = {"moneyline": "h2h", "spreads": "spreads", "totals": "totals"}
    frames = []
    for mk in markets_tuple:
        od = fn.fetch_odds_current(api_key, market=api_market_map[mk], bookmakers=books or None)
        if isinstance(od, pd.DataFrame) and not od.empty and od.dropna(axis=1, how="all").shape[1] > 0:
            frames.append(od)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="Running walk-forward backtest (one model per season)...")
def cached_backtest(season_list_tuple: tuple, eval_seasons_tuple: tuple) -> pd.DataFrame:
    sched, weekly = load_data(season_list_tuple)
    return fn.run_backtest(sched, weekly, list(eval_seasons_tuple))

def backtest_for(eval_seasons: List[int]) -> pd.DataFrame:
    eval_s = sorted({int(s) for s in eval_seasons})
    all_s  = [s for s in range(min(eval_s) - fn.TRAIN_WINDOW_SEASONS, max(eval_s) + 1) if s >= 2000]
    return cached_backtest(tuple(all_s), tuple(eval_s))

# ---- Sidebar ----------------------------------------------------------------

st.sidebar.title("Controls")
_env_key = os.getenv("THE_ODDS_API_KEY", "")
if not _env_key:
    try:
        _env_key = st.secrets.get("THE_ODDS_API_KEY", "")   # Streamlit Cloud secrets
    except Exception:
        pass
# SECURITY: never pre-fill the widget with the server-side key — widget values
# are sent to the viewer's browser. The input is an override only.
_key_override = st.sidebar.text_input(
    "The Odds API key" + (" (configured — leave blank)" if _env_key else ""),
    value="", type="password", key="api_key",
    help="Leave blank to use the key from .env / Streamlit secrets.")
api_key = _key_override or _env_key
books        = st.sidebar.text_input("Bookmakers (comma-separated)", value="fanduel,draftkings,betmgm", key="books_text")
season_pick  = st.sidebar.number_input("Season (target)", min_value=2000, max_value=2100, value=int(utc_now_year()), step=1, key="season_input")
markets_choice = st.sidebar.multiselect(
    "Markets", ["moneyline","spreads","totals"], default=["moneyline","spreads"],
    key="markets_multiselect",
    help="Totals is off by default — backtests show the totals model loses money.")
kelly_bankroll = st.sidebar.number_input("Bankroll for Kelly ($)", min_value=0.0, value=10000.0, step=100.0)
kelly_frac     = st.sidebar.slider("Kelly fraction", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
min_edge_pts   = st.sidebar.slider("Min edge vs line (pts)", min_value=0.0, max_value=7.0, value=2.0, step=0.5,
                                   help="Skip spread/total picks where the model and the book differ by less "
                                        "than this. Backtests show the edge concentrates in bigger disagreements.")
top_n          = st.sidebar.slider("Show top N", min_value=5, max_value=50, value=25, step=5)
with st.sidebar.expander("Optional team metrics CSVs"):
    uploaded_files = st.file_uploader("Upload CSVs (team, season, [week], plus metrics)", type=["csv"], accept_multiple_files=True, key="uploader")

st.title("NFL Picks")

tab_picks, tab_tracker, tab_vegas, tab_backtest, tab_diag = st.tabs(
    ["Picks", "Season Tracker", "Model vs Vegas", "Backtest", "Diagnostics"])

# ============================================================================
# TAB 1: PICKS
# ============================================================================
with tab_picks:
    if st.button("Generate Picks", type="primary"):
        try:
            season = int(season_pick)
            train_seasons: List[int] = list(range(season - fn.TRAIN_WINDOW_SEASONS, season))
            season_list:   List[int] = sorted(set(train_seasons + [season]))

            sched_all, weekly_all = load_data(tuple(season_list))

            extra_df = fn.load_extra_from_uploads(uploaded_files)
            if not extra_df.empty:
                weekly_all = fn.integrate_extra(weekly_all, extra_df)

            feats_all  = fn.build_game_features(sched_all, weekly_all)
            # Prior seasons + this season's completed games (unplayed rows have
            # no result and drop out of training automatically)
            train_df   = fn.make_training_set(feats_all[feats_all["season"] <= season].copy())

            (margin_model, s_margin, cv_mae_margin,
             total_model,  s_total,  cv_mae_total) = fit_models(train_df)

            if not api_key:
                st.error("Enter your **The Odds API** key in the sidebar (or set THE_ODDS_API_KEY in .env).")
                st.stop()

            odds_live = fetch_odds(api_key, tuple(markets_choice), books)
            if odds_live.empty:
                st.warning("No odds returned. Check API key/plan, bookmakers, or market selections.")
                st.stop()
            odds_live = fn.prep_odds(odds_live)

            sched_target = sched_all[sched_all["season"] == season].copy()
            merged_any   = fn.merge_odds_with_schedule(odds_live, sched_target)
            if merged_any.empty: merged_any = fn.merge_odds_with_schedule(odds_live, sched_all)
            if merged_any.empty: merged_any = fn.merge_odds_by_pair_only(odds_live, sched_target)
            if merged_any.empty: merged_any = fn.merge_odds_by_pair_only(odds_live, sched_all)
            if merged_any.empty:
                st.error("No market rows matched the schedule (after time and pair-only fallbacks).")
                with st.expander("Debug: odds vs schedule pairs"):
                    st.write("Sample odds rows:", odds_live.head(10))
                    st.write("Sample schedule rows:", sched_target.head(10))
                st.stop()

            # Keep EVERY matched upcoming game — odds can span two weeks late in
            # the season (remaining MNF game + next week's board, wk18 + wildcard)
            merged = _coalesce_team_cols(merged_any)
            weeks = pd.to_numeric(merged["week"], errors="coerce").dropna().astype(int)
            week_label = (f"Week {weeks.min()}" if weeks.min() == weeks.max()
                          else f"Weeks {weeks.min()}–{weeks.max()}")

            if "home_team" not in merged.columns or "away_team" not in merged.columns:
                sched_teams = sched_all[["game_id","home_team","away_team"]].drop_duplicates("game_id")
                merged = merged.merge(sched_teams, on="game_id", how="left", suffixes=("","_sch"))
                for side in ["home","away"]:
                    base = f"{side}_team"; sch = f"{base}_sch"
                    if base not in merged.columns or merged[base].isna().all():
                        if sch in merged.columns: merged[base] = merged[sch]
                    if sch in merged.columns: merged.drop(columns=sch, inplace=True)

            feat_all_cols = fn.FEATURE_COLS
            present_feat_cols = [c for c in feat_all_cols if c in feats_all.columns]
            gp_cols = [c for c in ["home_games_played","away_games_played"] if c in feats_all.columns]
            base_cols = ["game_id","season","week","home_team","away_team"]
            merged = merged.merge(
                feats_all[base_cols+present_feat_cols+gp_cols].drop_duplicates("game_id"),
                on=["game_id","season","week"],
                how="left", validate="m:1", suffixes=("","_feat"))
            merged = _coalesce_team_cols(merged)

            game_feats = merged.drop_duplicates(subset=["game_id"]).copy().reset_index(drop=True)
            for c in feat_all_cols + ["home_games_played","away_games_played"]:
                if c not in game_feats.columns: game_feats[c] = np.nan
            X_full = game_feats[feat_all_cols].apply(pd.to_numeric, errors="coerce")

            mu_margin = margin_model.predict(fn.align_features_for_model(margin_model, X_full))
            mu_total  = total_model.predict(fn.align_features_for_model(total_model,  X_full))
            sigma_mult = fn.game_sigma_multiplier(game_feats["home_games_played"], game_feats["away_games_played"])
            sigma_margin_arr = s_margin * sigma_mult
            sigma_total_arr  = s_total  * sigma_mult
            p_home    = fn.win_prob_from_margin(mu_margin, sigma_margin_arr)

            p_home_df       = pd.DataFrame({"game_id": game_feats["game_id"].values, "p_home_model": p_home})
            mu_margin_map    = dict(zip(game_feats["game_id"].values, mu_margin))
            mu_total_map     = dict(zip(game_feats["game_id"].values,  mu_total))
            sigma_margin_map = dict(zip(game_feats["game_id"].values, sigma_margin_arr))
            sigma_total_map  = dict(zip(game_feats["game_id"].values, sigma_total_arr))

            picks_df_all = fn.assemble_picks(
                merged, markets_choice, p_home_df,
                mu_margin_map, mu_total_map,
                sigma_margin_map, sigma_total_map,
                float(kelly_bankroll), float(kelly_frac),
            )
            picks_df_live = fn.filter_bettable(picks_df_all, float(min_edge_pts))

            if picks_df_live.empty:
                st.warning(f"No picks passed the edge/EV filter "
                           f"({len(picks_df_all)} candidates, none with positive EV and ≥{min_edge_pts:g} pts edge). "
                           f"Lower the min-edge slider to see weaker candidates.")
                st.stop()

            pred = pd.DataFrame({
                "game_id":    game_feats["game_id"],
                "week":       game_feats.get("week", pd.Series(dtype="Int64")),
                "home_team":  game_feats.get("home_team", pd.Series(dtype=str)),
                "away_team":  game_feats.get("away_team", pd.Series(dtype=str)),
                "pred_margin": mu_margin,
                "pred_total":  mu_total,
            })
            pred["home_pts"] = (pred["pred_total"] + pred["pred_margin"]) / 2.0
            pred["away_pts"] =  pred["pred_total"] - pred["home_pts"]
            pred["p_home"]   = p_home
            pred["kickoff"] = ""
            pred["kickoff_dt"] = pd.NaT
            if "commence_dt" in game_feats.columns:
                kick = pd.to_datetime(game_feats["commence_dt"], errors="coerce", utc=True)
                try:
                    et = kick.dt.tz_convert("US/Eastern")
                    pred["kickoff_dt"] = et.values
                    pred["kickoff"] = et.dt.strftime("%a %-I:%M %p ET").fillna("")
                except Exception:
                    pass

            st.session_state["picks"] = picks_df_live
            st.session_state["picks_all"] = picks_df_all
            st.session_state["pred_scores"] = pred
            st.session_state["last_diagnostics"] = {
                "train_rows":           int(train_df["y_home_win"].notna().sum()),
                "train_seasons":        f"{train_seasons[0]}–{season} (incl. this season's completed games)",
                "win_prob_source":      "derived from margin model: P(home) = Phi(margin/sigma)",
                "margin_used_features": list(map(str, fn.model_used_features(margin_model))),
                "total_used_features":  list(map(str, fn.model_used_features(total_model))),
                "margin_sigma_oof":     float(s_margin),
                "total_sigma_oof":      float(s_total),
                "cv_mae_margin":        float(cv_mae_margin),
                "cv_mae_total":         float(cv_mae_total),
                "extra_csv_rows":       int(len(extra_df)) if isinstance(extra_df, pd.DataFrame) else 0,
                "week_label":           week_label,
                "n_games":              int(game_feats["game_id"].nunique()),
                "season":               season,
            }

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            with st.expander("Traceback", expanded=True):
                st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))

    # Render from session state so results survive reruns (slider changes, tab switches)
    picks = st.session_state.get("picks")
    if picks is None:
        st.info("Set your options in the sidebar, then click **Generate Picks**.")
    else:
        diag = st.session_state.get("last_diagnostics", {})
        if diag:
            st.caption(f"{diag['week_label']}, Season {diag['season']} — all {diag['n_games']} games, "
                       f"trained on {diag['train_seasons']}")
        pred_all = st.session_state.get("pred_scores")
        slot_by_game = {}
        if pred_all is not None and "kickoff_dt" in pred_all.columns:
            slot_by_game = {g: _time_slot(t) for g, t in zip(pred_all["game_id"], pred_all["kickoff_dt"])}

        # Filter bar — week defaults to the current (earliest upcoming) week.
        # Options come from the full game list (pred_all), not just games with
        # a bettable pick, so a week with zero qualifying edges is still pickable.
        week_source = pred_all if pred_all is not None and not pred_all.empty else picks
        c1, c2, c3, c4 = st.columns(4)
        weeks = sorted(pd.to_numeric(week_source["week"], errors="coerce").dropna().astype(int).unique())
        wk_sel   = c1.selectbox("Week", ["All weeks"] + [f"Week {w}" for w in weeks],
                                index=1 if weeks else 0, key="flt_week")
        conf_sel = c2.selectbox("Conference", ["All", "AFC", "NFC"], key="flt_conf")
        div_sel  = c3.selectbox("Division", ["All"] + DIVISIONS, key="flt_div")
        slots_present = [s for s in TIME_SLOTS if s in set(slot_by_game.values())]
        slot_sel = c4.selectbox("Kickoff (ET)", ["All times"] + slots_present, key="flt_slot")

        _flt = dict(
            week=int(wk_sel.split()[-1]) if wk_sel != "All weeks" else None,
            conf=conf_sel if conf_sel != "All" else None,
            div=div_sel if div_sel != "All" else None,
            slot=slot_sel if slot_sel != "All times" else None,
        )
        # Every scheduled/odds-matched game in the filtered range gets a card —
        # not just games with a bettable pick, so a game with no qualifying
        # edge (e.g. today's board with EV too thin) still shows up.
        games_shown = filter_picks(pred_all, slot_by_game, **_flt) if pred_all is not None and not pred_all.empty else pd.DataFrame()
        shown = filter_picks(picks, slot_by_game, **_flt)
        if games_shown.empty and shown.empty:
            st.info("No games match the selected filters.")
        else:
            total_games = pred_all["game_id"].nunique() if pred_all is not None and not pred_all.empty else picks["game_id"].nunique()
            n_shown = games_shown["game_id"].nunique() if not games_shown.empty else shown["game_id"].nunique()
            st.caption(f"Showing {n_shown} of {total_games} games "
                       f"({shown['game_id'].nunique()} with a pick clearing the edge/EV filter)")
            render_espn_picks(games_shown if not games_shown.empty else shown, shown, pred_all)

        st.subheader("Moneyline & Spread — model pick vs. EV pick")
        st.caption("For every game: the model's predicted side (win prob. / spread) next to the "
                   "single best-EV market price for that side. Ignores the min-edge slider, so you "
                   "can see a candidate even where it didn't clear the bettable threshold.")
        if not games_shown.empty:
            game_table = build_game_table(games_shown, st.session_state.get("picks_all"))
            st.dataframe(
                game_table, use_container_width=True, hide_index=True,
                column_config={
                    "week":              st.column_config.NumberColumn("Wk"),
                    "kickoff":           "Kickoff (ET)",
                    "matchup":           "Matchup",
                    "model_ml_pick":     "Model ML pick",
                    "ev_ml_pick":        "EV ML pick",
                    "model_spread_pick": "Model spread",
                    "ev_spread_pick":    "EV spread pick",
                },
            )

        with st.expander(f"Table view (top {top_n} of filtered picks)", expanded=False):
            st.dataframe(pick_top(shown, top_n), use_container_width=True,
                         column_config=PICKS_COLUMN_CONFIG, hide_index=True)

        with st.expander("🎲 Bankroll Simulation", expanded=False):
            st.caption(
                "Monte Carlo simulation over the picks in the table above: each trial redraws every "
                "pick's outcome from the model's own stated win probability (not a real result — these "
                "haven't been played yet), at its actual stake and price. Shows the range of outcomes "
                "the model's own confidence implies, not a prediction of what will happen."
            )
            sim_bets = pick_top(shown, top_n)
            if st.button("Run simulation (5,000 trials)", key="run_bankroll_sim"):
                sim = fn.simulate_bankroll(sim_bets, n_sims=5000, seed=None)
                st.session_state["bankroll_sim"] = sim
            sim = st.session_state.get("bankroll_sim")
            if sim:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Bets simulated", sim["n_bets"])
                c2.metric("Median outcome", f"${sim['pctiles'][50]:+,.0f}")
                c3.metric("Chance of profit", f"{sim['prob_positive']:.0%}")
                c4.metric("Typical worst dip", f"${sim['median_drawdown']:+,.0f}",
                          delta=f"5th pct: ${sim['worst_5pct_drawdown']:+,.0f}", delta_color="off")
                st.caption(
                    f"5th–95th percentile range: **${sim['pctiles'][5]:+,.0f}** to **${sim['pctiles'][95]:+,.0f}** "
                    f"on ${sim['total_staked']:,.0f} staked across {sim['n_bets']} bets "
                    f"(mean ${sim['mean_pl']:+,.0f})."
                )
                counts, edges = np.histogram(sim["season_pl_dist"], bins=40)
                hist_df = pd.DataFrame({"P&L ($)": edges[:-1].round(0), "Trials": counts}).set_index("P&L ($)")
                st.bar_chart(hist_df)
            elif sim_bets.empty:
                st.info("No priced picks available to simulate.")
            else:
                st.caption("Click the button to run the simulation.")

        with st.expander("📝 Share picks (tweets)", expanded=False):
            st.caption("Top 5 picks by EV for each week, formatted to post — click the copy icon "
                       "in the top-right of each box.")
            tweets_by_week = build_tweets(picks, per_week=5)
            if not tweets_by_week:
                st.info("No picks with a week number to build tweets from.")
            for wk, tweets in tweets_by_week.items():
                st.markdown(f"**Week {wk}**")
                for t in tweets:
                    st.code(t, language=None)
        st.download_button("Download picks CSV", picks.to_csv(index=False), "picks.csv", "text/csv")
        with st.expander("Predicted Scores", expanded=False):
            if pred_all is not None:
                pred_shown = pred_all[pred_all["game_id"].isin(shown["game_id"])] if not shown.empty else pred_all
                pred_cols = [c for c in ["week","kickoff","home_team","away_team","home_pts","away_pts","pred_total","pred_margin"]
                             if c in pred_shown.columns]
                st.dataframe(pred_shown[pred_cols],
                             use_container_width=True, hide_index=True,
                             column_config={c: st.column_config.NumberColumn(format="%.1f")
                                            for c in ["home_pts","away_pts","pred_total","pred_margin"]})

# ============================================================================
# TAB 2: SEASON TRACKER — predicted winner vs actual winner, this season
# ============================================================================

def _render_flat_bet_block(block: dict, label: str, stake_key: str, price_label: str,
                           per_game_cols: list, csv_name: str) -> None:
    """Render one flat-stake tracker block (moneyline OR spread) as its own
    self-contained section: metrics, weekly table, game-by-game expander."""
    stake = block.get(stake_key, 10.0)
    st.markdown(f"#### {label}")
    if not block.get("n_games"):
        st.info(f"No {label.lower()} games available.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Record (W-L-P)" if block.get("pushes") else "Record (W-L)",
              f"{block['wins']}-{block['losses']}" + (f"-{block['pushes']}" if block.get("pushes") else ""))
    c2.metric("Win %", f"{block['accuracy']:.1%}" if pd.notna(block['accuracy']) else "—")
    c3.metric("Games", block["n_games"])
    c4.metric(f"${stake:g} bet every game", f"{block['total_pl']:+.2f}",
              delta=f"{block['roi']:+.1%} ROI on {block['n_priced']} priced games" if block['n_priced'] else "no odds available",
              delta_color="off")

    st.dataframe(
        block["weekly"][["week","record","accuracy","games","pl"]],
        use_container_width=True, hide_index=True,
        column_config={
            "week":     st.column_config.NumberColumn("Week"),
            "record":   "Record",
            "accuracy": st.column_config.NumberColumn("Win %", format="percent"),
            "games":    st.column_config.NumberColumn("Games"),
            "pl":       st.column_config.NumberColumn(f"${stake:g}/game P&L", format="dollar"),
        },
    )
    with st.expander(f"{label} — game-by-game", expanded=False):
        pg = block["per_game"]
        cfg = {
            "week":         st.column_config.NumberColumn("Week"),
            "home_score":   st.column_config.NumberColumn("Home pts", format="%.0f"),
            "away_score":   st.column_config.NumberColumn("Away pts", format="%.0f"),
            "p_home_pred":  st.column_config.NumberColumn("Model P(home win)", format="percent"),
            "vegas_spread": st.column_config.NumberColumn("Closing spread (home)", format="%.1f"),
            "actual_margin": st.column_config.NumberColumn("Actual margin", format="%.0f"),
            "ml_price":     st.column_config.NumberColumn(price_label, format="%+d"),
            "spread_price": st.column_config.NumberColumn(price_label, format="%+d"),
            "bet_pl":       st.column_config.NumberColumn(f"${stake:g} bet P&L", format="dollar"),
        }
        st.dataframe(pg[[c for c in per_game_cols if c in pg.columns]], use_container_width=True,
                     hide_index=True, column_config=cfg)
        st.download_button(f"Download {label.lower()} tracker CSV", pg.to_csv(index=False),
                           csv_name, "text/csv", key=f"dl_{stake_key}")

with tab_tracker:
    st.subheader("Season-to-Date Tracker")
    st.caption(
        "Two independent trackers for every completed game this season: moneyline (predicted winner) "
        "and against-the-spread (predicted side of the closing line), each with its own flat-stake $ P&L "
        "at the real closing price. Kept separate since a game's ML and ATS picks can disagree. Predictions "
        "are computed walk-forward with in-season retraining (same process live picks use), so each game's "
        "prediction only ever used data available before it was played."
    )
    current_year = int(utc_now_year())
    c_season, c_ml_stake, c_sp_stake = st.columns(3)
    st_season   = c_season.number_input("Season", min_value=2000, max_value=2100, value=current_year, step=1, key="st_season")
    st_ml_stake = c_ml_stake.number_input("Moneyline bet size ($)", min_value=1.0, value=10.0, step=1.0, key="st_ml_stake")
    st_sp_stake = c_sp_stake.number_input("Spread bet size ($)", min_value=1.0, value=10.0, step=1.0, key="st_sp_stake")

    if st.button("Refresh Season Tracker", type="primary"):
        try:
            bt = backtest_for([int(st_season)])
            st.session_state["season_tracker"] = fn.summarize_season_to_date(
                bt, ml_stake=float(st_ml_stake), spread_stake=float(st_sp_stake))
            st.session_state["season_tracker_season"] = int(st_season)
        except Exception as e:
            st.error(f"Season tracker failed: {e}")
            st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))

    tracker = st.session_state.get("season_tracker")
    if tracker is None:
        st.info("Click **Refresh Season Tracker** to compute this season's record.")
    elif not tracker.get("moneyline", {}).get("n_games") and not tracker.get("spread", {}).get("n_games"):
        st.info(f"No completed games found for season {st.session_state.get('season_tracker_season', int(st_season))} yet.")
    else:
        yr = st.session_state.get('season_tracker_season', int(st_season))
        _render_flat_bet_block(
            tracker.get("moneyline", {}), "Moneyline", "ml_stake", "ML price",
            ["week","home_team","away_team","home_score","away_score",
             "p_home_pred","pred_winner","actual_winner","ml_result","ml_price","bet_pl"],
            f"season_tracker_moneyline_{yr}.csv",
        )
        st.divider()
        _render_flat_bet_block(
            tracker.get("spread", {}), "Against the Spread", "spread_stake", "Spread price",
            ["week","home_team","away_team","vegas_spread","picked_team","actual_margin",
             "ats_result","spread_price","bet_pl"],
            f"season_tracker_spread_{yr}.csv",
        )

# ============================================================================
# TAB 3: MODEL vs VEGAS
# ============================================================================
with tab_vegas:
    st.subheader("Model vs Vegas Closing Lines")
    st.caption(
        "Walk-forward predictions compared to the market: e.g. the model makes a team a 10-point "
        "favorite, Vegas closed them at −8, and they won by 11 — the model was closer (error 1.0 vs 3.0) "
        "and its side of the line covered. Lines come from nflverse; no API key needed."
    )
    current_year = int(utc_now_year())
    mv_seasons = st.multiselect(
        "Season(s) to evaluate",
        options=list(range(2018, current_year + 1)),
        default=[current_year - 1],
        key="mv_seasons",
    )
    if st.button("Evaluate vs Vegas", type="primary"):
        if not mv_seasons:
            st.warning("Select at least one season.")
        else:
            try:
                bt_df = backtest_for(mv_seasons)
                per_game, summary = fn.evaluate_vs_market(bt_df)
                if per_game.empty:
                    st.warning("No completed games with Vegas lines found for those seasons.")
                else:
                    st.session_state["mv_per_game"] = per_game
                    st.session_state["mv_summary"] = summary
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))

    summary = st.session_state.get("mv_summary")
    per_game = st.session_state.get("mv_per_game")
    if summary:
        sp, tt, ml = summary["spread"], summary["totals"], summary.get("moneyline", {})

        st.markdown("#### Spreads")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model margin MAE", f"{sp['model_mae']:.2f} pts")
        c2.metric("Vegas margin MAE", f"{sp['vegas_mae']:.2f} pts",
                  delta=f"{sp['model_mae']-sp['vegas_mae']:+.2f} model vs line", delta_color="inverse")
        c3.metric("Model closer than line", f"{sp['model_closer_pct']:.1%}")
        a = sp["ats"]
        c4.metric("Model ATS record", f"{a['wins']}-{a['losses']}-{a['pushes']}",
                  delta=(f"{a['win_pct']:.1%} (breakeven 52.4%)" if (a['wins']+a['losses']) else None),
                  delta_color="off")
        st.caption("ATS win rate by how much the model disagrees with the line (bigger claimed edge should win more):")
        st.dataframe(pd.DataFrame(sp["ats_by_disagreement"]), hide_index=True,
                     column_config={
                         "min_edge_pts": st.column_config.NumberColumn("Min disagreement (pts)", format="%.0f"),
                         "win_pct": st.column_config.NumberColumn("Win %", format="percent"),
                     })

        st.markdown("#### Totals")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model total MAE", f"{tt['model_mae']:.2f} pts")
        c2.metric("Vegas total MAE", f"{tt['vegas_mae']:.2f} pts",
                  delta=f"{tt['model_mae']-tt['vegas_mae']:+.2f} model vs line", delta_color="inverse")
        c3.metric("Model closer than line", f"{tt['model_closer_pct']:.1%}")
        o = tt["ou"]
        c4.metric("Model O/U record", f"{o['wins']}-{o['losses']}-{o['pushes']}",
                  delta=(f"{o['win_pct']:.1%}" if (o['wins']+o['losses']) else None), delta_color="off")

        if ml:
            st.markdown("#### Moneyline (probability quality)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Model Brier", f"{ml['model_brier']:.4f}")
            c2.metric("Vegas Brier", f"{ml['vegas_brier']:.4f}",
                      delta=f"{ml['model_brier']-ml['vegas_brier']:+.4f} model vs market", delta_color="inverse")
            c3.metric("Model accuracy", f"{ml['model_accuracy']:.1%}")
            c4.metric("Vegas accuracy", f"{ml['vegas_accuracy']:.1%}")

        with st.expander("Per-game detail", expanded=False):
            show_cols = ["season","week","home_team","away_team",
                         "pred_margin","vegas_spread","actual_margin",
                         "model_spread_error","vegas_spread_error","ats_pick","ats_result",
                         "pred_total","vegas_total","actual_total","ou_pick","ou_result"]
            det = per_game[[c for c in show_cols if c in per_game.columns]]
            st.dataframe(det, use_container_width=True, hide_index=True,
                         column_config={
                             "pred_margin":  st.column_config.NumberColumn("Model spread (home)", format="%.1f"),
                             "vegas_spread": st.column_config.NumberColumn("Vegas spread (home)", format="%.1f"),
                             "actual_margin": st.column_config.NumberColumn("Actual margin", format="%.0f"),
                             "model_spread_error": st.column_config.NumberColumn("Model err", format="%.1f"),
                             "vegas_spread_error": st.column_config.NumberColumn("Vegas err", format="%.1f"),
                             "pred_total":  st.column_config.NumberColumn("Model total", format="%.1f"),
                             "vegas_total": st.column_config.NumberColumn("Vegas total", format="%.1f"),
                             "actual_total": st.column_config.NumberColumn("Actual total", format="%.0f"),
                         })
            st.download_button("Download per-game CSV", per_game.to_csv(index=False),
                               "model_vs_vegas.csv", "text/csv")

# ============================================================================
# TAB 4: BACKTEST
# ============================================================================
with tab_backtest:
    st.subheader("Walk-Forward Backtest")
    st.caption(
        "For each selected evaluation season, models are trained on all *prior* seasons "
        "and evaluated on completed games in that season. This gives honest out-of-sample metrics."
    )
    current_year = int(utc_now_year())
    bt_eval_seasons = st.multiselect(
        "Evaluation season(s)",
        options=list(range(2018, current_year + 1)),
        default=[current_year - 1],
        key="bt_eval_seasons",
    )
    if st.button("Run Backtest"):
        if not bt_eval_seasons:
            st.warning("Select at least one evaluation season.")
        else:
            try:
                bt_df = backtest_for(bt_eval_seasons)
                if bt_df.empty:
                    st.warning("No completed games found. Try an older season.")
                else:
                    st.session_state["bt_df"] = bt_df
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))

    bt_df = st.session_state.get("bt_df")
    if bt_df is not None and not bt_df.empty:
        acc   = float(((bt_df["p_home_pred"] > 0.5) == bt_df["actual_home_win"]).mean())
        brier = float(((bt_df["p_home_pred"] - bt_df["actual_home_win"]) ** 2).mean())
        mae_m = float((bt_df["pred_margin"] - bt_df["actual_margin"]).abs().mean())
        mae_t = float((bt_df["pred_total"]  - bt_df["actual_total"]).abs().mean())

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Games evaluated", len(bt_df))
        c2.metric("Win pred accuracy", f"{acc:.1%}")
        c3.metric("Brier score", f"{brier:.4f}", help="Lower is better; 0.25 = random")
        c4.metric("Margin MAE", f"{mae_m:.1f} pts")
        c5.metric("Total MAE",  f"{mae_t:.1f} pts")

        _, bt_summary = fn.evaluate_vs_market(bt_df)
        u = bt_summary.get("units", {})
        if u:
            st.markdown("#### Scoreboard — 1 unit on every model pick")
            st.caption("Every game, flat 1-unit stakes, settled at nflverse closing prices "
                       "(−110 assumed where a price is missing). Spread/total picks are the model's "
                       "side of the closing line; moneyline picks are the model's predicted winner.")
            cols = st.columns(4)
            for col, key, name in [(cols[0], "spread", "Spread"), (cols[1], "totals", "Totals"),
                                   (cols[2], "moneyline", "Moneyline")]:
                b = u.get(key, {})
                if b.get("n_bets"):
                    col.metric(f"{name} P/L", f"{b['units']:+.1f} u",
                               delta=f"{b['wins']}-{b['losses']}-{b['pushes']} · {b['roi']:+.1%} ROI",
                               delta_color="off")
                else:
                    col.metric(f"{name} P/L", "—")
            total_u = u.get("total_units", 0.0)
            n_total = sum(u.get(k, {}).get("n_bets", 0) for k in ("spread","totals","moneyline"))
            cols[3].metric("Combined P/L", f"{total_u:+.1f} u",
                           delta=f"{n_total} bets · {total_u/n_total:+.1%} ROI" if n_total else None,
                           delta_color="off")

        with st.expander("Backtest detail (all games)", expanded=False):
            st.dataframe(bt_df, use_container_width=True, hide_index=True)

# ============================================================================
# TAB 5: DIAGNOSTICS
# ============================================================================
with tab_diag:
    st.subheader("Model Diagnostics")
    diag = st.session_state.get("last_diagnostics")
    if diag is None:
        st.info("Generate picks first to see model diagnostics here.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Training games",      diag["train_rows"])
        c2.metric("Margin σ (OOF)",      f"{diag['margin_sigma_oof']:.1f} pts")
        c3.metric("Margin CV MAE",       f"{diag['cv_mae_margin']:.2f} pts")
        c4.metric("Total CV MAE",        f"{diag['cv_mae_total']:.2f} pts")
        st.json({k: v for k, v in diag.items() if k not in
                 ("train_rows","margin_sigma_oof","cv_mae_margin","cv_mae_total")})
