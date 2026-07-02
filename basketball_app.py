"""
Basketball Picks — NBA & NCAAB
Streamlit app (parallel to app.py for the NFL tool).

Run:
  streamlit run basketball_app.py
"""
from __future__ import annotations

import datetime
import importlib
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

st.set_page_config(
    page_title="Basketball Picks — NBA & NCAAB",
    layout="wide",
    initial_sidebar_state="expanded",
)

fn = importlib.import_module("fetch_basketball_data")

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _utc_year() -> int:
    try:
        from datetime import UTC
        return datetime.datetime.now(UTC).year
    except Exception:
        return datetime.datetime.utcnow().year


def _pick_top(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    order_cols = [c for c in ["ev_per_usd", "edge_vs_book_devig", "odds_decimal"] if c in df.columns]
    if not order_cols:
        return df.head(n)
    return df.sort_values(order_cols, ascending=[False] * len(order_cols)).head(n)


def _coalesce_team_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for side in ["home", "away"]:
        base = f"{side}_team"; x = f"{base}_x"; y = f"{base}_y"
        series = d.get(base)
        if x in d.columns:
            series = d[x] if series is None else series.where(series.notna(), d[x])
        if y in d.columns:
            series = d[y] if series is None else series.where(series.notna(), d[y])
        if series is not None:
            d[base] = series
        for c in (x, y):
            if c in d.columns:
                d.drop(columns=c, inplace=True)
    return d


def _format_picks(df: pd.DataFrame) -> pd.DataFrame:
    """Rename and format picks columns for clean display."""
    d = df.copy()

    # Percentage columns
    for col, label in [("p_true", "Model %"), ("book_p_devig", "Book %"), ("edge_vs_book_devig", "Edge")]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce").mul(100).round(1).astype(str) + "%"
            d = d.rename(columns={col: label})

    # Numeric rounding
    for col, label, dec in [
        ("ev_per_usd", "EV/$", 3),
        ("odds_decimal", "Odds", 2),
        ("kelly_frac",  "Kelly%", 3),
        ("stake_usd",   "Stake $", 2),
        ("line",        "Line", 1),
    ]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce").round(dec)
            d = d.rename(columns={col: label})

    # Label columns
    for col, label in [
        ("market_key",  "Market"),
        ("side",        "Side"),
        ("book_title",  "Book"),
        ("odds_american", "American Odds"),
        ("season",      "Season"),
    ]:
        if col in d.columns:
            d = d.rename(columns={col: label})

    # Preferred column order
    front = ["matchup", "date", "Market", "Side", "Line", "American Odds", "Odds",
             "Model %", "Book %", "Edge", "EV/$", "Kelly%", "Stake $", "Book"]
    ordered = [c for c in front if c in d.columns]
    rest    = [c for c in d.columns if c not in ordered]
    return d[ordered + rest]


# ─────────────────────────────────────────────────────────────────────────────
# March Madness bracket helpers
# ─────────────────────────────────────────────────────────────────────────────

_MM_REGIONS   = ["East", "West", "Midwest", "South"]

_MM_2026_BRACKET: dict = {
    "East": [
        "Duke Blue Devils",          # 1
        "UConn Huskies",             # 2
        "Michigan State Spartans",   # 3
        "Kansas Jayhawks",           # 4
        "St. John's Red Storm",      # 5
        "Louisville Cardinals",      # 6
        "UCLA Bruins",               # 7
        "Ohio State Buckeyes",       # 8
        "TCU Horned Frogs",          # 9
        "UCF Knights",               # 10
        "South Florida Bulls",       # 11
        "Northern Iowa Panthers",    # 12
        "California Baptist Lancers",# 13
        "North Dakota State Bison",  # 14
        "Furman Paladins",           # 15
        "Siena Saints",              # 16
    ],
    "West": [
        "Arizona Wildcats",          # 1
        "Purdue Boilermakers",       # 2
        "Gonzaga Bulldogs",          # 3
        "Arkansas Razorbacks",       # 4
        "Wisconsin Badgers",         # 5
        "BYU Cougars",               # 6
        "Miami Hurricanes",          # 7
        "Villanova Wildcats",        # 8
        "Utah State Aggies",         # 9
        "Missouri Tigers",           # 10
        "Texas Longhorns",           # 11 (First Four: Texas/NC State — update after play-in)
        "High Point Panthers",       # 12
        "Hawaii Rainbow Warriors",   # 13
        "Kennesaw State Owls",       # 14
        "Queens University Royals",  # 15
        "Long Island University Sharks", # 16
    ],
    "Midwest": [
        "Michigan Wolverines",       # 1
        "Iowa State Cyclones",       # 2
        "Virginia Cavaliers",        # 3
        "Alabama Crimson Tide",      # 4
        "Texas Tech Red Raiders",    # 5
        "Tennessee Volunteers",      # 6
        "Kentucky Wildcats",         # 7
        "Georgia Bulldogs",          # 8
        "Saint Louis Billikens",     # 9
        "Santa Clara Broncos",       # 10
        "SMU Mustangs",              # 11 (First Four: SMU/Miami (OH) — update after play-in)
        "Akron Zips",                # 12
        "Hofstra Pride",             # 13
        "Wright State Raiders",      # 14
        "Tennessee State Tigers",    # 15
        "UMBC Retrievers",           # 16 (First Four: UMBC/Howard — update after play-in)
    ],
    "South": [
        "Florida Gators",            # 1
        "Houston Cougars",           # 2
        "Illinois Fighting Illini",  # 3
        "Nebraska Cornhuskers",      # 4
        "Vanderbilt Commodores",     # 5
        "North Carolina Tar Heels",  # 6
        "Saint Mary's Gaels",        # 7
        "Clemson Tigers",            # 8
        "Iowa Hawkeyes",             # 9
        "Texas A&M Aggies",          # 10
        "VCU Rams",                  # 11
        "McNeese Cowboys",           # 12
        "Troy Trojans",              # 13
        "Pennsylvania Quakers",      # 14
        "Idaho Vandals",             # 15
        "Prairie View A&M Panthers", # 16 (First Four: Prairie View/Lehigh — update after play-in)
    ],
}
_FF_PAIRINGS  = [("East", "West"), ("South", "Midwest")]
# Round-1 seed matchups; index k → seed k+1 (0-based)
_R1_PAIRS     = [(0,15),(7,8),(4,11),(3,12),(5,10),(2,13),(6,9),(1,14)]


def _elo_p(elo_a: float, elo_b: float) -> float:
    """P(A beats B) on neutral court."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def _pick(a, b, elo_a, elo_b, stochastic=True):
    p = _elo_p(elo_a, elo_b)
    if stochastic:
        return a if np.random.random() < p else b
    return a if p >= 0.5 else b


def _sim_region(teams, elos, default, stochastic=True):
    """Simulate a 16-team region. Returns (champ, {round: [(a,b,winner), ...]})."""
    slots = list(teams)
    get = lambda t: elos.get(t, default) if t else default

    def play(a, b):
        if not a: return b
        if not b: return a
        return _pick(a, b, get(a), get(b), stochastic)

    # Round of 64
    r64w = [play(slots[ai], slots[bi]) for ai, bi in _R1_PAIRS]
    r64  = [(slots[ai], slots[bi], r64w[i]) for i, (ai, bi) in enumerate(_R1_PAIRS)]

    # Round of 32
    r32w = [play(r64w[i], r64w[i+1]) for i in range(0, 8, 2)]
    r32  = [(r64w[i], r64w[i+1], r32w[i//2]) for i in range(0, 8, 2)]

    # Sweet 16
    s16w = [play(r32w[i], r32w[i+1]) for i in range(0, 4, 2)]
    s16  = [(r32w[i], r32w[i+1], s16w[i//2]) for i in range(0, 4, 2)]

    # Elite 8
    champ = play(s16w[0], s16w[1]) if len(s16w) >= 2 else (s16w[0] if s16w else "")
    e8    = [(s16w[0], s16w[1], champ)] if len(s16w) >= 2 else []

    return champ, {"Round of 64": r64, "Round of 32": r32, "Sweet 16": s16, "Elite 8": e8}


def _simulate_tournament(bracket, elos, n=1000, default_elo=1500.0):
    """Monte Carlo tournament simulation. Returns counts dict with keys r32/s16/e8/ff/f2/champ."""
    counts: dict = {k: {} for k in ("r32", "s16", "e8", "ff", "f2", "champ")}

    def inc(key, team, amt=1):
        if team:
            counts[key][team] = counts[key].get(team, 0) + amt

    for _ in range(n):
        reg_champs = {}
        for region in _MM_REGIONS:
            teams = bracket.get(region, [""] * 16)
            if not any(t for t in teams if t):
                continue
            slots = list(teams)
            get = lambda t: elos.get(t, default_elo) if t else default_elo

            def play(a, b):
                if not a: return b
                if not b: return a
                return _pick(a, b, get(a), get(b), stochastic=True)

            r64w = [play(slots[ai], slots[bi]) for ai, bi in _R1_PAIRS]
            for w in r64w: inc("r32", w)

            r32w = [play(r64w[i], r64w[i+1]) for i in range(0, 8, 2)]
            for w in r32w: inc("s16", w)

            s16w = [play(r32w[i], r32w[i+1]) for i in range(0, 4, 2)]
            for w in s16w: inc("e8", w)

            champ = play(s16w[0], s16w[1]) if len(s16w) >= 2 else (s16w[0] if s16w else "")
            inc("ff", champ)
            reg_champs[region] = champ

        ff_winners = []
        for r1, r2 in _FF_PAIRINGS:
            a, b = reg_champs.get(r1, ""), reg_champs.get(r2, "")
            if not a and not b: continue
            w = (a or b) if not (a and b) else _pick(a, b, elos.get(a, default_elo), elos.get(b, default_elo))
            ff_winners.append(w)

        if len(ff_winners) == 2:
            a, b = ff_winners
            inc("f2", a); inc("f2", b)
            w = _pick(a, b, elos.get(a, default_elo), elos.get(b, default_elo))
            inc("champ", w)

    return counts


def _best_bracket(bracket, elos, default_elo=1500.0):
    """Deterministic best-pick bracket. Returns {regions, region_champs, final_four, championship}."""
    region_results, region_champs = {}, {}
    for region in _MM_REGIONS:
        teams = bracket.get(region, [""] * 16)
        if not any(t for t in teams if t):
            continue
        champ, results = _sim_region(teams, elos, default_elo, stochastic=False)
        region_results[region] = results
        region_champs[region] = champ

    ff_results, ff_winners = [], []
    for r1, r2 in _FF_PAIRINGS:
        a, b = region_champs.get(r1, ""), region_champs.get(r2, "")
        if not a and not b: continue
        w = (a or b) if not (a and b) else _pick(a, b, elos.get(a, default_elo), elos.get(b, default_elo), stochastic=False)
        ff_winners.append(w)
        ff_results.append((a, b, w, f"{r1} vs {r2}"))

    champ_result = None
    if len(ff_winners) == 2:
        a, b = ff_winners
        w = _pick(a, b, elos.get(a, default_elo), elos.get(b, default_elo), stochastic=False)
        champ_result = (a, b, w)

    return {"regions": region_results, "region_champs": region_champs,
            "final_four": ff_results, "championship": champ_result}


# ─────────────────────────────────────────────────────────────────────────────
# Cached data / model loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Loading schedule and results...")
def _load_data(sport: str, season_list_tuple: tuple, cache_dir: str) -> pd.DataFrame:
    sched = fn.load_schedule(sport, list(season_list_tuple), cache_dir=cache_dir, include_upcoming=True)
    return sched


@st.cache_data(ttl=3600, show_spinner="Loading advanced team stats...")
def _load_advanced_stats(sport: str, season_list_tuple: tuple) -> pd.DataFrame:
    if sport != "nba":
        return pd.DataFrame()
    return fn.load_nba_advanced_stats(list(season_list_tuple))


@st.cache_data(ttl=900, show_spinner="Fetching injury report...")
def _load_injuries(sport: str) -> pd.DataFrame:
    try:
        return fn.fetch_injury_report(sport)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner="Training models...")
def _fit_models(
    sport: str,
    train_df: pd.DataFrame,
    default_mu_total: float,
    default_sigma_margin: float,
    default_sigma_total: float,
) -> tuple:
    model_ml,   cv_acc              = fn.fit_logistic_bball(train_df)
    margin_mod, s_margin, cv_mae_mm = fn.fit_margin_model_bball(
        train_df, default_sigma=default_sigma_margin
    )
    total_mod,  s_total,  cv_mae_tm = fn.fit_total_model_bball(
        train_df, default_mu=default_mu_total, default_sigma=default_sigma_total
    )
    return model_ml, cv_acc, margin_mod, s_margin, cv_mae_mm, total_mod, s_total, cv_mae_tm


@st.cache_data(ttl=300, show_spinner="Fetching live odds...")
def _fetch_odds(api_key: str, sport: str, markets_tuple: tuple, books: str) -> pd.DataFrame:
    api_map = {"moneyline": "h2h", "spreads": "spreads", "totals": "totals"}
    frames = []
    for mk in markets_tuple:
        try:
            od = fn.fetch_odds_basketball(api_key, sport=sport, market=api_map[mk], bookmakers=books or None)
            if isinstance(od, pd.DataFrame) and not od.empty:
                frames.append(od)
        except Exception as e:
            st.warning(f"Could not fetch {mk} odds: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Controls")

    st.subheader("Sport & Season")
    sport_choice = st.selectbox("Sport", ["NBA", "NCAAB"], key="sport_choice")
    sport = sport_choice.lower()
    defs  = fn.SPORT_DEFAULTS[sport]

    current_year = _utc_year()
    season_label = f"{current_year - 1}–{str(current_year)[-2:]}"
    season_pick  = st.number_input(
        f"Season end-year (e.g. {current_year} = {season_label})",
        min_value=2015, max_value=current_year + 1,
        value=current_year, step=1, key="season_input",
    )

    st.subheader("Odds & Markets")
    _env_key = os.getenv("THE_ODDS_API_KEY", "")
    api_key  = st.text_input("The Odds API key", value=_env_key, type="password", key="api_key")
    books    = st.text_input("Bookmakers (comma-separated)", value="fanduel,draftkings,betmgm", key="books_text")
    markets_choice = st.multiselect(
        "Markets", ["moneyline", "spreads", "totals"],
        default=["moneyline", "spreads", "totals"], key="markets_multiselect",
    )

    st.subheader("Bet Sizing")
    kelly_bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=10000.0, step=100.0)
    kelly_frac     = st.slider("Kelly fraction", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
    min_edge       = st.slider(
        "Min edge vs book",
        min_value=0.0, max_value=0.15, value=0.03, step=0.01,
        help="Hide picks where model edge over devigged book probability is below this threshold",
        format="%.0f%%",
    )
    top_n = st.slider("Show top N picks", min_value=5, max_value=50, value=25, step=5)

    st.subheader("Advanced")
    cache_dir = st.text_input("Cache dir", value="./data/bball_cache", key="cache_dir")

    if sport == "ncaab":
        st.caption(
            "**NCAAB note:** Schedule data is fetched day-by-day from ESPN. "
            "First load may take 1–3 min; results are cached locally."
        )

# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────

sport_label = "NBA" if sport == "nba" else "NCAA Basketball"
today_str   = datetime.date.today().strftime("%B %-d, %Y")
st.title(f"{sport_label} Picks")
st.caption(f"{season_label} Season  ·  {today_str}")
st.divider()

tab_picks, tab_backtest, tab_diag, tab_bracket = st.tabs(["Picks", "Backtest", "Diagnostics", "March Madness"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PICKS
# ─────────────────────────────────────────────────────────────────────────────
with tab_picks:
    with st.form("picks_form", clear_on_submit=False):
        submitted = st.form_submit_button("Generate Picks", type="primary", use_container_width=True)

    if submitted:
        try:
            train_seasons: List[int] = [int(season_pick) - 3, int(season_pick) - 2, int(season_pick) - 1]
            season_list:   List[int] = sorted(set(train_seasons + [int(season_pick)]))

            # ── Load schedule ──────────────────────────────────────────────
            with st.spinner(f"Loading {sport.upper()} data for seasons {season_list}..."):
                sched_all = _load_data(sport, tuple(season_list), cache_dir)

            if sched_all.empty:
                st.error("No schedule data could be loaded. Check your internet connection.")
                st.stop()

            # ── Build features ─────────────────────────────────────────────
            team_stats = fn.build_team_rolling_stats(sched_all)
            adv_stats  = _load_advanced_stats(sport, tuple(season_list))
            feats_all  = fn.build_bball_game_features(
                sched_all, team_stats, pyth_exp=defs["pyth_exp"], advanced_stats=adv_stats
            )
            train_mask = feats_all["season"].isin(train_seasons)
            train_df   = fn.make_training_set_bball(feats_all.loc[train_mask].copy())

            # ── D-I filter for NCAAB ───────────────────────────────────────
            # Remove teams with < 15 games in a season (non-D-I or exhibition noise)
            if sport == "ncaab" and "home_team" in train_df.columns and "away_team" in train_df.columns:
                home_counts = train_df.groupby(["season", "home_team"]).size().reset_index(name="n")
                away_counts = train_df.groupby(["season", "away_team"]).size().reset_index(name="n")
                home_counts = home_counts.rename(columns={"home_team": "team"})
                away_counts = away_counts.rename(columns={"away_team": "team"})
                game_counts = pd.concat([home_counts, away_counts]).groupby(["season", "team"])["n"].sum()
                d1_teams = game_counts[game_counts >= 15].reset_index()["team"].unique()
                train_df = train_df[
                    train_df["home_team"].isin(d1_teams) & train_df["away_team"].isin(d1_teams)
                ].copy()

            # ── NCAAB: narrow training data to March Madness teams if entered ──
            if sport == "ncaab":
                mm_teams_raw = [
                    t.strip()
                    for teams in st.session_state.get("mm_bracket", {}).values()
                    for t in teams if t and t.strip()
                ]
                if mm_teams_raw:
                    mm_canon = {fn.canonical_ncaab_team(t) for t in mm_teams_raw}
                    mm_mask = (
                        train_df["home_team"].isin(mm_canon) |
                        train_df["away_team"].isin(mm_canon)
                    )
                    if mm_mask.any():
                        train_df = train_df[mm_mask].copy().reset_index(drop=True)
                        st.caption(f"Training on {len(mm_canon)} March Madness teams — {len(train_df):,} training rows.")

            # ── Recency weighting: boost the most recent training season ───
            # Duplicate rows by weight: season-1 = 1x, season-2 = 2x, season-3 = 4x
            # (most recent gets the highest repetition)
            if "season" in train_df.columns and len(train_seasons) > 1:
                sorted_seasons = sorted(train_seasons)
                max_s = max(sorted_seasons)
                boosts = {s: 2 ** (max_s - s) for s in sorted_seasons}
                # Cap at 4x to avoid overwhelming the earlier seasons entirely
                boosts = {s: min(v, 4) for s, v in boosts.items()}
                parts = [train_df]
                for s, extra_copies in boosts.items():
                    if extra_copies > 1:
                        chunk = train_df[train_df["season"] == s]
                        parts.extend([chunk] * (extra_copies - 1))
                train_df = pd.concat(parts, ignore_index=True)

            # ── Train models ───────────────────────────────────────────────
            (model_ml, cv_acc,
             margin_mod, s_margin, cv_mae_mm,
             total_mod,  s_total,  cv_mae_tm) = _fit_models(
                sport, train_df,
                defs["default_mu_total"],
                defs["default_sigma_margin"],
                defs["default_sigma_total"],
            )

            # ── Fetch odds ─────────────────────────────────────────────────
            if not api_key:
                st.error("Enter your **The Odds API** key in the sidebar.")
                st.stop()

            odds_raw = _fetch_odds(api_key, sport, tuple(markets_choice), books)
            if odds_raw.empty:
                st.warning("No odds returned. Check API key, bookmakers, or sport availability.")
                st.stop()
            odds_live = fn.prep_odds_bball(odds_raw, sport=sport)

            # ── Match odds to schedule ─────────────────────────────────────
            sched_target = sched_all[sched_all["season"] == int(season_pick)].copy()
            merged = fn.merge_odds_with_schedule_bball(odds_live, sched_target)

            if merged.empty:
                st.error("No market rows matched the schedule.")
                with st.expander("Matching diagnostics"):
                    def _pairs(df):
                        if df is None or df.empty: return set()
                        h, a = "home_team", "away_team"
                        if h not in df or a not in df: return set()
                        return {"|".join(sorted([r[h], r[a]])) for _, r in df[[h, a]].dropna().iterrows()}
                    odds_pairs  = _pairs(odds_live)
                    sched_pairs = _pairs(sched_target)
                    st.write("Odds pairs not in schedule:", sorted(odds_pairs - sched_pairs)[:40])
                    st.write("Schedule pairs not in odds:", sorted(sched_pairs - odds_pairs)[:40])
                st.stop()

            merged = _coalesce_team_cols(merged)

            # ── Attach features ────────────────────────────────────────────
            game_ids     = merged["game_id"].unique()
            feats_target = feats_all[feats_all["game_id"].isin(game_ids)].copy()
            present      = [c for c in fn.BBALL_FEATURE_COLS if c in feats_target.columns]
            if present:
                merged = merged.merge(
                    feats_target[["game_id"] + present].drop_duplicates("game_id"),
                    on="game_id", how="left",
                )
            merged = _coalesce_team_cols(merged)

            game_feats = merged.drop_duplicates(subset=["game_id"]).copy().reset_index(drop=True)
            feats_lookup = feats_all.drop_duplicates("game_id").set_index("game_id")
            for c in fn.BBALL_FEATURE_COLS:
                if c in feats_lookup.columns:
                    game_feats[c] = game_feats["game_id"].map(feats_lookup[c])
                elif c not in game_feats.columns:
                    game_feats[c] = np.nan

            X_full = game_feats[fn.BBALL_FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
            feature_means = train_df[fn.BBALL_FEATURE_COLS].apply(pd.to_numeric, errors="coerce").mean()
            X_full = X_full.fillna(feature_means)

            X_logit  = fn._align_features(model_ml,  X_full)
            X_margin = fn._align_features(margin_mod, X_full)
            X_total  = fn._align_features(total_mod,  X_full)

            p_home    = fn.predict_home_prob(model_ml, X_logit)
            mu_margin = margin_mod.predict(X_margin) if hasattr(margin_mod, "predict") else np.zeros(len(game_feats))
            mu_total  = total_mod.predict(X_total)   if hasattr(total_mod,  "predict") else np.full(len(game_feats), defs["default_mu_total"])

            p_home_df     = pd.DataFrame({"game_id": game_feats["game_id"].values, "p_home_model": p_home})
            mu_margin_map = dict(zip(game_feats["game_id"].values, mu_margin))
            mu_total_map  = dict(zip(game_feats["game_id"].values, mu_total))

            picks_df = fn.assemble_picks_bball(
                merged, markets_choice, p_home_df,
                mu_margin_map, mu_total_map,
                s_margin, s_total,
                float(kelly_bankroll), float(kelly_frac),
            )

            if picks_df.empty:
                st.warning("No candidate picks produced for the selected markets.")
                st.stop()

            if "edge_vs_book_devig" in picks_df.columns and min_edge > 0:
                picks_df = picks_df[picks_df["edge_vs_book_devig"] >= min_edge].copy()
            if picks_df.empty:
                st.warning(f"No picks pass the {min_edge:.0%} edge filter. Lower it in the sidebar.")
                st.stop()

            # ── Predicted scores ───────────────────────────────────────────
            pred = pd.DataFrame({
                "game_dt":     game_feats.get("game_dt",   pd.Series(dtype="datetime64[ns, UTC]")),
                "home_team":   game_feats.get("home_team", pd.Series(dtype=str)),
                "away_team":   game_feats.get("away_team", pd.Series(dtype=str)),
                "pred_margin": mu_margin.round(1),
                "pred_total":  mu_total.round(1),
            })
            pred["home_pts"] = ((pred["pred_total"] + pred["pred_margin"]) / 2.0).round(1)
            pred["away_pts"] = (pred["pred_total"] - pred["home_pts"]).round(1)
            for col in ["home_team", "away_team"]:
                pred[col] = pred[col].apply(lambda x: fn.display_team_name(x, sport))
            if "game_dt" in pred.columns:
                pred["date"] = pd.to_datetime(pred["game_dt"], errors="coerce", utc=True).dt.strftime("%b %-d")
                pred = pred.drop(columns=["game_dt"])
            pred = pred.rename(columns={
                "home_team": "Home", "away_team": "Away",
                "home_pts": "Home Pts", "away_pts": "Away Pts",
                "pred_total": "Total", "pred_margin": "Margin",
            })

            # ── Build display picks ────────────────────────────────────────
            display_picks = picks_df.copy()
            for col in ["home_team", "away_team"]:
                if col in display_picks.columns:
                    display_picks[col] = display_picks[col].apply(lambda x: fn.display_team_name(x, sport))
            if "away_team" in display_picks.columns and "home_team" in display_picks.columns:
                if "game_dt" in display_picks.columns:
                    display_picks["date"] = pd.to_datetime(
                        display_picks["game_dt"], errors="coerce", utc=True
                    ).dt.strftime("%b %-d")
                    display_picks = display_picks.drop(columns=["game_dt"], errors="ignore")
                display_picks.insert(0, "matchup", display_picks["away_team"] + " @ " + display_picks["home_team"])
                display_picks = display_picks.drop(columns=["game_id", "home_team", "away_team"], errors="ignore")
                if "date" in display_picks.columns:
                    display_picks.insert(1, "date", display_picks.pop("date"))

            display_picks = _format_picks(display_picks)
            top_picks = _pick_top(display_picks, top_n)

            # ── Summary metrics ────────────────────────────────────────────
            n_picks  = len(top_picks)
            n_games  = picks_df["game_id"].nunique() if "game_id" in picks_df.columns else "—"
            avg_edge = picks_df["edge_vs_book_devig"].mean() if "edge_vs_book_devig" in picks_df.columns else 0
            avg_ev   = picks_df["ev_per_usd"].mean()         if "ev_per_usd"          in picks_df.columns else 0
            total_stake = picks_df["stake_usd"].sum()        if "stake_usd"           in picks_df.columns else 0

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Picks found",   n_picks)
            m2.metric("Games",         n_games)
            m3.metric("Avg edge",      f"{avg_edge:.1%}")
            m4.metric("Avg EV/$",      f"{avg_ev:.3f}")
            m5.metric("Total stake",   f"${total_stake:,.0f}")

            st.divider()

            # ── Picks table ────────────────────────────────────────────────
            st.subheader("Top Picks")
            st.dataframe(
                top_picks,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "matchup":  st.column_config.TextColumn("Matchup", width="large",
                        help="Away team @ Home team for this game."),
                    "date":     st.column_config.TextColumn("Date", width="small",
                        help="Scheduled game date."),
                    "Market":   st.column_config.TextColumn("Market", width="small",
                        help="Bet type: moneyline (h2h), spread, or totals (over/under)."),
                    "Side":     st.column_config.TextColumn("Side", width="small",
                        help="Which side of the market to bet (team name, Over, or Under)."),
                    "Line":     st.column_config.NumberColumn("Line", format="%.1f",
                        help="The point spread or total line offered by the book."),
                    "Odds":     st.column_config.NumberColumn("Odds", format="%.2f",
                        help="Best available decimal odds across all books (e.g. 1.91 = -110 American)."),
                    "Model %":  st.column_config.TextColumn("Model %", width="small",
                        help="Probability assigned by the XGBoost model for this outcome."),
                    "Book %":   st.column_config.TextColumn("Book %", width="small",
                        help="Implied probability from the book's odds after removing the vig (overround)."),
                    "Edge":     st.column_config.TextColumn("Edge", width="small",
                        help="Model probability minus devigged book probability. Higher = more value found."),
                    "EV/$":     st.column_config.NumberColumn("EV/$", format="%.3f",
                        help="Expected profit per dollar wagered. Positive means the bet has long-run value."),
                    "Kelly%":   st.column_config.NumberColumn("Kelly%", format="%.3f",
                        help="Recommended fraction of bankroll to wager per the Kelly criterion, scaled by your Kelly fraction setting."),
                    "Stake $":  st.column_config.NumberColumn("Stake $", format="$%.2f",
                        help="Dollar amount to wager based on your bankroll and Kelly fraction."),
                    "Book":     st.column_config.TextColumn("Book", width="small",
                        help="Sportsbook offering the best odds for this pick."),
                },
            )

            # ── Predicted scores ───────────────────────────────────────────
            st.divider()
            st.subheader("Predicted Scores")
            pred_cols = [c for c in ["date", "Away", "Home", "Away Pts", "Home Pts", "Total", "Margin"] if c in pred.columns]
            st.dataframe(pred[pred_cols], use_container_width=True, hide_index=True)

            # ── Injury report ─────────────────────────────────────────────
            st.divider()
            st.subheader("Injury Report")
            injuries = _load_injuries(sport)
            if injuries.empty:
                st.caption("No injury data available.")
            else:
                # Filter to teams in today's games
                active_teams = set(
                    game_feats.get("home_team", pd.Series(dtype=str)).tolist() +
                    game_feats.get("away_team", pd.Series(dtype=str)).tolist()
                )
                inj_today = injuries[injuries["team"].isin(active_teams)].copy() if active_teams else injuries.copy()
                inj_display = inj_today if not inj_today.empty else injuries
                inj_display = inj_display.copy()
                inj_display["team"] = inj_display["team"].apply(lambda x: fn.display_team_name(x, sport))
                inj_display = inj_display.rename(columns={
                    "team": "Team", "player": "Player", "position": "Pos",
                    "status": "Status", "details": "Details",
                })
                for team_name, group in inj_display.groupby("Team", sort=True):
                    with st.expander(team_name, expanded=True):
                        st.dataframe(
                            group[["Player", "Pos", "Status", "Details"]].reset_index(drop=True),
                            use_container_width=True, hide_index=True,
                        )

            # ── Stash diagnostics ─────────────────────────────────────────
            st.session_state["last_bball_diagnostics"] = {
                "sport":               sport.upper(),
                "season":              int(season_pick),
                "train_rows":          int(len(train_df)),
                "logit_cv_accuracy":   float(cv_acc),
                "margin_sigma_oof":    float(s_margin),
                "total_sigma_oof":     float(s_total),
                "cv_mae_margin":       float(cv_mae_mm),
                "cv_mae_total":        float(cv_mae_tm),
                "logit_features":      list(map(str, fn.model_used_features(model_ml))),
                "margin_features":     list(map(str, fn.model_used_features(margin_mod))),
                "total_features":      list(map(str, fn.model_used_features(total_mod))),
                "games_with_odds":     int(len(game_feats)),
            }

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            with st.expander("Traceback", expanded=True):
                st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
with tab_backtest:
    st.subheader("Walk-Forward Backtest")
    st.caption(
        "For each evaluation season, models are trained on all **prior** seasons "
        "and evaluated on completed games in that season."
    )

    if sport == "ncaab":
        st.info("NCAAB backtesting fetches ESPN data day-by-day. Results are cached after the first run.")

    available_start = 2018 if sport == "nba" else 2019
    bt_eval_seasons = st.multiselect(
        "Evaluation season(s) (end-year)",
        options=list(range(available_start, current_year + 1)),
        default=[current_year - 1],
        key="bt_eval_seasons",
    )

    if st.button("Run Backtest", key="bt_run", type="primary"):
        if not bt_eval_seasons:
            st.warning("Select at least one evaluation season.")
        else:
            try:
                eval_s      = sorted(bt_eval_seasons)
                all_bt_seas = [s for s in range(min(eval_s) - 3, max(eval_s) + 1) if s >= 2015]

                with st.spinner(f"Loading {sport.upper()} schedule ({all_bt_seas})..."):
                    sched_bt = _load_data(sport, tuple(all_bt_seas), cache_dir)

                bt_adv = _load_advanced_stats(sport, tuple(all_bt_seas))
                with st.spinner("Running walk-forward evaluation..."):
                    bt_df = fn.run_backtest_basketball(
                        sched_bt, eval_s, sport=sport, advanced_stats=bt_adv
                    )

                if bt_df.empty:
                    st.warning("No completed games found. Try a different season range.")
                else:
                    n     = len(bt_df)
                    acc   = float(((bt_df["p_home_pred"] > 0.5) == bt_df["actual_home_win"]).mean())
                    brier = float(((bt_df["p_home_pred"] - bt_df["actual_home_win"]) ** 2).mean())
                    mae_m = float((bt_df["pred_margin"] - bt_df["actual_margin"]).abs().mean())
                    mae_t = float((bt_df["pred_total"]  - bt_df["actual_total"]).abs().mean())

                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Games evaluated", f"{n:,}")
                    c2.metric("Win accuracy",    f"{acc:.1%}")
                    c3.metric("Brier score",     f"{brier:.4f}", help="Lower is better; 0.25 = random")
                    c4.metric("Margin MAE",      f"{mae_m:.1f} pts")
                    c5.metric("Total MAE",       f"{mae_t:.1f} pts")

                    with st.expander("Full backtest results"):
                        st.dataframe(bt_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
with tab_diag:
    st.subheader("Model Diagnostics")
    diag = st.session_state.get("last_bball_diagnostics")
    if diag is None:
        st.info("Generate picks first to populate diagnostics.")
    else:
        st.caption(f"**{diag['sport']}**  ·  Season {diag['season']}")
        st.divider()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Training rows",    f"{diag['train_rows']:,}")
        c2.metric("Win CV accuracy",  f"{diag['logit_cv_accuracy']:.1%}")
        c3.metric("Margin CV MAE",    f"{diag['cv_mae_margin']:.2f} pts")
        c4.metric("Total CV MAE",     f"{diag['cv_mae_total']:.2f} pts")

        c5, c6, c7 = st.columns(3)
        c5.metric("Games with odds",  diag["games_with_odds"])
        c6.metric("Margin σ (OOF)",   f"{diag['margin_sigma_oof']:.2f} pts")
        c7.metric("Total σ (OOF)",    f"{diag['total_sigma_oof']:.2f} pts")

        st.divider()
        st.caption("Features used by each model")
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.markdown("**Moneyline**")
            for f in diag["logit_features"]:
                st.caption(f)
        with fc2:
            st.markdown("**Spread**")
            for f in diag["margin_features"]:
                st.caption(f)
        with fc3:
            st.markdown("**Totals**")
            for f in diag["total_features"]:
                st.caption(f)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — MARCH MADNESS BRACKET
# ─────────────────────────────────────────────────────────────────────────────
with tab_bracket:
    st.subheader("March Madness Bracket Simulator")

    if sport != "ncaab":
        st.info("Switch to **NCAA Basketball** in the sidebar to use the bracket simulator.")
    else:
        st.caption(
            "Enter the 64 tournament teams by region and seed. "
            "Win probabilities are estimated from ELO ratings computed on the loaded season schedule. "
            "All games are treated as neutral-site."
        )

        # ── Bracket entry ──────────────────────────────────────────────────
        if "mm_bracket" not in st.session_state:
            st.session_state.mm_bracket = {r: list(_MM_2026_BRACKET[r]) for r in _MM_REGIONS}

        if st.button("Reset to 2026 Field", key="mm_reset"):
            st.session_state.mm_bracket = {r: list(_MM_2026_BRACKET[r]) for r in _MM_REGIONS}
            st.rerun()

        st.markdown("### 2026 Bracket (seed 1 → 16 per region)")
        st.caption("Edit any name to override. Team names are matched case-insensitively to schedule data.")

        entry_cols = st.columns(4)
        for ci, region in enumerate(_MM_REGIONS):
            with entry_cols[ci]:
                st.markdown(f"**{region}**")
                for seed in range(1, 17):
                    val = st.text_input(
                        f"{seed}",
                        value=st.session_state.mm_bracket[region][seed - 1],
                        key=f"mm_{region}_{seed}",
                        label_visibility="visible",
                    )
                    st.session_state.mm_bracket[region][seed - 1] = val

        st.divider()

        sim_col1, sim_col2 = st.columns([1, 5])
        n_sims_mm = sim_col1.selectbox("Simulations", [500, 1000, 5000, 10000], index=1, key="mm_n_sims")

        if st.button("Simulate Bracket", type="primary", key="mm_simulate"):
            all_entered = [
                t for teams in st.session_state.mm_bracket.values()
                for t in teams if t and t.strip()
            ]

            if len(all_entered) < 4:
                st.warning("Enter at least a few team names to simulate.")
            else:
                with st.spinner(f"Loading ELO ratings and running {n_sims_mm:,} simulations..."):
                    try:
                        # Build canonical bracket + display name map
                        display_map: dict = {}
                        bracket_canon: dict = {}
                        for r, teams in st.session_state.mm_bracket.items():
                            canon_list = []
                            for t in teams:
                                if t and t.strip():
                                    c = fn.canonical_ncaab_team(t.strip())
                                    display_map[c] = t.strip()
                                    canon_list.append(c)
                                else:
                                    canon_list.append("")
                            bracket_canon[r] = canon_list

                        mm_canon_set = {t for t in display_map}

                        sched_elo = _load_data(sport, (int(season_pick),), cache_dir)
                        elo_latest: dict = {}
                        if not sched_elo.empty and "game_dt" in sched_elo.columns:
                            # Filter schedule to MM teams for speed
                            elo_mask = (
                                sched_elo["home_team"].isin(mm_canon_set) |
                                sched_elo["away_team"].isin(mm_canon_set)
                            )
                            sched_mm = sched_elo[elo_mask].copy()
                            elo_df_mm = fn.compute_elo_ratings(sched_mm)
                            sched_sorted = sched_mm.sort_values("game_dt").reset_index(drop=True)
                            merged_elo = sched_sorted[["game_id", "home_team", "away_team"]].merge(
                                elo_df_mm[["game_id", "elo_home", "elo_away"]], on="game_id", how="inner"
                            )
                            home_e = merged_elo[["home_team", "elo_home"]].rename(
                                columns={"home_team": "team", "elo_home": "elo"})
                            away_e = merged_elo[["away_team", "elo_away"]].rename(
                                columns={"away_team": "team", "elo_away": "elo"})
                            elo_latest = (
                                pd.concat([home_e, away_e])
                                .groupby("team")["elo"].last()
                                .to_dict()
                            )

                        default_elo_mm = 1500.0
                        best   = _best_bracket(bracket_canon, elo_latest, default_elo_mm)
                        counts = _simulate_tournament(bracket_canon, elo_latest, n=int(n_sims_mm), default_elo=default_elo_mm)

                        st.session_state["mm_results"] = {
                            "best": best, "counts": counts,
                            "n": int(n_sims_mm), "elos": elo_latest,
                            "display_map": display_map,
                        }
                    except Exception as e:
                        st.error(f"Simulation failed: {e}")
                        st.code(traceback.format_exc())

        # ── Results ────────────────────────────────────────────────────────
        if "mm_results" in st.session_state:
            res  = st.session_state["mm_results"]
            n    = res["n"]
            cts  = res["counts"]
            best = res["best"]
            elos = res["elos"]
            dmap = res.get("display_map", {})

            def disp(t):
                return dmap.get(t, t) if t else "TBD"

            st.divider()
            st.markdown(f"### Results — {n:,} simulations")

            # Championship probability table
            all_teams_mm = sorted(
                set(cts["champ"]) | set(cts["ff"]) | set(cts["e8"]),
                key=lambda t: cts["champ"].get(t, 0),
                reverse=True,
            )
            prob_rows = []
            for t in all_teams_mm:
                prob_rows.append({
                    "Team":        disp(t),
                    "Champion":    f"{cts['champ'].get(t, 0) / n:.1%}",
                    "Champ. Game": f"{cts['f2'].get(t, 0) / n:.1%}",
                    "Final Four":  f"{cts['ff'].get(t, 0) / n:.1%}",
                    "Elite 8":     f"{cts['e8'].get(t, 0) / n:.1%}",
                    "Sweet 16":    f"{cts['s16'].get(t, 0) / n:.1%}",
                    "ELO":         f"{elos.get(t, 1500):.0f}",
                })
            st.markdown("#### Championship Odds")
            st.dataframe(pd.DataFrame(prob_rows), use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("#### Model's Best-Pick Bracket")

            round_order = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8"]
            for region in _MM_REGIONS:
                if region not in best.get("regions", {}):
                    continue
                reg = best["regions"][region]
                reg_champ = best["region_champs"].get(region, "")
                with st.expander(f"{region} — Champion: **{disp(reg_champ)}**", expanded=False):
                    r_cols = st.columns(4)
                    for ci, rnd in enumerate(round_order):
                        games = reg.get(rnd, [])
                        if not games:
                            continue
                        with r_cols[ci]:
                            st.markdown(f"**{rnd}**")
                            for (a, b, w) in games:
                                loser = disp(b) if w == a else disp(a)
                                st.markdown(f"**{disp(w)}** def. {loser}")

            # Final Four + Championship
            st.markdown("#### Final Four & Championship")
            ff = best.get("final_four", [])
            champ_r = best.get("championship")
            if ff or champ_r:
                n_ff_cols = len(ff) + (1 if champ_r else 0)
                if n_ff_cols:
                    ff_cols = st.columns(n_ff_cols)
                    for ci, (a, b, w, label) in enumerate(ff):
                        with ff_cols[ci]:
                            st.markdown(f"**{label}**")
                            loser = disp(b) if w == a else disp(a)
                            st.markdown(f"**{disp(w)}** def. {loser}")
                    if champ_r:
                        a, b, w = champ_r
                        with ff_cols[-1]:
                            loser = disp(b) if w == a else disp(a)
                            st.markdown("**Championship**")
                            st.markdown(f"### {disp(w)}")
                            st.markdown(f"def. {loser}")
            else:
                st.caption("Enter teams for all four regions to see the Final Four.")
