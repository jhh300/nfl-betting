from __future__ import annotations
import os, ssl, time, math, re
from dataclasses import dataclass
from typing import Iterable, List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests

def _ssl_bootstrap():
    try:
        import certifi
        ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())  # type: ignore
    except Exception:
        pass

_LONG_TO_ABBR = {
    "ARIZONA CARDINALS":"ARI","ATLANTA FALCONS":"ATL","BALTIMORE RAVENS":"BAL","BUFFALO BILLS":"BUF",
    "CAROLINA PANTHERS":"CAR","CHICAGO BEARS":"CHI","CINCINNATI BENGALS":"CIN","CLEVELAND BROWNS":"CLE",
    "DALLAS COWBOYS":"DAL","DENVER BRONCOS":"DEN","DETROIT LIONS":"DET","GREEN BAY PACKERS":"GB",
    "HOUSTON TEXANS":"HOU","INDIANAPOLIS COLTS":"IND","JACKSONVILLE JAGUARS":"JAC","KANSAS CITY CHIEFS":"KC",
    "LAS VEGAS RAIDERS":"LV","LOS ANGELES CHARGERS":"LAC","LOS ANGELES RAMS":"LAR","MIAMI DOLPHINS":"MIA",
    "MINNESOTA VIKINGS":"MIN","NEW ENGLAND PATRIOTS":"NE","NEW ORLEANS SAINTS":"NO","NEW YORK GIANTS":"NYG",
    "NEW YORK JETS":"NYJ","PHILADELPHIA EAGLES":"PHI","PITTSBURGH STEELERS":"PIT","SAN FRANCISCO 49ERS":"SF",
    "SEATTLE SEAHAWKS":"SEA","TAMPA BAY BUCCANEERS":"TB","TENNESSEE TITANS":"TEN","WASHINGTON COMMANDERS":"WAS",
    "WASHINGTON FOOTBALL TEAM":"WAS","SAN DIEGO CHARGERS":"LAC","ST. LOUIS RAMS":"LAR","ST LOUIS RAMS":"LAR",
    "OAKLAND RAIDERS":"LV",
}
_ALIAS = {"ARZ":"ARI","JAX":"JAC","LA":"LAR","STL":"LAR","SD":"LAC","OAK":"LV","WSH":"WAS","WFT":"WAS"}
_NICK_TO_ABBR = {
    "CARDINALS":"ARI","FALCONS":"ATL","RAVENS":"BAL","BILLS":"BUF","PANTHERS":"CAR","BEARS":"CHI",
    "BENGALS":"CIN","BROWNS":"CLE","COWBOYS":"DAL","BRONCOS":"DEN","LIONS":"DET","PACKERS":"GB",
    "TEXANS":"HOU","COLTS":"IND","JAGUARS":"JAC","CHIEFS":"KC","RAIDERS":"LV","CHARGERS":"LAC","RAMS":"LAR",
    "DOLPHINS":"MIA","VIKINGS":"MIN","PATRIOTS":"NE","SAINTS":"NO","GIANTS":"NYG","JETS":"NYJ","EAGLES":"PHI",
    "STEELERS":"PIT","49ERS":"SF","SEAHAWKS":"SEA","BUCCANEERS":"TB","TITANS":"TEN","COMMANDERS":"WAS","REDSKINS":"WAS",
}

def canonical_team(x: str) -> str:
    if pd.isna(x): return x
    s = re.sub(r"\s+", " ", str(x).upper().replace(".", " ").strip())
    if len(s) <= 4 and s.isalpha(): return _ALIAS.get(s, s)
    if s in _LONG_TO_ABBR: return _LONG_TO_ABBR[s]
    return _NICK_TO_ABBR.get(s.split()[-1], s)

def american_to_decimal(a: float) -> float:
    a = float(a); return 1.0 + (a/100.0) if a > 0 else 1.0 + (100.0/abs(a))

def implied_prob_from_american(a: float) -> float:
    a = float(a); return 100.0/(a+100.0) if a > 0 else (-a)/(100.0-a)

try:
    from scipy.special import erf as _erf   # vectorized
except ImportError:
    _erf = np.vectorize(math.erf)

def norm_cdf(x):
    """Standard normal CDF; accepts scalars or arrays."""
    return 0.5*(1.0 + _erf(np.asarray(x, dtype=float)/math.sqrt(2.0)))

def _to_season_list(seasons) -> List[int]:
    if seasons is None: return []
    if isinstance(seasons, (int, np.integer)): return [int(seasons)]
    if isinstance(seasons, float): return [] if pd.isna(seasons) else [int(seasons)]
    try:
        return sorted({int(s) for s in seasons if not pd.isna(s)})
    except TypeError:
        try: return [int(seasons)]
        except Exception: return []

def _retry_once(fn, args, kwargs=None):
    kwargs = kwargs or {}
    try: return fn(*args, **kwargs)
    except Exception:
        time.sleep(0.8); return fn(*args, **kwargs)

# ---- Schedule loading (includes Vegas closing lines from nflverse) -----------

_SCHED_KEEP = ["game_id","season","week","home_team","away_team","home_score","away_score",
               "gameday","gametime","home_rest","away_rest",
               "spread_line","total_line","home_moneyline","away_moneyline",
               "home_spread_odds","away_spread_odds","over_odds","under_odds"]

def load_schedule(seasons: List[int]) -> pd.DataFrame:
    """Schedule + results + Vegas closing lines (spread_line is home-perspective:
    positive = home favored by that many points)."""
    import nfl_data_py as nfl
    frames = []
    for y in _to_season_list(seasons):
        s = _retry_once(nfl.import_schedules, [[y]])
        frames.append(s[[c for c in _SCHED_KEEP if c in s.columns]].copy())
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["game_id","season","week","home_team","away_team"])
    for c in ["home_team","away_team"]:
        if c in out: out[c] = out[c].apply(canonical_team)
    for c in ["season","week","home_score","away_score","home_rest","away_rest",
              "spread_line","total_line","home_moneyline","away_moneyline",
              "home_spread_odds","away_spread_odds","over_odds","under_odds"]:
        if c in out: out[c] = pd.to_numeric(out[c], errors="coerce")
    if "season" in out: out["season"] = out["season"].astype("Int64")
    if "week" in out: out["week"] = out["week"].astype("Int64")
    # Kickoff timestamp: nflverse ships gameday (date) + gametime (US Eastern)
    if "gameday" in out.columns:
        gt = out["gametime"].fillna("13:00").astype(str) if "gametime" in out.columns else "13:00"
        local = pd.to_datetime(out["gameday"].astype(str) + " " + gt, errors="coerce")
        try:
            out["game_dt"] = local.dt.tz_localize("US/Eastern", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
        except Exception:
            out["game_dt"] = local.dt.tz_localize("UTC")
    else:
        out["game_dt"] = pd.NaT
    return out

def _norm_col(s: str) -> str: return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def _clean_numeric(x):
    if pd.isna(x): return np.nan
    if isinstance(x, str):
        x = x.strip().replace("%","").replace(",","")
        if not x: return np.nan
    try: return float(x)
    except Exception: return np.nan

EXTRA_SYNONYMS: Dict[str, Iterable[str]] = {
    "off_epa_per_play": ["epa/play","epa per play","epa play","team epa/play","offense epa/play","epaoff","epa_on_play"],
    "success_rate_off": ["success %","success pct","success rate","success_rate","succ%","sr off","off success %"],
    "def_epa_per_play": ["def epa/play","epa defense/play","defense epa/play","epa_def","epa allowed/play"],
    "success_rate_def": ["def success %","def success rate","success_rate_def","sr def","def succ%"],
    "def_dvoa":         ["def_dvoa","def dvoa","dvoa def"],
    "st_efficiency":    ["st_efficiency","special teams efficiency","st eff","speceff"],
}
TEAM_KEYS   = ["team","team_abbr","abbr","recent_team","posteam","club_code"]
SEASON_KEYS = ["season","year","season_year","season_id","seasonid"]
WEEK_KEYS   = ["week","wk","week_num","game_week","week_number"]

def _canon_cols_any(df: pd.DataFrame) -> pd.DataFrame:
    norm_to_actual = {_norm_col(c): c for c in df.columns}
    ren: Dict[str,str] = {}
    for key in TEAM_KEYS:
        k = _norm_col(key)
        if k in norm_to_actual: ren[norm_to_actual[k]] = "team"; break
    for key in SEASON_KEYS:
        k = _norm_col(key)
        if k in norm_to_actual: ren[norm_to_actual[k]] = "season"; break
    has_week = False
    for key in WEEK_KEYS:
        k = _norm_col(key)
        if k in norm_to_actual: ren[norm_to_actual[k]] = "week"; has_week = True; break
    for tgt, syns in EXTRA_SYNONYMS.items():
        if tgt in df.columns: ren[tgt] = tgt; continue
        for s in syns:
            ns = _norm_col(s)
            if ns in norm_to_actual: ren[norm_to_actual[ns]] = tgt; break
    out = df.rename(columns=ren)
    keep = ["team","season"] + (["week"] if has_week else []) + [c for c in EXTRA_SYNONYMS if c in out.columns]
    out = out[[c for c in keep if c in out.columns]].copy()
    if "team" in out: out["team"] = out["team"].apply(canonical_team)
    if "season" in out: out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    if "week" in out: out["week"] = pd.to_numeric(out["week"], errors="coerce").astype("Int64")
    for c in out.columns:
        if c not in ("team","season","week"): out[c] = out[c].map(_clean_numeric)
    return out

def load_extra_from_uploads(files: List[object]) -> pd.DataFrame:
    frames = []
    for f in files or []:
        try:
            try: df = pd.read_csv(f)
            except UnicodeDecodeError: f.seek(0); df = pd.read_csv(f, encoding="latin-1")
            df = _canon_cols_any(df)
            if {"team","season"} <= set(df.columns): frames.append(df)
        except Exception: continue
    if not frames: return pd.DataFrame(columns=["team","season","week"])
    out = frames[0]
    for add in frames[1:]:
        keys = [k for k in ["team","season","week"] if k in out.columns and k in add.columns]
        out = out.merge(add, on=keys, how="outer")
    keys_present = [k for k in ["team","season","week"] if k in out.columns]
    out = out.sort_values(keys_present).reset_index(drop=True)
    if "week" in out.columns:
        grp = out.groupby(["team","season"], group_keys=False)
        num_cols = [c for c in out.columns if c not in ("team","season","week")]
        for c in num_cols:
            out[c] = grp[c].apply(lambda s: s.ffill())
    return out

def integrate_extra(weekly_df: pd.DataFrame, extra_df: pd.DataFrame) -> pd.DataFrame:
    if extra_df is None or extra_df.empty: return weekly_df
    base = weekly_df.copy()
    for k in ("team","season","week"):
        if k not in base.columns: raise KeyError(f"weekly_df is missing key column: {k}")
    base["team"]   = base["team"].apply(canonical_team)
    base["season"] = pd.to_numeric(base["season"], errors="coerce").astype("Int64")
    base["week"]   = pd.to_numeric(base["week"],   errors="coerce").astype("Int64")
    ex = extra_df.copy()
    ex["team"]   = ex["team"].apply(canonical_team)
    ex["season"] = pd.to_numeric(ex["season"], errors="coerce").astype("Int64")
    if "week" in ex.columns: ex["week"] = pd.to_numeric(ex["week"], errors="coerce").astype("Int64")
    metric_cols = [c for c in ex.columns if c not in ("team","season","week")]
    if "week" not in ex.columns or ex["week"].isna().all():
        ex_ts = ex.drop(columns=[c for c in ["week"] if c in ex.columns]).drop_duplicates(["team","season"])
        merged = base.merge(ex_ts, on=["team","season"], how="left")
        grp = merged.groupby(["team","season"], group_keys=False)
        for c in metric_cols:
            if c in merged.columns:
                merged[c] = pd.to_numeric(merged[c], errors="coerce")
                merged[f"{c}_roll4"] = grp[c].transform(lambda s: s.rolling(4, min_periods=1).mean())
        return merged
    merged = base.merge(ex, on=["team","season","week"], how="left")
    grp = merged.groupby(["team","season"], group_keys=False)
    for c in metric_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
        merged[c] = grp[c].apply(lambda s: s.ffill())
        merged[f"{c}_roll4"] = grp[c].transform(lambda s: s.rolling(4, min_periods=1).mean())
    return merged

# ---- Team weekly stats (schedule points + free nflverse EPA) -----------------

_EPA_STATS = ["off_epa_per_play","def_epa_per_play","pass_epa_db","def_pass_epa_db","turnover_margin"]

def _team_epa_weekly(seasons: List[int]) -> pd.DataFrame:
    """Team-week EPA and turnover stats aggregated from free nflverse
    player-weekly data (no API key). Defensive numbers are the opponent's
    offensive numbers that week. Returns empty frame on any failure so the
    pipeline degrades gracefully to points-only features."""
    cols = ["recent_team","opponent_team","season","week","attempts","sacks","carries",
            "passing_epa","rushing_epa","interceptions",
            "sack_fumbles_lost","rushing_fumbles_lost","receiving_fumbles_lost"]
    try:
        import nfl_data_py as nfl
        frames = []
        for y in _to_season_list(seasons):
            try: frames.append(_retry_once(nfl.import_weekly_data, [[y], cols], {"downcast": False}))
            except Exception: continue
        if not frames: return pd.DataFrame()
        w = pd.concat(frames, ignore_index=True)
    except Exception:
        return pd.DataFrame()
    w["team"]     = w["recent_team"].apply(canonical_team)
    w["opponent"] = w["opponent_team"].apply(canonical_team)
    for c in cols[3:]:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    g = (w.groupby(["team","season","week"], as_index=False)
          .agg(opponent=("opponent", "first"),
               pass_epa=("passing_epa","sum"), rush_epa=("rushing_epa","sum"),
               attempts=("attempts","sum"), sacks=("sacks","sum"), carries=("carries","sum"),
               interceptions=("interceptions","sum"), sfl=("sack_fumbles_lost","sum"),
               rfl=("rushing_fumbles_lost","sum"), refl=("receiving_fumbles_lost","sum")))
    dropbacks = (g["attempts"] + g["sacks"]).replace(0, np.nan)
    plays     = (dropbacks + g["carries"]).replace(0, np.nan)
    g["off_epa_per_play"] = (g["pass_epa"] + g["rush_epa"]) / plays
    g["pass_epa_db"]      = g["pass_epa"] / dropbacks
    g["giveaways"]        = g["interceptions"] + g["sfl"] + g["rfl"] + g["refl"]
    opp = g[["team","season","week","off_epa_per_play","pass_epa_db","giveaways"]].rename(
        columns={"team":"opponent","off_epa_per_play":"def_epa_per_play",
                 "pass_epa_db":"def_pass_epa_db","giveaways":"takeaways"})
    g = g.merge(opp, on=["opponent","season","week"], how="left")
    g["turnover_margin"] = g["takeaways"] - g["giveaways"]
    g["season"] = pd.to_numeric(g["season"], errors="coerce").astype("Int64")
    g["week"]   = pd.to_numeric(g["week"],   errors="coerce").astype("Int64")
    return g[["team","season","week"] + _EPA_STATS]

def team_weekly_from_schedule(sched: pd.DataFrame, with_epa: bool = True) -> pd.DataFrame:
    """One row per team per completed week. Row for week W holds stats THROUGH
    week W; build_game_features joins as-of strictly-before the game week, so
    no game's own result ever leaks into its features."""
    s = sched.dropna(subset=["home_score","away_score"]).copy()
    h = s[["season","week","home_team","home_score","away_score"]].rename(
        columns={"home_team":"team","home_score":"points_scored","away_score":"points_allowed"})
    a = s[["season","week","away_team","away_score","home_score"]].rename(
        columns={"away_team":"team","away_score":"points_scored","home_score":"points_allowed"})
    w = pd.concat([h, a], ignore_index=True)
    w["season"] = pd.to_numeric(w["season"], errors="coerce").astype("Int64")
    w["week"]   = pd.to_numeric(w["week"],   errors="coerce").astype("Int64")
    w = w.sort_values(["team","season","week"]).reset_index(drop=True)
    grp = w.groupby(["team","season"], group_keys=False)
    for c in ["points_scored","points_allowed"]:
        w[c] = pd.to_numeric(w[c], errors="coerce")
        w[f"{c}_roll4"] = grp[c].transform(lambda x: x.rolling(4, min_periods=1).mean())
    w["point_diff"]     = w["points_scored"] - w["points_allowed"]
    w["point_diff_avg"] = grp["point_diff"].transform(lambda x: x.expanding().mean())
    w["win"]            = (w["point_diff"] > 0).astype(float)
    w["win_pct"]        = grp["win"].transform(lambda x: x.expanding().mean())

    if with_epa:
        epa = _team_epa_weekly(sorted(pd.to_numeric(s["season"], errors="coerce").dropna().astype(int).unique()))
        if not epa.empty:
            w = w.merge(epa, on=["team","season","week"], how="left").sort_values(["team","season","week"])
            grp = w.groupby(["team","season"], group_keys=False)
            for c in _EPA_STATS:
                w[f"{c}_roll4"] = grp[c].transform(lambda x: x.rolling(4, min_periods=1).mean())

    for stub in TEAM_LEVEL_STATS:
        if stub not in w.columns: w[stub] = np.nan
    return w[["team","season","week"] + TEAM_LEVEL_STATS].reset_index(drop=True)

def load_weekly_team_stats(seasons: List[int]) -> pd.DataFrame:
    return team_weekly_from_schedule(load_schedule(seasons))

# ---- Feature definitions ----------------------------------------------------

BASE_ROLL4S = [
    "points_scored_roll4","points_allowed_roll4",
    "off_epa_per_play_roll4","def_epa_per_play_roll4",
    "success_rate_off_roll4","success_rate_def_roll4",
]
TEAM_LEVEL_STATS = BASE_ROLL4S + [
    "pass_epa_db_roll4","def_pass_epa_db_roll4","turnover_margin_roll4",
    "point_diff_avg","win_pct",
]
CANDIDATE_FEATURES = [f"{s}_diff" for s in TEAM_LEVEL_STATS] + ["pyth_diff","rest_days_diff"]
FEATURE_COLS = CANDIDATE_FEATURES

# ---- Feature building (leakage-free as-of join) ------------------------------

_CARRYOVER_SHRINK = 0.65  # prior-season stats regress this far from the league mean

def _asof_team_stats(sched: pd.DataFrame, weekly: pd.DataFrame, side: str) -> pd.DataFrame:
    """For each game, attach the team's stats from its most recent game STRICTLY
    BEFORE kickoff week — including across season boundaries, so week-1 and
    offseason games use last season's form (shrunk toward the league mean)
    instead of having no features at all."""
    stat_cols = [c for c in weekly.columns if c not in ("team","season","week")]
    s = sched[["game_id","season","week",f"{side}_team"]].rename(columns={f"{side}_team":"team"}).copy()
    w = weekly[["team","season","week"]+stat_cols].copy()
    for d in (s, w):
        d["week"]   = pd.to_numeric(d["week"], errors="coerce")
        d["season"] = pd.to_numeric(d["season"], errors="coerce")
    s = s.dropna(subset=["team","season","week"]).astype({"week":"int64","season":"int64"})
    w = w.dropna(subset=["team","season","week"]).astype({"week":"int64","season":"int64"})
    if s.empty or w.empty:
        return pd.DataFrame(columns=[f"{side}_{c}" for c in stat_cols])
    # Continuous time key so "most recent game" can reach back into the prior season
    s["t"] = s["season"]*100 + s["week"]
    w["t"] = w["season"]*100 + w["week"]
    w = w.rename(columns={"season":"stat_season"}).drop(columns=["week"])
    j = pd.merge_asof(s.sort_values("t"), w.sort_values("t"), on="t", by="team",
                      direction="backward", allow_exact_matches=False)
    carried = j["stat_season"].notna() & (j["stat_season"] < j["season"])
    if carried.any():
        for c in stat_cols:
            mu = w[c].mean()
            if pd.notna(mu):
                j.loc[carried, c] = mu + (j.loc[carried, c] - mu) * _CARRYOVER_SHRINK
    return j.set_index("game_id")[stat_cols].rename(columns={c: f"{side}_{c}" for c in stat_cols})

def build_game_features(sched: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    for k in ["team","season","week"]:
        if k not in weekly.columns: raise KeyError(f"weekly is missing {k}")
    feats = sched.copy()
    feats = feats.merge(_asof_team_stats(sched, weekly, "home"), left_on="game_id", right_index=True, how="left")
    feats = feats.merge(_asof_team_stats(sched, weekly, "away"), left_on="game_id", right_index=True, how="left")

    for stub in TEAM_LEVEL_STATS:
        for pfx in ("home_","away_"):
            if pfx+stub not in feats.columns: feats[pfx+stub] = np.nan
        feats[f"{stub}_diff"] = (
            pd.to_numeric(feats[f"home_{stub}"], errors="coerce") -
            pd.to_numeric(feats[f"away_{stub}"], errors="coerce")
        )

    # Pythagorean expectation differential
    for side in ("home_","away_"):
        ps = pd.to_numeric(feats.get(f"{side}points_scored_roll4", np.nan), errors="coerce")
        pa = pd.to_numeric(feats.get(f"{side}points_allowed_roll4", np.nan), errors="coerce")
        feats[f"{side}pyth"] = ps**2.37 / (ps**2.37 + pa**2.37 + 1e-9)
    feats["pyth_diff"] = feats["home_pyth"] - feats["away_pyth"]

    # Rest differential straight from nflverse schedule columns
    if "home_rest" in feats.columns and "away_rest" in feats.columns:
        feats["rest_days_diff"] = (
            pd.to_numeric(feats["home_rest"], errors="coerce") -
            pd.to_numeric(feats["away_rest"], errors="coerce")
        )
    else:
        feats["rest_days_diff"] = np.nan
    return feats

def make_training_set(feats_df: pd.DataFrame) -> pd.DataFrame:
    """Adds y_home_win/y_margin/y_total, NaN for any game without a final
    score. Must be computed per-row: a frame mixing played and unplayed games
    (e.g. this season's completed weeks next to its future schedule) is the
    normal case, not the exception — `hs > as_` on NaN scores evaluates to
    False rather than NaN, so an any-completed-anywhere check would wrongly
    label every future game a "away team won" result."""
    df = feats_df.copy()
    for c in ["home_score","away_score"]:
        if c not in df: df[c] = np.nan
    hs  = pd.to_numeric(df["home_score"], errors="coerce")
    as_ = pd.to_numeric(df["away_score"], errors="coerce")
    played = hs.notna() & as_.notna()
    df["y_home_win"] = np.where(played, (hs > as_).astype(float), np.nan)
    df["y_margin"]   = np.where(played, hs - as_, np.nan)
    df["y_total"]    = np.where(played, hs + as_, np.nan)
    return df

# ---- Fallback models --------------------------------------------------------

@dataclass
class BiasRegressor:
    mu: float = 0.0
    used_features: List[str] = None
    def predict(self, X):
        n = len(X) if hasattr(X,"__len__") else 1
        return np.full(n, self.mu, dtype=float)

# ---- Model utilities --------------------------------------------------------

def _prepare_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    have = [c for c in FEATURE_COLS if c in df.columns]
    X = df[have].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target_col], errors="coerce")
    mask = ~y.isna()
    X = X.loc[mask].copy(); y = y.loc[mask].values
    non_all_nan = [c for c in X.columns if not X[c].isna().all()]
    return X[non_all_nan], y, non_all_nan

def model_used_features(model) -> List[str]:
    feats = getattr(model, "used_features", None)
    if feats is None: feats = getattr(model, "feature_names_in_", None)
    return list(feats) if feats is not None else []

def align_features_for_model(model, X: pd.DataFrame) -> pd.DataFrame:
    feats = model_used_features(model)
    if not feats: return X.apply(pd.to_numeric, errors="coerce")
    return X.reindex(columns=feats, fill_value=np.nan).apply(pd.to_numeric, errors="coerce")

def _build_pipe(estimator):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("est", estimator)])

_GBM_PARAMS = dict(n_estimators=300, max_depth=2, learning_rate=0.03,
                   subsample=0.8, min_samples_leaf=20, random_state=42)

# ---- Model fitting -----------------------------------------------------------

def _fit_gbm(train_df: pd.DataFrame, target_col: str, default_mu: float, default_sigma: float,
             compute_cv: bool = True) -> Tuple[object, float, float]:
    """Gradient-boosted regressor. Returns (model, oof_sigma, cv_mae) where sigma
    comes from out-of-fold residuals so cover/total probabilities are honest.
    compute_cv=False skips cross-validation (for cheap in-season refits) and
    returns default_sigma — pass the previously estimated sigma."""
    X, y, feats = _prepare_xy(train_df, target_col)
    if X.empty or len(y) == 0:
        return BiasRegressor(mu=default_mu, used_features=["__bias__"]), default_sigma, default_sigma

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_predict

    pipe = _build_pipe(GradientBoostingRegressor(**_GBM_PARAMS))

    n_cv = min(5, len(y) // 15) if (compute_cv and len(y) >= 30) else 0
    if n_cv >= 2:
        yhat_cv = cross_val_predict(pipe, X, y, cv=n_cv)
        resid_cv = y - yhat_cv
        sigma = float(np.std(resid_cv, ddof=1))
        cv_mae = float(np.abs(resid_cv).mean())
    else:
        sigma = default_sigma
        cv_mae = np.nan

    pipe.fit(X, y)
    pipe.used_features = feats  # type: ignore
    return pipe, sigma, cv_mae


def fit_margin_model(train_df: pd.DataFrame) -> Tuple[object, float, float]:
    return _fit_gbm(train_df, "y_margin", 0.0, 14.0)

def fit_total_model(train_df: pd.DataFrame) -> Tuple[object, float, float]:
    return _fit_gbm(train_df, "y_total", 44.0, 13.0)

# ---- Prediction helpers -----------------------------------------------------

def win_prob_from_margin(mu_margin, sigma: float) -> np.ndarray:
    """P(home win) = Phi(mu/sigma) — derived from the margin model so the
    moneyline probability, spread, and predicted scores always agree."""
    sig = float(sigma) if sigma and sigma > 0 else 13.0
    return np.clip(norm_cdf(np.asarray(mu_margin, dtype=float) / sig), 1e-6, 1-1e-6)

def prob_home_covers(line: float, mu_margin: float, sigma: float) -> float:
    """P(home covers). `line` is the HOME handicap (negative = home favored).
    Home covers when margin > -line."""
    if sigma is None or sigma <= 0 or pd.isna(mu_margin) or pd.isna(line): return np.nan
    return float(norm_cdf((mu_margin + line) / sigma))

def prob_away_covers(line: float, mu_margin: float, sigma: float) -> float:
    """P(away covers). `line` is the AWAY handicap (positive = away underdog).
    Away covers when home margin < line."""
    if sigma is None or sigma <= 0 or pd.isna(mu_margin) or pd.isna(line): return np.nan
    return float(norm_cdf((line - mu_margin) / sigma))

def prob_over(total: float, mu_total: float, sigma: float) -> float:
    if sigma is None or sigma <= 0 or pd.isna(mu_total) or pd.isna(total): return np.nan
    return float(1.0 - norm_cdf((total - mu_total) / sigma))

def prob_under(total: float, mu_total: float, sigma: float) -> float:
    if sigma is None or sigma <= 0 or pd.isna(mu_total) or pd.isna(total): return np.nan
    return float(norm_cdf((total - mu_total) / sigma))

# ---- Odds fetching ----------------------------------------------------------

def fetch_odds_current(api_key: str, market: str = "h2h", regions: str = "us", odds_format: str = "american",
                       bookmakers: Optional[str] = None) -> pd.DataFrame:
    _ssl_bootstrap()
    params = {"apiKey":api_key,"markets":market,"regions":regions,"oddsFormat":odds_format}
    if bookmakers: params["bookmakers"] = bookmakers
    r = requests.get("https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/", params=params, timeout=30)
    r.raise_for_status()
    rows = []
    for ev in r.json():
        home = canonical_team(ev.get("home_team")); away = canonical_team(ev.get("away_team"))
        for bm in ev.get("bookmakers",[]):
            for m in bm.get("markets",[]):
                for out in m.get("outcomes",[]):
                    rows.append({
                        "odds_event_id":ev.get("id"),"commence_dt":ev.get("commence_time"),
                        "home_team":home,"away_team":away,
                        "book_key":bm.get("key"),"book_title":bm.get("title"),
                        "market_key":m.get("key"),"outcome_name":str(out.get("name","")),
                        "price_american":out.get("price"),"line":out.get("point"),
                    })
    return pd.DataFrame(rows)

def fetch_closing_from_history(api_key: str, event_id: str, book_key: str, market_key: str = "h2h") -> dict:
    _ssl_bootstrap()
    url = f"https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/events/{event_id}/odds"
    r = requests.get(url, params={"apiKey":api_key,"markets":market_key,"regions":"us","oddsFormat":"american","bookmakers":book_key}, timeout=30)
    r.raise_for_status()
    data = r.json().get("data", {})
    info: dict = {"home_price":None,"away_price":None,"home_line":None,"away_line":None,"total_line":None,"over_price":None,"under_price":None}
    home = canonical_team(data.get("home_team",""))
    for bm in data.get("bookmakers",[]):
        for m in bm.get("markets",[]):
            for out in m.get("outcomes",[]):
                name  = str(out.get("name","")).strip().lower()
                price = out.get("price"); point = out.get("point")
                if market_key == "h2h":
                    if canonical_team(out.get("name","")) == home: info["home_price"] = price
                    else: info["away_price"] = price
                elif market_key == "spreads":
                    if canonical_team(out.get("name","")) == home: info["home_price"] = price; info["home_line"] = point
                    else: info["away_price"] = price; info["away_line"] = point
                elif market_key == "totals":
                    if name == "over": info["over_price"] = price; info["total_line"] = point
                    elif name == "under": info["under_price"] = price
    return info

def prep_odds(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()
    out["commence_dt"] = pd.to_datetime(out["commence_dt"], errors="coerce", utc=True)
    def _is_home(row) -> bool:
        n = str(row["outcome_name"]).strip().lower()
        return n == "home" if n in ("home","away","over","under") else canonical_team(row["outcome_name"]) == row["home_team"]
    out["is_home_outcome"] = out.apply(_is_home, axis=1)
    out["is_over_outcome"] = out["outcome_name"].str.strip().str.lower().eq("over")
    out["odds_decimal"] = out["price_american"].map(lambda x: american_to_decimal(x) if pd.notna(x) else np.nan)
    out["book_p_raw"]   = out["price_american"].map(lambda x: implied_prob_from_american(x) if pd.notna(x) else np.nan)
    # NOTE: don't group by "line" — spread sides carry different lines
    # (home -3.5 vs away +3.5), which would split each pair into its own
    # size-1 group and make devig a no-op; h2h has no line at all, which
    # would drop those rows from grouping entirely. Event+book+market alone
    # already pairs the two opposing outcomes correctly for every market.
    grp_keys = ["odds_event_id","book_key","market_key"]
    out["vig_sum"] = out.groupby(grp_keys)["book_p_raw"].transform("sum")
    out["book_p_devig"] = (out["book_p_raw"] / out["vig_sum"]).clip(lower=1e-6, upper=1-1e-6)
    return out

def _pair_key_u(home: str, away: str) -> str:
    return "|".join(sorted([str(home), str(away)]))

_ODDS_KEEP = ["odds_event_id","commence_dt","home_team","away_team","book_key","book_title","market_key","outcome_name",
              "price_american","line","odds_decimal","book_p_raw","vig_sum","book_p_devig","season","week","game_id",
              "is_home_outcome","is_over_outcome"]

def merge_odds_by_pair_only(odds: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    if odds is None or sched is None or odds.empty or sched.empty: return pd.DataFrame()
    o = odds.copy(); s = sched.copy()
    for c in ["home_team","away_team"]:
        if c in o: o[c] = o[c].apply(canonical_team)
        if c in s: s[c] = s[c].apply(canonical_team)
    o["pair_key_u"] = o.apply(lambda r: _pair_key_u(r["home_team"],r["away_team"]), axis=1)
    s["pair_key_u"] = s.apply(lambda r: _pair_key_u(r["home_team"],r["away_team"]), axis=1)
    # Odds are for upcoming games: prefer the not-yet-played meeting so a late-season
    # divisional rematch doesn't get mapped to the pair's first meeting months earlier
    if {"home_score","away_score"} <= set(s.columns):
        upcoming = s[s["home_score"].isna() | s["away_score"].isna()]
        if not upcoming.empty: s = upcoming
    s_uniq = s.sort_values(["season","week"]).drop_duplicates(subset=["pair_key_u"], keep="first")
    matched = o.merge(s_uniq[["pair_key_u","season","week","game_id"]], on="pair_key_u", how="inner")
    return matched[[c for c in _ODDS_KEEP if c in matched.columns]].reset_index(drop=True)

def merge_odds_with_schedule(odds: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    if odds is None or sched is None or odds.empty or sched.empty: return pd.DataFrame()
    o = odds.copy()
    o["commence_dt"] = pd.to_datetime(o.get("commence_dt"), errors="coerce", utc=True)
    for c in ["home_team","away_team"]:
        if c in o: o[c] = o[c].apply(canonical_team)
    o["pair_key_u"] = o.apply(lambda r: _pair_key_u(r["home_team"],r["away_team"]), axis=1)
    s = sched.copy()
    for c in ["home_team","away_team"]:
        if c in s: s[c] = s[c].apply(canonical_team)
    if "game_dt" not in s.columns: s["game_dt"] = pd.NaT
    s["pair_key_u"] = s.apply(lambda r: _pair_key_u(r["home_team"],r["away_team"]), axis=1)

    def _pass_with_window(window_days: int) -> pd.DataFrame:
        oo = o.reset_index(drop=False).rename(columns={"index":"oid"})
        cand = oo.merge(s[["pair_key_u","season","week","game_id","game_dt"]], on="pair_key_u", how="left")
        cand["commence_dt"] = pd.to_datetime(cand["commence_dt"], errors="coerce", utc=True)
        cand["game_dt"]     = pd.to_datetime(cand["game_dt"],     errors="coerce", utc=True)
        cand["dt_diff_hours"] = np.where(
            cand["commence_dt"].notna() & cand["game_dt"].notna(),
            (cand["commence_dt"]-cand["game_dt"]).dt.total_seconds().abs()/3600.0, np.inf)
        if cand["dt_diff_hours"].isna().all() or np.isinf(cand["dt_diff_hours"]).all(): return pd.DataFrame()
        matched = cand.loc[cand.groupby("oid")["dt_diff_hours"].idxmin()].copy()
        matched = matched[matched["dt_diff_hours"] <= 24.0*float(window_days)].copy()
        return matched[[c for c in _ODDS_KEEP if c in matched.columns]].reset_index(drop=True)

    for days in (7, 21):
        m = _pass_with_window(days)
        if not m.empty: return m
    return merge_odds_by_pair_only(o, s)

# ---- Kelly criterion --------------------------------------------------------

def _kelly_fraction(p: float, dec: float) -> float:
    b = max(dec-1.0, 1e-9); q = 1.0-p
    return float(max(0.0, (b*p-q)/b))

def compose_picks_for_market(g: pd.DataFrame, market_friendly: str, bankroll: float = 10000.0, kelly_fraction_use: float = 0.5) -> pd.DataFrame:
    if g.empty: return g
    df = g.copy()
    if market_friendly == "totals":
        cond = df["is_over_outcome"] if "is_over_outcome" in df.columns else df["outcome_name"].str.lower().eq("over")
        df["side"] = np.where(cond, "OVER", "UNDER")
    else:
        cond = df["is_home_outcome"] if "is_home_outcome" in df.columns else (
            df["outcome_name"].str.lower().eq("home") | df["home_team"].eq(df["outcome_name"].apply(canonical_team)))
        df["side"] = np.where(cond, "HOME", "AWAY")
    df["p_true"] = pd.to_numeric(df["p_true"], errors="coerce")
    df["odds_decimal"] = pd.to_numeric(df["odds_decimal"], errors="coerce")
    df["ev_per_$1"] = df["p_true"]*(df["odds_decimal"]-1.0) - (1.0-df["p_true"])
    df["kelly_frac"] = (df.apply(
        lambda r: _kelly_fraction(r["p_true"],r["odds_decimal"]) if pd.notna(r["p_true"]) and pd.notna(r["odds_decimal"]) else 0.0,
        axis=1).clip(lower=0.0) * float(kelly_fraction_use))
    df["stake_usd"]  = (df["kelly_frac"]*float(bankroll)).round(2)
    df["market_key"] = market_friendly
    if "book_p_devig" in df.columns: df["edge_vs_book_devig"] = df["p_true"]-df["book_p_devig"]
    keep = ["season","week","game_id","home_team","away_team","book_title","market_key","side","line",
            "model_line","edge_pts","price_american","odds_decimal","book_p_devig","p_true","ev_per_$1",
            "kelly_frac","stake_usd","edge_vs_book_devig"]
    return df[[c for c in keep if c in df.columns]].copy().rename(columns={"price_american":"odds_american","ev_per_$1":"ev_per_usd"})

# ---- Shared picks assembly --------------------------------------------------

def assemble_picks(
    merged: pd.DataFrame,
    markets: List[str],
    p_home_df: pd.DataFrame,
    mu_margin_map: dict,
    mu_total_map: dict,
    s_margin: float,
    s_total: float,
    bankroll: float,
    kelly_fraction_use: float,
) -> pd.DataFrame:
    """Single source of truth for building picks from a merged odds+schedule frame.
    Adds `model_line` (the model's fair line for that side) and `edge_pts`
    (points of cushion between the model's number and the book's line)."""
    mk_map = {"moneyline": "h2h", "spreads": "spreads", "totals": "totals"}
    picks_all = []

    for mk in markets:
        sub = merged[merged["market_key"].eq(mk_map[mk])].copy()
        if sub.empty:
            continue
        sub = sub.merge(p_home_df, on="game_id", how="left")

        cond_home = (
            sub["is_home_outcome"] if "is_home_outcome" in sub.columns else
            (sub["outcome_name"].str.lower().eq("home") |
             sub["home_team"].eq(sub["outcome_name"].apply(canonical_team)))
        ).to_numpy()

        if mk == "moneyline":
            sub["p_true"] = np.where(cond_home, sub["p_home_model"], 1.0 - sub["p_home_model"])

        elif mk == "spreads":
            mu   = pd.to_numeric(sub["game_id"].map(mu_margin_map), errors="coerce").to_numpy()
            line = pd.to_numeric(sub["line"], errors="coerce").to_numpy()
            sig  = float(s_margin) if s_margin and s_margin > 0 else np.nan
            p = np.where(cond_home, norm_cdf((mu + line)/sig), norm_cdf((line - mu)/sig))
            sub["p_true"]     = np.where(np.isnan(mu) | np.isnan(line), np.nan, p)
            sub["model_line"] = np.where(cond_home, -mu, mu)
            sub["edge_pts"]   = np.where(cond_home, mu + line, line - mu)

        else:  # totals
            mu   = pd.to_numeric(sub["game_id"].map(mu_total_map), errors="coerce").to_numpy()
            line = pd.to_numeric(sub["line"], errors="coerce").to_numpy()
            sig  = float(s_total) if s_total and s_total > 0 else np.nan
            cond_over = (
                sub["is_over_outcome"] if "is_over_outcome" in sub.columns else
                sub["outcome_name"].str.lower().eq("over")
            ).to_numpy()
            p = np.where(cond_over, 1.0 - norm_cdf((line - mu)/sig), norm_cdf((line - mu)/sig))
            sub["p_true"]     = np.where(np.isnan(mu) | np.isnan(line), np.nan, p)
            sub["model_line"] = mu
            sub["edge_pts"]   = np.where(cond_over, mu - line, line - mu)

        picks_mkt = compose_picks_for_market(sub, mk, bankroll=bankroll, kelly_fraction_use=kelly_fraction_use)
        if not picks_mkt.empty:
            picks_all.append(picks_mkt)

    if not picks_all:
        return pd.DataFrame()
    return pd.concat(picks_all, ignore_index=True)

# ---- CLI entry point --------------------------------------------------------

TRAIN_WINDOW_SEASONS = 8    # seasons of history used to train
MIN_EDGE_PTS_DEFAULT = 2.0  # skip spread/total picks where model and book differ by less

def filter_bettable(picks: pd.DataFrame, min_edge_pts: float = MIN_EDGE_PTS_DEFAULT) -> pd.DataFrame:
    """Keep only picks worth betting: positive EV, and for spreads/totals the
    model must disagree with the book line by at least min_edge_pts (backtests
    show the edge concentrates in larger disagreements)."""
    if picks is None or picks.empty: return picks
    f = picks[pd.to_numeric(picks["ev_per_usd"], errors="coerce") > 0]
    if "edge_pts" in f.columns and min_edge_pts > 0:
        edge = pd.to_numeric(f["edge_pts"], errors="coerce")
        f = f[(~f["market_key"].isin(["spreads","totals"])) | (edge >= min_edge_pts)]
    return f.copy()

def run_picks(season: int, markets=None, books: str = "fanduel,draftkings,betmgm",
              bankroll: float = 10000.0, kelly_fraction: float = 0.5, top_n: int = 20,
              save_dir: str = "./data", log_sqlite: bool = False, db_path: str = "./data/bets.db",
              api_key: str = None, min_edge_pts: float = MIN_EDGE_PTS_DEFAULT):
    api_key = api_key or os.getenv("THE_ODDS_API_KEY")
    if not api_key: raise ValueError("No API key provided (set THE_ODDS_API_KEY or pass --api_key)")
    markets = markets or ["moneyline","spreads"]   # totals excluded by default: backtests show it loses
    train_seasons = list(range(season - TRAIN_WINDOW_SEASONS, season))
    season_list   = sorted(set(train_seasons+[season]))
    sched_all  = load_schedule(season_list)
    weekly_all = team_weekly_from_schedule(sched_all)
    feats_all  = build_game_features(sched_all, weekly_all)
    # Train on prior seasons AND this season's completed games (unplayed games
    # have no result and are dropped automatically)
    train_df   = make_training_set(feats_all[feats_all["season"] <= season].copy())
    margin_model, s_margin, _ = fit_margin_model(train_df)
    total_model,  s_total,  _ = fit_total_model(train_df)
    api_map = {"moneyline":"h2h","spreads":"spreads","totals":"totals"}
    frames = []
    for mk in markets:
        od = fetch_odds_current(api_key, market=api_map[mk], bookmakers=books or None)
        if isinstance(od, pd.DataFrame) and not od.empty: frames.append(od)
    if not frames: raise RuntimeError("No odds returned.")
    odds_live = prep_odds(pd.concat(frames, ignore_index=True))
    sched_target = sched_all[sched_all["season"]==season].copy()
    merged = merge_odds_with_schedule(odds_live, sched_target)
    if merged.empty: merged = merge_odds_by_pair_only(odds_live, sched_target)
    if merged.empty: raise RuntimeError("No market rows matched schedule.")
    # Keep every matched upcoming game — odds can span two weeks late in the season
    weeks = pd.to_numeric(merged["week"], errors="coerce").dropna().astype(int)
    wk_label = f"week{weeks.min()}" if weeks.min() == weeks.max() else f"weeks{weeks.min()}-{weeks.max()}"
    present = [c for c in FEATURE_COLS if c in feats_all.columns]
    merged = merged.merge(feats_all[["game_id","season","week"]+present].drop_duplicates("game_id"),
                          on=["game_id","season","week"], how="left", validate="m:1")
    game_feats = merged.drop_duplicates(subset=["game_id"]).copy().reset_index(drop=True)
    for c in FEATURE_COLS:
        if c not in game_feats.columns: game_feats[c] = np.nan
    X_full = game_feats[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    mu_margin = margin_model.predict(align_features_for_model(margin_model, X_full))
    mu_total  = total_model.predict(align_features_for_model(total_model,  X_full))
    p_home    = win_prob_from_margin(mu_margin, s_margin)

    p_home_df     = pd.DataFrame({"game_id": game_feats["game_id"].values, "p_home_model": p_home})
    mu_margin_map = dict(zip(game_feats["game_id"], mu_margin))
    mu_total_map  = dict(zip(game_feats["game_id"],  mu_total))

    df_out = assemble_picks(merged, markets, p_home_df, mu_margin_map, mu_total_map,
                            s_margin, s_total, bankroll, kelly_fraction)
    df_out = filter_bettable(df_out, min_edge_pts)
    if df_out.empty: raise RuntimeError("No picks passed the edge/EV filter.")
    df_out = df_out.sort_values(["ev_per_usd","edge_vs_book_devig"], ascending=False).head(top_n)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"picks_season{season}_{wk_label}.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Top {top_n} picks ({wk_label}, Season {season}, {merged['game_id'].nunique()} games):")
    print(df_out.to_string(index=False))
    print(f"\nSaved to {out_path}")
    return df_out

# ---- Walk-forward backtest --------------------------------------------------

def run_backtest(sched_all: pd.DataFrame, weekly_all: pd.DataFrame, eval_seasons: List[int]) -> pd.DataFrame:
    """Walk-forward backtest: for each season in eval_seasons, train on all prior
    seasons in the data and evaluate on completed games in that season.
    Output carries the Vegas closing lines so results can be compared to the market."""
    all_seasons = sorted({int(s) for s in sched_all["season"].dropna().unique()})
    eval_seasons = sorted({int(s) for s in eval_seasons})

    feats_all = build_game_features(sched_all, weekly_all)
    results = []

    for target_season in eval_seasons:
        prior_seasons = [s for s in all_seasons if s < target_season]
        if not prior_seasons:
            continue

        train_mask = feats_all["season"].isin(prior_seasons)
        train_df = make_training_set(feats_all.loc[train_mask].copy())

        if train_df["y_home_win"].notna().sum() < 20:
            continue

        margin_model, s_margin, _ = fit_margin_model(train_df)
        total_model,  s_total,  _ = fit_total_model(train_df)

        eval_df   = make_training_set(feats_all[feats_all["season"] == target_season].copy())
        completed = eval_df[eval_df["y_home_win"].notna()].reset_index(drop=True)
        if completed.empty:
            continue

        for c in FEATURE_COLS:
            if c not in completed.columns:
                completed[c] = np.nan

        # In-season retraining: refit every 4 weeks on prior seasons + the
        # target season's already-completed weeks (mirrors live behavior;
        # sigma from the initial out-of-fold estimate is reused).
        completed = completed.sort_values("week").reset_index(drop=True)
        weeks_sorted = sorted(pd.to_numeric(completed["week"], errors="coerce").dropna().unique())
        mu_margin = np.full(len(completed), np.nan)
        mu_total  = np.full(len(completed), np.nan)
        for i in range(0, len(weeks_sorted), 4):
            batch_weeks = weeks_sorted[i:i+4]
            if i > 0:
                seen = completed[completed["week"] < batch_weeks[0]]
                tdf = pd.concat([train_df, seen], ignore_index=True)
                margin_model, _, _ = _fit_gbm(tdf, "y_margin", 0.0,  s_margin, compute_cv=False)
                total_model,  _, _ = _fit_gbm(tdf, "y_total", 44.0, s_total,  compute_cv=False)
            idx = completed.index[completed["week"].isin(batch_weeks)]
            Xb = completed.loc[idx, FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
            mu_margin[idx] = margin_model.predict(align_features_for_model(margin_model, Xb))
            mu_total[idx]  = total_model.predict(align_features_for_model(total_model,  Xb))
        p_home = win_prob_from_margin(mu_margin, s_margin)

        def _col(name):
            return pd.to_numeric(completed[name], errors="coerce").values if name in completed.columns \
                else np.full(len(completed), np.nan)

        rec = pd.DataFrame({
            "season":          completed["season"].values,
            "week":            completed["week"].values,
            "game_id":         completed["game_id"].values,
            "home_team":       completed.get("home_team", pd.Series(np.nan, index=completed.index)).values,
            "away_team":       completed.get("away_team", pd.Series(np.nan, index=completed.index)).values,
            "p_home_pred":     p_home,
            "actual_home_win": completed["y_home_win"].values,
            "pred_margin":     mu_margin,
            "actual_margin":   completed["y_margin"].values,
            "pred_total":      mu_total,
            "actual_total":    completed["y_total"].values,
            "vegas_spread":    _col("spread_line"),   # home-perspective: + = home favored
            "vegas_total":     _col("total_line"),
            "home_moneyline":  _col("home_moneyline"),
            "away_moneyline":  _col("away_moneyline"),
            "home_spread_odds": _col("home_spread_odds"),
            "away_spread_odds": _col("away_spread_odds"),
            "over_odds":        _col("over_odds"),
            "under_odds":       _col("under_odds"),
        })
        results.append(rec)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)

# ---- Season-to-date tracker: predicted winner vs actual winner ---------------

def summarize_season_to_date(bt_df: pd.DataFrame) -> dict:
    """Straight-up predicted-winner-vs-actual-winner record for every completed
    game in a walk-forward backtest (see run_backtest / TRAIN_WINDOW_SEASONS —
    each prediction only ever uses data available before that game was played,
    so this mirrors what the model would have said at kickoff). Not a betting
    record — no odds/EV involved, just "did the model pick the right team."
    """
    if bt_df is None or bt_df.empty:
        return {}
    d = bt_df.copy()
    d["week"] = pd.to_numeric(d["week"], errors="coerce")
    d["pred_winner"]   = np.where(d["p_home_pred"] > 0.5, d["home_team"], d["away_team"])
    d["actual_winner"] = np.where(d["actual_home_win"] == 1, d["home_team"], d["away_team"])
    d["correct"] = d["pred_winner"] == d["actual_winner"]
    d["home_score"] = (d["actual_total"] + d["actual_margin"]) / 2.0
    d["away_score"] =  d["actual_total"] - d["home_score"]

    n = int(len(d))
    correct = int(d["correct"].sum())
    weekly = (d.groupby("week", as_index=False)
                .agg(games=("game_id", "count"), correct=("correct", "sum")))
    weekly["record"]   = weekly.apply(lambda r: f"{int(r['correct'])}-{int(r['games']-r['correct'])}", axis=1)
    weekly["accuracy"] = weekly["correct"] / weekly["games"]

    keep = ["week","game_id","home_team","away_team","home_score","away_score",
            "p_home_pred","pred_winner","actual_winner","correct","pred_margin","actual_margin"]
    per_game = d[[c for c in keep if c in d.columns]].sort_values(["week","game_id"]).reset_index(drop=True)

    return {
        "n_games":   n,
        "correct":   correct,
        "incorrect": n - correct,
        "accuracy":  (correct / n) if n else np.nan,
        "weekly":    weekly,
        "per_game":  per_game,
    }

# ---- Model vs Vegas performance tracking -------------------------------------

def evaluate_vs_market(bt_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Compare model predictions to Vegas closing lines on completed games.

    Example: model says home by 10, Vegas closed home -8, home won by 11 →
    model error 1.0 vs Vegas error 3.0, model was closer, and the model's
    side (home, laying 8) covered.

    Returns (per_game_df, summary) where summary includes spread/total MAEs,
    how often the model was closer than the line, the model's ATS/O-U record
    when it disagrees with the line, and win rates by disagreement size.
    """
    if bt_df is None or bt_df.empty:
        return pd.DataFrame(), {}

    d = bt_df.copy()
    for c in ["pred_margin","actual_margin","vegas_spread","pred_total","actual_total","vegas_total",
              "p_home_pred","actual_home_win","home_moneyline","away_moneyline"]:
        if c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")

    # --- Spreads ---
    has_spread = d["vegas_spread"].notna() & d["actual_margin"].notna()
    d["model_spread_error"] = (d["pred_margin"]  - d["actual_margin"]).abs()
    d["vegas_spread_error"] = (d["vegas_spread"] - d["actual_margin"]).abs()
    d["model_closer_spread"] = np.where(has_spread, d["model_spread_error"] < d["vegas_spread_error"], np.nan)
    d["spread_disagree_pts"] = d["pred_margin"] - d["vegas_spread"]   # + = model likes home more than Vegas

    d["ats_pick"] = np.select(
        [~has_spread | d["spread_disagree_pts"].eq(0), d["spread_disagree_pts"] > 0],
        [None, "HOME"], default="AWAY")
    home_covered = d["actual_margin"] > d["vegas_spread"]
    push         = d["actual_margin"] == d["vegas_spread"]
    d["ats_result"] = np.select(
        [d["ats_pick"].isna(), push,
         (d["ats_pick"].eq("HOME") & home_covered) | (d["ats_pick"].eq("AWAY") & ~home_covered)],
        [None, "push", "win"], default="loss")

    # --- Totals ---
    has_total = d["vegas_total"].notna() & d["actual_total"].notna()
    d["model_total_error"] = (d["pred_total"]  - d["actual_total"]).abs()
    d["vegas_total_error"] = (d["vegas_total"] - d["actual_total"]).abs()
    d["model_closer_total"] = np.where(has_total, d["model_total_error"] < d["vegas_total_error"], np.nan)
    d["total_disagree_pts"] = d["pred_total"] - d["vegas_total"]

    d["ou_pick"] = np.select(
        [~has_total | d["total_disagree_pts"].eq(0), d["total_disagree_pts"] > 0],
        [None, "OVER"], default="UNDER")
    went_over = d["actual_total"] > d["vegas_total"]
    ou_push   = d["actual_total"] == d["vegas_total"]
    d["ou_result"] = np.select(
        [d["ou_pick"].isna(), ou_push,
         (d["ou_pick"].eq("OVER") & went_over) | (d["ou_pick"].eq("UNDER") & ~went_over)],
        [None, "push", "win"], default="loss")

    # --- Flat 1-unit betting P/L on every model pick, at closing prices --------
    def _dec_series(am: pd.Series, fallback: float = -110.0) -> pd.Series:
        return pd.to_numeric(am, errors="coerce").fillna(fallback).map(american_to_decimal)

    for c in ["home_spread_odds","away_spread_odds","over_odds","under_odds"]:
        if c not in d.columns: d[c] = np.nan

    ats_dec = _dec_series(d["home_spread_odds"].where(d["ats_pick"].eq("HOME"), d["away_spread_odds"]))
    d["ats_units"] = np.select(
        [d["ats_result"].eq("win"), d["ats_result"].eq("loss")],
        [ats_dec - 1.0, -1.0], default=0.0)
    d.loc[d["ats_pick"].isna(), "ats_units"] = np.nan

    ou_dec = _dec_series(d["over_odds"].where(d["ou_pick"].eq("OVER"), d["under_odds"]))
    d["ou_units"] = np.select(
        [d["ou_result"].eq("win"), d["ou_result"].eq("loss")],
        [ou_dec - 1.0, -1.0], default=0.0)
    d.loc[d["ou_pick"].isna(), "ou_units"] = np.nan

    # --- Moneyline (probability quality vs de-vigged market) ---
    ml_ok = d["home_moneyline"].notna() & d["away_moneyline"].notna() & d["actual_home_win"].notna()
    ph_raw = d.loc[ml_ok, "home_moneyline"].map(implied_prob_from_american)
    pa_raw = d.loc[ml_ok, "away_moneyline"].map(implied_prob_from_american)
    d.loc[ml_ok, "vegas_p_home"] = ph_raw / (ph_raw + pa_raw)

    d["ml_pick"] = np.where(ml_ok & d["p_home_pred"].notna(),
                            np.where(d["p_home_pred"] > 0.5, "HOME", "AWAY"), None)
    ml_won = (d["ml_pick"].eq("HOME") & d["actual_home_win"].eq(1)) | \
             (d["ml_pick"].eq("AWAY") & d["actual_home_win"].eq(0))
    d["ml_result"] = np.select([d["ml_pick"].isna(), ml_won], [None, "win"], default="loss")
    ml_dec = d["home_moneyline"].where(d["ml_pick"].eq("HOME"), d["away_moneyline"]).map(
        lambda a: american_to_decimal(a) if pd.notna(a) else np.nan)
    d["ml_units"] = np.select(
        [d["ml_result"].eq("win"), d["ml_result"].eq("loss")],
        [ml_dec - 1.0, -1.0], default=np.nan)

    def _rec(res: pd.Series) -> dict:
        w = int(res.eq("win").sum()); l = int(res.eq("loss").sum()); p = int(res.eq("push").sum())
        return {"wins": w, "losses": l, "pushes": p,
                "win_pct": (w/(w+l)) if (w+l) else np.nan}

    def _units_rec(res: pd.Series, units: pd.Series) -> dict:
        bets = res.isin(["win","loss"])
        n = int(bets.sum()); total = float(units[bets].sum()) if n else 0.0
        return {"n_bets": n, "units": total, "roi": (total/n) if n else np.nan, **_rec(res)}

    sp = d[has_spread]; tt = d[has_total]
    summary: dict = {
        "n_games": int(len(d)),
        "spread": {
            "n": int(len(sp)),
            "model_mae": float(sp["model_spread_error"].mean()) if len(sp) else np.nan,
            "vegas_mae": float(sp["vegas_spread_error"].mean()) if len(sp) else np.nan,
            "model_closer_pct": float(sp["model_closer_spread"].mean()) if len(sp) else np.nan,
            "ats": _rec(sp["ats_result"]),
            "ats_by_disagreement": [],
        },
        "totals": {
            "n": int(len(tt)),
            "model_mae": float(tt["model_total_error"].mean()) if len(tt) else np.nan,
            "vegas_mae": float(tt["vegas_total_error"].mean()) if len(tt) else np.nan,
            "model_closer_pct": float(tt["model_closer_total"].mean()) if len(tt) else np.nan,
            "ou": _rec(tt["ou_result"]),
            "ou_by_disagreement": [],
        },
        "moneyline": {},
        "breakeven_pct_at_minus110": 0.5238,
        "units": {
            "spread":    _units_rec(d["ats_result"], d["ats_units"]),
            "totals":    _units_rec(d["ou_result"],  d["ou_units"]),
            "moneyline": _units_rec(d["ml_result"],  d["ml_units"]),
        },
    }
    summary["units"]["total_units"] = float(sum(summary["units"][k]["units"] for k in ("spread","totals","moneyline")))
    for t in (0.0, 1.0, 2.0, 3.0, 5.0):
        s_t = sp[sp["spread_disagree_pts"].abs() >= t]
        summary["spread"]["ats_by_disagreement"].append({"min_edge_pts": t, "n": int(len(s_t)), **_rec(s_t["ats_result"])})
        t_t = tt[tt["total_disagree_pts"].abs() >= t]
        summary["totals"]["ou_by_disagreement"].append({"min_edge_pts": t, "n": int(len(t_t)), **_rec(t_t["ou_result"])})

    ml = d[ml_ok]
    if len(ml):
        summary["moneyline"] = {
            "n": int(len(ml)),
            "model_brier": float(((ml["p_home_pred"]  - ml["actual_home_win"])**2).mean()),
            "vegas_brier": float(((ml["vegas_p_home"] - ml["actual_home_win"])**2).mean()),
            "model_accuracy": float(((ml["p_home_pred"]  > 0.5) == ml["actual_home_win"]).mean()),
            "vegas_accuracy": float(((ml["vegas_p_home"] > 0.5) == ml["actual_home_win"]).mean()),
        }
    return d, summary

def run_market_eval(eval_seasons: List[int]) -> Tuple[pd.DataFrame, dict]:
    """Load data, run the walk-forward backtest, and score the model against
    Vegas closing lines. No odds-API key needed (lines come from nflverse)."""
    eval_seasons = _to_season_list(eval_seasons)
    if not eval_seasons: raise ValueError("No evaluation seasons given.")
    season_list = list(range(min(eval_seasons) - TRAIN_WINDOW_SEASONS, max(eval_seasons) + 1))
    sched  = load_schedule(season_list)
    weekly = team_weekly_from_schedule(sched)
    bt = run_backtest(sched, weekly, eval_seasons)
    return evaluate_vs_market(bt)

def format_market_summary(summary: dict) -> str:
    if not summary: return "No completed games with Vegas lines found."
    lines = [f"Games evaluated: {summary['n_games']}"]
    sp = summary.get("spread", {})
    if sp.get("n"):
        a = sp["ats"]
        lines += [
            "",
            f"SPREADS (n={sp['n']})",
            f"  Margin MAE   — model: {sp['model_mae']:.2f} pts | Vegas: {sp['vegas_mae']:.2f} pts",
            f"  Model closer to final margin than the line: {sp['model_closer_pct']:.1%}",
            f"  Model ATS when it disagrees with the line: {a['wins']}-{a['losses']}-{a['pushes']} ({a['win_pct']:.1%})" if a['wins']+a['losses'] else "  Model ATS: n/a",
            "  ATS by disagreement size (need >52.4% to beat -110):",
        ]
        for row in sp["ats_by_disagreement"]:
            wp = f"{row['win_pct']:.1%}" if row["wins"]+row["losses"] else "n/a"
            lines.append(f"    >= {row['min_edge_pts']:.0f} pts: {row['wins']}-{row['losses']}-{row['pushes']} ({wp}, n={row['n']})")
    tt = summary.get("totals", {})
    if tt.get("n"):
        o = tt["ou"]
        lines += [
            "",
            f"TOTALS (n={tt['n']})",
            f"  Total MAE    — model: {tt['model_mae']:.2f} pts | Vegas: {tt['vegas_mae']:.2f} pts",
            f"  Model closer to final total than the line: {tt['model_closer_pct']:.1%}",
            f"  Model O/U when it disagrees with the line: {o['wins']}-{o['losses']}-{o['pushes']} ({o['win_pct']:.1%})" if o['wins']+o['losses'] else "  Model O/U: n/a",
        ]
    ml = summary.get("moneyline", {})
    if ml.get("n"):
        lines += [
            "",
            f"MONEYLINE (n={ml['n']})",
            f"  Brier score  — model: {ml['model_brier']:.4f} | Vegas: {ml['vegas_brier']:.4f} (lower is better)",
            f"  Win accuracy — model: {ml['model_accuracy']:.1%} | Vegas: {ml['vegas_accuracy']:.1%}",
        ]
    u = summary.get("units", {})
    if u:
        lines += ["", "SCOREBOARD — 1 unit on every model pick (closing prices):"]
        for key, name in [("spread","Spread"),("totals","Totals"),("moneyline","Moneyline")]:
            b = u.get(key, {})
            if b.get("n_bets"):
                lines.append(f"  {name:<9} {b['wins']}-{b['losses']}-{b['pushes']}: "
                             f"{b['units']:+.1f} units over {b['n_bets']} bets ({b['roi']:+.1%} ROI)")
        lines.append(f"  {'TOTAL':<9} {u.get('total_units', 0.0):+.1f} units")
    return "\n".join(lines)
