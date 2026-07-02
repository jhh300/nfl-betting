"""
Basketball (NBA + NCAAB) data, model, and odds utilities.

Mirrors fetch_nfl_data.py for the football betting tool — fully self-contained
so neither file needs to be modified.

Data sources
------------
  NBA   — nba_api (pip install nba_api)
  NCAAB — ESPN public scoreboard API (no auth required)
  Odds  — The Odds API (same key as the NFL tool)

Sport keys (The Odds API)
-------------------------
  NBA   : basketball_nba
  NCAAB : basketball_ncaab
"""
from __future__ import annotations

import datetime
import json
import math
import os
import re
import ssl
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ─────────────────────────────────────────────────────────────────────────────
# SSL bootstrap (same pattern as fetch_nfl_data)
# ─────────────────────────────────────────────────────────────────────────────

def _ssl_bootstrap() -> None:
    try:
        import certifi
        ssl._create_default_https_context = (  # type: ignore[attr-defined]
            lambda: ssl.create_default_context(cafile=certifi.where())
        )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Pure-math utilities (identical logic to fetch_nfl_data, sport-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def american_to_decimal(a: float) -> float:
    a = float(a)
    return 1.0 + (a / 100.0) if a > 0 else 1.0 + (100.0 / abs(a))


def implied_prob_from_american(a: float) -> float:
    a = float(a)
    return 100.0 / (a + 100.0) if a > 0 else (-a) / (100.0 - a)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _kelly_fraction(p: float, dec: float) -> float:
    b = max(dec - 1.0, 1e-9)
    q = 1.0 - p
    return float(max(0.0, (b * p - q) / b))


# ─────────────────────────────────────────────────────────────────────────────
# NBA team-name canonicalization
# ─────────────────────────────────────────────────────────────────────────────

_NBA_FULL: Dict[str, str] = {
    "ATLANTA HAWKS": "ATL", "BOSTON CELTICS": "BOS", "BROOKLYN NETS": "BKN",
    "CHARLOTTE HORNETS": "CHA", "CHICAGO BULLS": "CHI", "CLEVELAND CAVALIERS": "CLE",
    "DALLAS MAVERICKS": "DAL", "DENVER NUGGETS": "DEN", "DETROIT PISTONS": "DET",
    "GOLDEN STATE WARRIORS": "GSW", "HOUSTON ROCKETS": "HOU", "INDIANA PACERS": "IND",
    "LOS ANGELES CLIPPERS": "LAC", "LOS ANGELES LAKERS": "LAL",
    "LA CLIPPERS": "LAC", "LA LAKERS": "LAL",
    "MEMPHIS GRIZZLIES": "MEM", "MIAMI HEAT": "MIA", "MILWAUKEE BUCKS": "MIL",
    "MINNESOTA TIMBERWOLVES": "MIN", "NEW ORLEANS PELICANS": "NOP",
    "NEW YORK KNICKS": "NYK", "OKLAHOMA CITY THUNDER": "OKC",
    "ORLANDO MAGIC": "ORL", "PHILADELPHIA 76ERS": "PHI",
    "PHOENIX SUNS": "PHX", "PORTLAND TRAIL BLAZERS": "POR",
    "SACRAMENTO KINGS": "SAC", "SAN ANTONIO SPURS": "SAS",
    "TORONTO RAPTORS": "TOR", "UTAH JAZZ": "UTA", "WASHINGTON WIZARDS": "WAS",
    # Historical / relocated
    "NEW JERSEY NETS": "BKN", "SEATTLE SUPERSONICS": "OKC",
    "NEW ORLEANS HORNETS": "NOP", "CHARLOTTE BOBCATS": "CHA",
    "NEW ORLEANS/OKLAHOMA CITY HORNETS": "NOP",
}
_NBA_NICK: Dict[str, str] = {
    "HAWKS": "ATL", "CELTICS": "BOS", "NETS": "BKN", "HORNETS": "CHA",
    "BULLS": "CHI", "CAVALIERS": "CLE", "CAVS": "CLE", "MAVERICKS": "DAL",
    "MAVS": "DAL", "NUGGETS": "DEN", "PISTONS": "DET", "WARRIORS": "GSW",
    "ROCKETS": "HOU", "PACERS": "IND", "CLIPPERS": "LAC", "LAKERS": "LAL",
    "GRIZZLIES": "MEM", "HEAT": "MIA", "BUCKS": "MIL",
    "TIMBERWOLVES": "MIN", "WOLVES": "MIN", "PELICANS": "NOP",
    "KNICKS": "NYK", "THUNDER": "OKC", "MAGIC": "ORL",
    "76ERS": "PHI", "SIXERS": "PHI", "SUNS": "PHX", "BLAZERS": "POR",
    "KINGS": "SAC", "SPURS": "SAS", "RAPTORS": "TOR", "JAZZ": "UTA",
    "WIZARDS": "WAS",
}
_NBA_VALID = set(_NBA_FULL.values())

# Full display names keyed by abbreviation (for readable output)
_NBA_DISPLAY: Dict[str, str] = {
    "ATL": "Atlanta Hawks",        "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",        "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",        "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",     "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",      "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",      "IND": "Indiana Pacers",
    "LAC": "LA Clippers",          "LAL": "LA Lakers",
    "MEM": "Memphis Grizzlies",    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",      "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",   "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers","SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",            "WAS": "Washington Wizards",
}


def display_team_name(abbr: str, sport: str = "nba") -> str:
    """Return a human-readable team name for display."""
    if pd.isna(abbr):
        return abbr
    if sport == "ncaab":
        return str(abbr).title()
    s = str(abbr).strip()
    su = s.upper()
    # If it's already a full name (e.g., from odds API), return it nicely cased
    if su in _NBA_FULL:
        return s.title()
    # Otherwise treat as abbreviation and look up display name
    return _NBA_DISPLAY.get(su, s)


def canonical_nba_team(x: str) -> str:
    if pd.isna(x):
        return x
    s = re.sub(r"\s+", " ", str(x).upper().strip())
    if s in _NBA_VALID:
        return s
    if s in _NBA_FULL:
        return _NBA_FULL[s]
    return _NBA_NICK.get(s.split()[-1] if s else s, s)


# ─────────────────────────────────────────────────────────────────────────────
# NCAAB — normalised full name (no fixed abbreviation set; hundreds of teams)
# ─────────────────────────────────────────────────────────────────────────────

def canonical_ncaab_team(x: str) -> str:
    if pd.isna(x):
        return x
    return re.sub(r"\s+", " ", str(x).strip().upper())


# ─────────────────────────────────────────────────────────────────────────────
# NBA data loading via nba_api
# ─────────────────────────────────────────────────────────────────────────────

def _nba_season_str(season: int) -> str:
    """Convert season year (2025) → '2024-25'."""
    return f"{season - 1}-{season % 100:02d}"


def load_nba_game_logs(seasons: List[int]) -> pd.DataFrame:
    """
    Load per-team, per-game results from nba_api.LeagueGameLog.

    Returns columns: team, game_id, game_date, pts_scored, pts_allowed, is_home, season
    """
    try:
        from nba_api.stats.endpoints import LeagueGameLog  # type: ignore
    except ImportError:
        raise ImportError(
            "nba_api is required for NBA data. Install with:\n"
            "  pip install nba_api"
        )

    frames = []
    for season in seasons:
        season_str = _nba_season_str(season)
        try:
            time.sleep(0.7)  # courtesy rate-limit
            gl = LeagueGameLog(season=season_str, player_or_team="T", timeout=60)
            df = gl.get_data_frames()[0]
            df["_season"] = season
            frames.append(df)
        except Exception as e:
            print(f"[NBA] Warning: could not load season {season_str}: {e}")

    if not frames:
        return pd.DataFrame(
            columns=["team", "game_id", "game_date", "pts_scored", "pts_allowed", "is_home", "season"]
        )

    raw = pd.concat(frames, ignore_index=True)
    # nba_api columns are uppercase
    raw.columns = [c.upper() for c in raw.columns]

    for col in ["TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "PTS", "PLUS_MINUS", "MATCHUP", "_SEASON"]:
        if col not in raw.columns:
            raw[col] = np.nan

    out = raw[["TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "PTS", "PLUS_MINUS", "MATCHUP", "_SEASON"]].copy()
    out.columns = ["team", "game_id", "game_date", "pts_scored", "plus_minus", "matchup", "season"]

    out["team"] = out["team"].apply(canonical_nba_team)
    out["pts_scored"] = pd.to_numeric(out["pts_scored"], errors="coerce")
    out["plus_minus"] = pd.to_numeric(out["plus_minus"], errors="coerce")
    # PLUS_MINUS = pts_scored - pts_allowed  →  pts_allowed = pts_scored - PLUS_MINUS
    out["pts_allowed"] = out["pts_scored"] - out["plus_minus"]
    # "BOS vs. MIA" = home;  "BOS @ MIA" = away
    out["is_home"] = out["matchup"].str.contains(r"vs\.", na=False)
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce", utc=True)
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")

    return out.sort_values(["team", "game_date"]).reset_index(drop=True)


def load_nba_advanced_stats(seasons: List[int]) -> pd.DataFrame:
    """
    Load season-level advanced team stats (OffRtg, DefRtg, Pace, eFG%, TOV%, OREB%)
    from nba_api.LeagueDashTeamStats.  One row per team per season.
    """
    try:
        from nba_api.stats.endpoints import LeagueDashTeamStats  # type: ignore
    except ImportError:
        return pd.DataFrame()

    adv_frames: List[pd.DataFrame] = []
    ff_frames:  List[pd.DataFrame] = []

    for season in seasons:
        season_str = _nba_season_str(season)
        for measure, store in [("Advanced", adv_frames), ("Four Factors", ff_frames)]:
            try:
                time.sleep(0.7)
                resp = LeagueDashTeamStats(
                    season=season_str,
                    measure_type_simple=measure,
                    per_mode_simple="PerGame",
                    timeout=60,
                )
                df = resp.get_data_frames()[0]
                df["_season"] = season
                store.append(df)
            except Exception as exc:
                print(f"[NBA Advanced/{measure}] season {season_str}: {exc}")

    if not adv_frames and not ff_frames:
        return pd.DataFrame()

    def _norm(frames: List[pd.DataFrame]) -> pd.DataFrame:
        out = pd.concat(frames, ignore_index=True)
        out.columns = [c.upper() for c in out.columns]
        return out

    col_map  = {"TEAM_ABBREVIATION": "team", "_SEASON": "season"}
    adv_cols = {"OFF_RATING": "off_rtg", "DEF_RATING": "def_rtg",
                "NET_RATING": "net_rtg", "PACE": "pace", "TS_PCT": "ts_pct"}
    ff_cols  = {"EFG_PCT": "efg_pct", "TM_TOV_PCT": "tov_pct", "OREB_PCT": "oreb_pct"}

    parts: List[pd.DataFrame] = []
    for frames, extra in [(adv_frames, adv_cols), (ff_frames, ff_cols)]:
        if not frames:
            continue
        raw = _norm(frames)
        wanted = {**col_map, **extra}
        for c in wanted:
            if c not in raw.columns:
                raw[c] = np.nan
        sub = raw[list(wanted.keys())].copy()
        sub.columns = list(wanted.values())
        sub["team"]   = sub["team"].apply(canonical_nba_team)
        sub["season"] = pd.to_numeric(sub["season"], errors="coerce").astype("Int64")
        for c in list(extra.values()):
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        parts.append(sub)

    if not parts:
        return pd.DataFrame()
    if len(parts) == 1:
        return parts[0]
    return parts[0].merge(parts[1], on=["team", "season"], how="outer").reset_index(drop=True)


def fetch_injury_report(sport: str = "nba") -> pd.DataFrame:
    """
    Fetch current injury report from ESPN public API.
    Returns: team (canonical abbr), player, position, status, details
    """
    _ssl_bootstrap()
    sport_path = "nba" if sport.lower() == "nba" else "mens-college-basketball"
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/{sport_path}/injuries"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return pd.DataFrame(columns=["team", "player", "position", "status", "details"])

    canon_fn = canonical_nba_team if sport.lower() == "nba" else canonical_ncaab_team
    rows: List[dict] = []
    items = data if isinstance(data, list) else data.get("injuries", [])
    for entry in items:
        team_obj = entry.get("team", {})
        raw_abbr = team_obj.get("abbreviation") or team_obj.get("displayName", "")
        team_abbr = canon_fn(raw_abbr)
        for inj in entry.get("injuries", []):
            athlete = inj.get("athlete", {})
            rows.append({
                "team":     team_abbr,
                "player":   athlete.get("displayName", ""),
                "position": (athlete.get("position") or {}).get("abbreviation", ""),
                "status":   inj.get("status", ""),
                "details":  inj.get("longComment") or inj.get("shortComment") or "",
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["team", "player", "position", "status", "details"]
    )


def load_nba_schedule(seasons: List[int]) -> pd.DataFrame:
    """
    Build game-level schedule (home_team, away_team, home_score, away_score, game_dt)
    from NBA game logs.  Only returns completed games.
    """
    logs = load_nba_game_logs(seasons)
    if logs.empty:
        return pd.DataFrame(
            columns=["game_id", "season", "home_team", "away_team", "home_score", "away_score", "game_dt"]
        )

    home_logs = logs[logs["is_home"]].copy()
    away_logs = logs[~logs["is_home"]].copy()

    home_side = home_logs[["game_id", "season", "game_date", "team", "pts_scored", "pts_allowed"]].rename(
        columns={"team": "home_team", "pts_scored": "home_score", "pts_allowed": "away_score", "game_date": "game_dt"}
    )
    away_side = away_logs[["game_id", "team"]].rename(columns={"team": "away_team"})

    sched = home_side.merge(away_side, on="game_id", how="inner")
    sched["home_team"] = sched["home_team"].apply(canonical_nba_team)
    sched["away_team"] = sched["away_team"].apply(canonical_nba_team)
    return sched.sort_values("game_dt").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# ESPN public API helpers (used for NCAAB and for upcoming-game schedules)
# ─────────────────────────────────────────────────────────────────────────────

_ESPN_NCAAB = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
)
_ESPN_NBA = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"


def _espn_events_for_date(sport: str, date: datetime.date) -> List[dict]:
    """Fetch ESPN scoreboard events for one date."""
    url = _ESPN_NCAAB if sport == "ncaab" else _ESPN_NBA
    params: Dict[str, str] = {"dates": date.strftime("%Y%m%d"), "limit": "500"}
    if sport == "ncaab":
        params["groups"] = "50"  # D-I Men's Basketball only
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("events", [])
    except Exception:
        return []


def _parse_espn_events(events: List[dict], completed_only: bool = False) -> List[dict]:
    """Parse ESPN event list into flat game records."""
    rows = []
    for ev in events:
        try:
            comps = ev.get("competitions", [{}])
            if not comps:
                continue
            comp = comps[0]
            is_complete = comp.get("status", {}).get("type", {}).get("completed", False)
            if completed_only and not is_complete:
                continue
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue
            rows.append({
                "game_id":    ev.get("id", ""),
                "game_date":  comp.get("date") or ev.get("date"),
                "home_team":  home.get("team", {}).get("displayName", ""),
                "away_team":  away.get("team", {}).get("displayName", ""),
                "home_score": home.get("score") if is_complete else None,
                "away_score": away.get("score") if is_complete else None,
                "completed":  is_complete,
            })
        except Exception:
            continue
    return rows


def _season_date_range(sport: str, season: int) -> Tuple[datetime.date, datetime.date]:
    """Return (start, end) for a season's date range."""
    if sport == "nba":
        start = datetime.date(season - 1, 10, 1)
        end   = min(datetime.date.today(), datetime.date(season, 6, 30))
    else:  # ncaab
        start = datetime.date(season - 1, 11, 1)
        end   = min(datetime.date.today(), datetime.date(season, 4, 15))
    return start, end


def _load_espn_schedule(
    sport: str,
    seasons: List[int],
    cache_dir: Optional[str] = None,
    completed_only: bool = False,
) -> pd.DataFrame:
    """
    Load schedule + results for a basketball sport from ESPN, day-by-day.

    Results are cached to cache_dir (one JSON file per sport+season) so
    subsequent runs are fast.
    """
    _ssl_bootstrap()
    canon_fn = canonical_ncaab_team if sport == "ncaab" else canonical_nba_team
    all_rows: List[dict] = []

    for season in seasons:
        start, end = _season_date_range(sport, season)

        # ── Try file cache ──
        cache_file = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"{sport}_sched_{season}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file) as f:
                        rows = json.load(f)
                    for r in rows:
                        r["_season"] = season
                    all_rows.extend(rows)
                    continue
                except Exception:
                    pass

        # ── Fetch day by day ──
        season_rows: List[dict] = []
        curr = start
        while curr <= end:
            events = _espn_events_for_date(sport, curr)
            parsed = _parse_espn_events(events, completed_only=completed_only)
            for r in parsed:
                r["_season"] = season
            season_rows.extend(parsed)
            curr += datetime.timedelta(days=1)
            time.sleep(0.04)

        if cache_file and season_rows:
            try:
                with open(cache_file, "w") as f:
                    json.dump(season_rows, f)
            except Exception:
                pass

        all_rows.extend(season_rows)

    if not all_rows:
        return pd.DataFrame(
            columns=["game_id", "season", "home_team", "away_team", "home_score", "away_score", "game_dt", "completed"]
        )

    df = pd.DataFrame(all_rows).rename(columns={"_season": "season"})
    df["game_dt"]    = pd.to_datetime(df["game_date"], errors="coerce", utc=True)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["season"]     = pd.to_numeric(df["season"],     errors="coerce").astype("Int64")
    df["home_team"]  = df["home_team"].apply(canon_fn)
    df["away_team"]  = df["away_team"].apply(canon_fn)
    df = df.drop(columns=["game_date"], errors="ignore")
    df = df.drop_duplicates(subset=["game_id"]).sort_values("game_dt").reset_index(drop=True)
    return df


def load_upcoming_games_espn(sport: str, days_ahead: int = 7) -> pd.DataFrame:
    """
    Fetch the next `days_ahead` days of upcoming games from ESPN.
    Returns rows without scores (home_score / away_score are NaN).
    """
    _ssl_bootstrap()
    canon_fn = canonical_ncaab_team if sport == "ncaab" else canonical_nba_team
    today = datetime.date.today()
    rows: List[dict] = []
    current_season = today.year if today.month >= 10 else today.year  # NBA/NCAAB season end year

    for i in range(days_ahead + 1):
        d = today + datetime.timedelta(days=i)
        events = _espn_events_for_date(sport, d)
        for r in _parse_espn_events(events, completed_only=False):
            if not r["completed"]:
                rows.append(r)

    if not rows:
        return pd.DataFrame(
            columns=["game_id", "season", "home_team", "away_team", "home_score", "away_score", "game_dt"]
        )

    df = pd.DataFrame(rows)
    df["game_dt"]   = pd.to_datetime(df["game_date"], errors="coerce", utc=True)
    df["home_score"] = np.nan
    df["away_score"] = np.nan
    df["season"]     = current_season
    df["home_team"]  = df["home_team"].apply(canon_fn)
    df["away_team"]  = df["away_team"].apply(canon_fn)
    df = df.drop(columns=["game_date", "completed"], errors="ignore")
    df = df.drop_duplicates(subset=["game_id"]).sort_values("game_dt").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Unified schedule loader
# ─────────────────────────────────────────────────────────────────────────────

def load_schedule(
    sport: str,
    seasons: List[int],
    cache_dir: Optional[str] = None,
    include_upcoming: bool = True,
) -> pd.DataFrame:
    """
    Load game schedule for sport ('nba' or 'ncaab') and given seasons.

    For NBA:   historical data via nba_api + upcoming games via ESPN
    For NCAAB: all data via ESPN public API
    """
    sport = sport.lower()
    if sport == "nba":
        hist = load_nba_schedule(seasons)
        if include_upcoming:
            upcoming = load_upcoming_games_espn("nba", days_ahead=7)
            if not upcoming.empty:
                hist = pd.concat([hist, upcoming], ignore_index=True)
    else:
        hist = _load_espn_schedule("ncaab", seasons, cache_dir=cache_dir, completed_only=False)
        if include_upcoming:
            upcoming = load_upcoming_games_espn("ncaab", days_ahead=7)
            if not upcoming.empty:
                hist = pd.concat([hist, upcoming], ignore_index=True)

    hist = hist.drop_duplicates(subset=["game_id"]).reset_index(drop=True)
    return hist


# ─────────────────────────────────────────────────────────────────────────────
# Rolling team statistics
# ─────────────────────────────────────────────────────────────────────────────

def build_team_rolling_stats(sched: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-team rolling stats from the schedule.

    Only uses rows that have actual scores (completed games).
    Returns per-team, per-game rolling stats at 4-game and 10-game windows,
    plus win rate and season game number.
    """
    _empty_cols = [
        "team", "game_id", "season", "game_dt",
        "pts_scored_roll4", "pts_allowed_roll4",
        "pts_scored_roll10", "pts_allowed_roll10",
        "win_rate_roll4", "win_rate_roll10",
        "season_game_num",
        "win_rate_home_roll6", "win_rate_away_roll6",
    ]
    if sched.empty:
        return pd.DataFrame(columns=_empty_cols)

    completed = sched[sched["home_score"].notna() & sched["away_score"].notna()].copy()
    if completed.empty:
        return pd.DataFrame(columns=_empty_cols)

    home_rows = completed[["game_id", "season", "game_dt", "home_team", "home_score", "away_score"]].copy()
    home_rows.columns = ["game_id", "season", "game_dt", "team", "pts_scored", "pts_allowed"]
    home_rows["is_home_game"] = 1

    away_rows = completed[["game_id", "season", "game_dt", "away_team", "away_score", "home_score"]].copy()
    away_rows.columns = ["game_id", "season", "game_dt", "team", "pts_scored", "pts_allowed"]
    away_rows["is_home_game"] = 0

    logs = pd.concat([home_rows, away_rows], ignore_index=True)
    logs = logs.sort_values(["team", "season", "game_dt"]).reset_index(drop=True)
    logs["win"] = (
        pd.to_numeric(logs["pts_scored"], errors="coerce") >
        pd.to_numeric(logs["pts_allowed"], errors="coerce")
    ).astype(float)

    grp = logs.groupby(["team", "season"], group_keys=False)

    for window, suffix in [(4, "roll4"), (10, "roll10")]:
        logs[f"pts_scored_{suffix}"] = grp["pts_scored"].transform(
            lambda s: pd.to_numeric(s, errors="coerce").rolling(window, min_periods=1).mean()
        )
        logs[f"pts_allowed_{suffix}"] = grp["pts_allowed"].transform(
            lambda s: pd.to_numeric(s, errors="coerce").rolling(window, min_periods=1).mean()
        )
        logs[f"win_rate_{suffix}"] = grp["win"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )

    logs["season_game_num"] = grp.cumcount() + 1

    # Venue-specific rolling win rates (home-game wins / away-game wins)
    for venue_val, venue_col in [(1, "win_rate_home_roll6"), (0, "win_rate_away_roll6")]:
        venue_logs = (
            logs[logs["is_home_game"] == venue_val]
            .sort_values(["team", "season", "game_dt"])
            .copy()
        )
        venue_logs[venue_col] = venue_logs.groupby(["team", "season"], group_keys=False)["win"].transform(
            lambda s: s.rolling(6, min_periods=1).mean()
        )
        logs = logs.merge(
            venue_logs[["game_id", "team", venue_col]], on=["game_id", "team"], how="left"
        )

    return logs[_empty_cols].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Feature columns and building
# ─────────────────────────────────────────────────────────────────────────────

# All rolling stat columns produced by build_team_rolling_stats (per team)
BBALL_ROLL_STATS = [
    "pts_scored_roll4", "pts_allowed_roll4",
    "pts_scored_roll10", "pts_allowed_roll10",
    "win_rate_roll4", "win_rate_roll10",
    "season_game_num",
    "win_rate_home_roll6", "win_rate_away_roll6",
]
# Kept for backwards compat in any code that uses BBALL_BASE_ROLL4S
BBALL_BASE_ROLL4S = ["pts_scored_roll4", "pts_allowed_roll4"]

# Pts/allowed get both diff and sum (sum matters for totals); win rate / game num get diff only
_BBALL_SUM_STATS = ["pts_scored_roll4", "pts_allowed_roll4", "pts_scored_roll10", "pts_allowed_roll10"]
_BBALL_DIFF_STATS = [
    "pts_scored_roll4", "pts_allowed_roll4",
    "pts_scored_roll10", "pts_allowed_roll10",
    "win_rate_roll4", "win_rate_roll10",
    "season_game_num",
]

# Season-level advanced stat diffs (NBA only; NaN for NCAAB → dropped from model automatically)
_BBALL_ADV_DIFF = ["off_rtg", "def_rtg", "net_rtg", "pace", "efg_pct", "tov_pct", "oreb_pct"]

BBALL_FEATURE_COLS = (
    [f"{s}_diff" for s in _BBALL_DIFF_STATS] +
    [f"{s}_sum"  for s in _BBALL_SUM_STATS] +
    [f"{s}_diff" for s in _BBALL_ADV_DIFF] +
    [
        "pace_sum",
        "pyth_diff", "pyth10_diff",
        "pts_margin_roll4_diff", "pts_margin_roll10_diff",
        "rest_days_diff", "home_b2b", "away_b2b",
        "elo_diff",
        # Venue-specific win rates (absolute, not diff — home team's home form, away team's road form)
        "home_win_rate_home_roll6", "away_win_rate_away_roll6",
    ]
)


def _add_rest_days(sched: pd.DataFrame) -> pd.DataFrame:
    out = sched.copy()
    if "game_dt" not in out.columns or out["game_dt"].isna().all():
        out["home_rest_days"] = np.nan
        out["away_rest_days"] = np.nan
        return out

    s = out.sort_values("game_dt")
    h = s[["game_id", "home_team", "game_dt"]].rename(columns={"home_team": "team"})
    a = s[["game_id", "away_team", "game_dt"]].rename(columns={"away_team": "team"})
    tg = pd.concat([h, a], ignore_index=True).sort_values("game_dt")
    tg["game_dt"] = pd.to_datetime(tg["game_dt"], utc=True, errors="coerce")
    tg["prev_dt"] = tg.groupby("team")["game_dt"].shift(1)
    tg["rest_days"] = (
        (tg["game_dt"] - tg["prev_dt"]).dt.total_seconds().div(86400.0).fillna(2.0)
    )

    hr = (
        tg.merge(s[["game_id", "home_team"]], on="game_id", how="inner")
          .loc[lambda d: d["team"] == d["home_team"], ["game_id", "rest_days"]]
          .rename(columns={"rest_days": "home_rest_days"})
    )
    ar = (
        tg.merge(s[["game_id", "away_team"]], on="game_id", how="inner")
          .loc[lambda d: d["team"] == d["away_team"], ["game_id", "rest_days"]]
          .rename(columns={"rest_days": "away_rest_days"})
    )

    out = out.merge(hr, on="game_id", how="left").merge(ar, on="game_id", how="left")
    return out


def build_bball_game_features(
    sched: pd.DataFrame,
    team_stats: pd.DataFrame,
    pyth_exp: float = 14.0,
    advanced_stats: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build game-level feature matrix (one row per game).

    team_stats is produced by build_team_rolling_stats and contains roll4/roll10
    stats, win rate, and season game number per team per game.

    advanced_stats (optional) is produced by load_nba_advanced_stats and provides
    season-level OffRtg, DefRtg, Pace, eFG%, TOV%, OREB% per team per season.
    """
    s = _add_rest_days(sched.copy())
    ts = team_stats.copy()

    # Stat columns we want to pull for home and away sides
    stat_cols = [c for c in BBALL_ROLL_STATS if c in ts.columns]

    home_ts = ts.rename(columns={c: f"home_{c}" for c in stat_cols})
    away_ts = ts.rename(columns={c: f"away_{c}" for c in stat_cols})

    feats = s.merge(
        home_ts[["game_id", "team"] + [f"home_{c}" for c in stat_cols]],
        left_on=["game_id", "home_team"], right_on=["game_id", "team"], how="left",
    ).drop(columns=["team"], errors="ignore")

    feats = feats.merge(
        away_ts[["game_id", "team"] + [f"away_{c}" for c in stat_cols]],
        left_on=["game_id", "away_team"], right_on=["game_id", "team"], how="left",
    ).drop(columns=["team"], errors="ignore")

    # For upcoming games, backfill with latest known stats per team
    latest_stats = (
        ts.sort_values("game_dt")
          .groupby("team", group_keys=False)
          .last()
          .reset_index()[["team"] + stat_cols]
    )
    for side in ("home_", "away_"):
        side_team = f"{side}team"
        if side_team not in feats.columns:
            continue
        side_latest = latest_stats.rename(
            columns={"team": side_team, **{c: f"{side}{c}" for c in stat_cols}}
        )
        for stub in stat_cols:
            col = f"{side}{stub}"
            if col not in feats.columns:
                feats[col] = np.nan
            tmp = feats[[side_team]].merge(
                side_latest[[side_team, col]].rename(columns={col: f"_fill_{col}"}),
                on=side_team, how="left",
            )
            feats[col] = feats[col].where(feats[col].notna(), tmp[f"_fill_{col}"].values)

    # Diff and sum features
    for stub in _BBALL_DIFF_STATS:
        if stub not in stat_cols:
            continue
        h = pd.to_numeric(feats.get(f"home_{stub}", np.nan), errors="coerce")
        a = pd.to_numeric(feats.get(f"away_{stub}", np.nan), errors="coerce")
        feats[f"{stub}_diff"] = h - a
        if stub in _BBALL_SUM_STATS:
            feats[f"{stub}_sum"] = h + a

    # Rolling scoring margin (explicit diff signal)
    for suffix in ("roll4", "roll10"):
        for side in ("home_", "away_"):
            ps = pd.to_numeric(feats.get(f"{side}pts_scored_{suffix}", np.nan), errors="coerce")
            pa = pd.to_numeric(feats.get(f"{side}pts_allowed_{suffix}", np.nan), errors="coerce")
            feats[f"{side}pts_margin_{suffix}"] = ps - pa
        feats[f"pts_margin_{suffix}_diff"] = (
            feats[f"home_pts_margin_{suffix}"] - feats[f"away_pts_margin_{suffix}"]
        )

    # Pythagorean expectation (roll4 and roll10)
    for side in ("home_", "away_"):
        ps4  = pd.to_numeric(feats.get(f"{side}pts_scored_roll4",  np.nan), errors="coerce")
        pa4  = pd.to_numeric(feats.get(f"{side}pts_allowed_roll4",  np.nan), errors="coerce")
        ps10 = pd.to_numeric(feats.get(f"{side}pts_scored_roll10", np.nan), errors="coerce")
        pa10 = pd.to_numeric(feats.get(f"{side}pts_allowed_roll10", np.nan), errors="coerce")
        feats[f"{side}pyth"]   = ps4  ** pyth_exp / (ps4  ** pyth_exp + pa4  ** pyth_exp + 1e-9)
        feats[f"{side}pyth10"] = ps10 ** pyth_exp / (ps10 ** pyth_exp + pa10 ** pyth_exp + 1e-9)
    feats["pyth_diff"]   = feats["home_pyth"]   - feats["away_pyth"]
    feats["pyth10_diff"] = feats["home_pyth10"] - feats["away_pyth10"]

    # Venue-specific win rates:
    # home team's rolling home-game win rate  → feature: home_win_rate_home_roll6
    # away team's rolling away-game win rate  → feature: away_win_rate_away_roll6
    feats["home_win_rate_home_roll6"] = pd.to_numeric(
        feats.get("home_win_rate_home_roll6", np.nan), errors="coerce"
    )
    feats["away_win_rate_away_roll6"] = pd.to_numeric(
        feats.get("away_win_rate_away_roll6", np.nan), errors="coerce"
    )

    # ELO ratings (computed from the same schedule, no external data needed)
    try:
        elo_df = compute_elo_ratings(sched)
        if not elo_df.empty and "game_id" in feats.columns:
            feats = feats.merge(elo_df[["game_id", "elo_diff"]], on="game_id", how="left")
        else:
            feats["elo_diff"] = np.nan
    except Exception:
        feats["elo_diff"] = np.nan

    # Rest-days differential + back-to-back flags
    if "home_rest_days" in feats.columns and "away_rest_days" in feats.columns:
        hr = pd.to_numeric(feats["home_rest_days"], errors="coerce")
        ar = pd.to_numeric(feats["away_rest_days"], errors="coerce")
        feats["rest_days_diff"] = hr - ar
        feats["home_b2b"] = (hr <= 1).astype(float)
        feats["away_b2b"] = (ar <= 1).astype(float)
    else:
        feats["rest_days_diff"] = np.nan
        feats["home_b2b"] = np.nan
        feats["away_b2b"] = np.nan

    # Season-level advanced stats (NBA only; silently skipped for NCAAB)
    if advanced_stats is not None and not advanced_stats.empty:
        adv_cols = [c for c in _BBALL_ADV_DIFF if c in advanced_stats.columns]
        if adv_cols and "season" in feats.columns:
            home_adv = advanced_stats[["team", "season"] + adv_cols].rename(
                columns={"team": "home_team", **{c: f"home_{c}" for c in adv_cols}}
            )
            away_adv = advanced_stats[["team", "season"] + adv_cols].rename(
                columns={"team": "away_team", **{c: f"away_{c}" for c in adv_cols}}
            )
            feats["season"] = pd.to_numeric(feats["season"], errors="coerce").astype("Int64")
            feats = feats.merge(home_adv, on=["home_team", "season"], how="left")
            feats = feats.merge(away_adv, on=["away_team", "season"], how="left")
            for col in adv_cols:
                h = pd.to_numeric(feats.get(f"home_{col}", np.nan), errors="coerce")
                a = pd.to_numeric(feats.get(f"away_{col}", np.nan), errors="coerce")
                feats[f"{col}_diff"] = h - a
            # pace_sum: total possessions in game — key for totals model
            if "home_pace" in feats.columns and "away_pace" in feats.columns:
                feats["pace_sum"] = (
                    pd.to_numeric(feats["home_pace"], errors="coerce") +
                    pd.to_numeric(feats["away_pace"], errors="coerce")
                )
            else:
                feats["pace_sum"] = np.nan
        else:
            for col in _BBALL_ADV_DIFF:
                feats[f"{col}_diff"] = np.nan
            feats["pace_sum"] = np.nan
    else:
        for col in _BBALL_ADV_DIFF:
            feats[f"{col}_diff"] = np.nan
        feats["pace_sum"] = np.nan

    return feats


def compute_elo_ratings(
    sched: pd.DataFrame,
    k: float = 20.0,
    home_advantage: float = 100.0,
    initial: float = 1500.0,
) -> pd.DataFrame:
    """
    Compute pre-game ELO ratings for every game in sched.

    Returns one row per game_id with columns: game_id, elo_home, elo_away, elo_diff.
    Uses numpy arrays for performance (10-20x faster than iterrows for large schedules).
    """
    if sched.empty or "home_score" not in sched.columns:
        return pd.DataFrame(columns=["game_id", "elo_home", "elo_away", "elo_diff"])

    completed = (
        sched[sched["home_score"].notna() & sched["away_score"].notna()]
        .sort_values("game_dt")
        .reset_index(drop=True)
    )

    # Extract to numpy for fast sequential updates
    game_ids_c  = completed["game_id"].values
    home_arr    = completed["home_team"].values
    away_arr    = completed["away_team"].values
    hs_arr      = pd.to_numeric(completed["home_score"], errors="coerce").fillna(0).values
    as_arr      = pd.to_numeric(completed["away_score"], errors="coerce").fillna(0).values

    ratings: Dict[str, float] = {}
    pre_elo_home = np.empty(len(completed), dtype=float)
    pre_elo_away = np.empty(len(completed), dtype=float)

    for i in range(len(completed)):
        h, a = home_arr[i], away_arr[i]
        elo_h = ratings.get(h, initial)
        elo_a = ratings.get(a, initial)
        pre_elo_home[i] = elo_h
        pre_elo_away[i] = elo_a
        exp_h   = 1.0 / (1.0 + 10.0 ** ((elo_a - (elo_h + home_advantage)) / 400.0))
        actual_h = 1.0 if hs_arr[i] > as_arr[i] else 0.0
        delta    = k * (actual_h - exp_h)
        ratings[h] = elo_h + delta
        ratings[a] = elo_a - delta

    records_c = pd.DataFrame({
        "game_id":  game_ids_c,
        "elo_home": pre_elo_home,
        "elo_away": pre_elo_away,
    })

    # Upcoming games get current ratings
    upcoming = sched[sched["home_score"].isna()][["game_id", "home_team", "away_team"]].copy()
    if not upcoming.empty:
        upcoming["elo_home"] = upcoming["home_team"].map(lambda t: ratings.get(t, initial))
        upcoming["elo_away"] = upcoming["away_team"].map(lambda t: ratings.get(t, initial))
        records_u = upcoming[["game_id", "elo_home", "elo_away"]]
        all_rec = pd.concat([records_c, records_u], ignore_index=True)
    else:
        all_rec = records_c

    if all_rec.empty:
        return pd.DataFrame(columns=["game_id", "elo_home", "elo_away", "elo_diff"])

    df = all_rec.drop_duplicates(subset=["game_id"], keep="last").reset_index(drop=True)
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    return df


def make_training_set_bball(feats_df: pd.DataFrame) -> pd.DataFrame:
    """Add target columns (y_home_win, y_margin, y_total) from scores."""
    df = feats_df.copy()
    for c in ["home_score", "away_score"]:
        if c not in df.columns:
            df[c] = np.nan
    hs  = pd.to_numeric(df["home_score"], errors="coerce")
    as_ = pd.to_numeric(df["away_score"], errors="coerce")
    if hs.notna().any() and as_.notna().any():
        df["y_home_win"] = (hs > as_).astype(int)
        df["y_margin"]   = hs - as_
        df["y_total"]    = hs + as_
    else:
        df["y_home_win"] = np.nan
        df["y_margin"]   = np.nan
        df["y_total"]    = np.nan
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Fallback "bias" models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _BiasClassifier:
    p_base: float = 0.5
    used_features: List[str] = field(default_factory=list)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        return np.vstack([1.0 - self.p_base * np.ones(n), self.p_base * np.ones(n)]).T


@dataclass
class _BiasRegressor:
    mu: float = 0.0
    used_features: List[str] = field(default_factory=list)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mu, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Model fitting (GBM — same architecture as NFL tool)
# ─────────────────────────────────────────────────────────────────────────────

def _build_pipe(estimator):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("est", estimator)])


def _prepare_xy(
    df: pd.DataFrame, target_col: str, feature_cols: List[str]
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    have = [c for c in feature_cols if c in df.columns]
    X = df[have].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target_col], errors="coerce")
    mask = ~y.isna()
    X, y = X.loc[mask].copy(), y.loc[mask].values
    non_nan = [c for c in X.columns if not X[c].isna().all()]
    return X[non_nan], y, non_nan


def _make_classifier(random_state: int = 42):
    """Return XGBoost classifier if available, else sklearn GBM."""
    try:
        from xgboost import XGBClassifier  # type: ignore
        # Instantiating triggers native lib load — catches macOS libomp errors
        clf = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=random_state, n_jobs=-1,
        )
        return clf
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.04,
            subsample=0.8, min_samples_leaf=5, random_state=random_state,
        )


def _make_regressor(random_state: int = 42):
    """Return XGBoost regressor if available, else sklearn GBM."""
    try:
        from xgboost import XGBRegressor  # type: ignore
        reg = XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=random_state, n_jobs=-1,
        )
        return reg
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.04,
            subsample=0.8, min_samples_leaf=5, random_state=random_state,
        )


def fit_logistic_bball(train_df: pd.DataFrame) -> Tuple[object, float]:
    X, y, feats = _prepare_xy(train_df, "y_home_win", BBALL_FEATURE_COLS)
    if X.empty or len(y) == 0:
        return _BiasClassifier(p_base=0.5, used_features=["__bias__"]), 0.5

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import cross_val_score

    clf = _make_classifier()
    pipe = _build_pipe(clf)
    n_cv = min(5, len(y) // 15) if len(y) >= 30 else 0
    cv_acc = 0.5
    if n_cv >= 2:
        cv_acc = float(cross_val_score(pipe, X, y, cv=n_cv, scoring="accuracy").mean())

    # Wrap with Platt-scaling calibration so predicted probabilities are
    # properly calibrated (reduces XGBoost overconfidence, critical for edge calc)
    calibrated = CalibratedClassifierCV(pipe, method="sigmoid", cv=max(3, n_cv) if n_cv >= 2 else 3)
    try:
        calibrated.fit(X, y)
        calibrated.used_features = feats  # type: ignore[attr-defined]
        return calibrated, cv_acc
    except Exception:
        # Fallback: use uncalibrated pipe if calibration fails (e.g. too few samples)
        pipe.fit(X, y)
        pipe.used_features = feats  # type: ignore[attr-defined]
        return pipe, cv_acc


def _fit_gbm_bball(
    train_df: pd.DataFrame, target_col: str, default_mu: float, default_sigma: float
) -> Tuple[object, float, float]:
    X, y, feats = _prepare_xy(train_df, target_col, BBALL_FEATURE_COLS)
    if X.empty or len(y) == 0:
        return _BiasRegressor(mu=default_mu, used_features=["__bias__"]), default_sigma, default_sigma

    from sklearn.model_selection import cross_val_predict

    reg = _make_regressor()
    pipe = _build_pipe(reg)
    n_cv = min(5, len(y) // 15) if len(y) >= 30 else 0
    if n_cv >= 2:
        yhat_cv = cross_val_predict(pipe, X, y, cv=n_cv)
        resid = y - yhat_cv
        sigma  = float(np.std(resid, ddof=1))
        cv_mae = float(np.abs(resid).mean())
    else:
        pipe.fit(X, y)
        resid  = y - pipe.predict(X)
        sigma  = float(np.std(resid, ddof=1)) if len(resid) > 1 else default_sigma
        cv_mae = float(np.abs(resid).mean())

    pipe.fit(X, y)
    pipe.used_features = feats  # type: ignore[attr-defined]
    return pipe, sigma, cv_mae


def fit_margin_model_bball(
    train_df: pd.DataFrame, default_mu: float = 0.0, default_sigma: float = 12.0
) -> Tuple[object, float, float]:
    return _fit_gbm_bball(train_df, "y_margin", default_mu, default_sigma)


def fit_total_model_bball(
    train_df: pd.DataFrame, default_mu: float = 220.0, default_sigma: float = 16.0
) -> Tuple[object, float, float]:
    return _fit_gbm_bball(train_df, "y_total", default_mu, default_sigma)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────

def model_used_features(model) -> List[str]:
    feats = getattr(model, "used_features", None) or getattr(model, "feature_names_in_", None)
    return list(feats) if feats is not None else []


def _align_features(model, X: pd.DataFrame) -> pd.DataFrame:
    feats = model_used_features(model)
    if not feats:
        return X.apply(pd.to_numeric, errors="coerce")
    return X.reindex(columns=feats, fill_value=np.nan).apply(pd.to_numeric, errors="coerce")


def predict_home_prob(model, X: pd.DataFrame) -> np.ndarray:
    Xa = _align_features(model, X)
    if hasattr(model, "predict_proba"):
        return np.clip(model.predict_proba(Xa)[:, 1].astype(float), 1e-6, 1 - 1e-6)
    return np.full(len(Xa), getattr(model, "p_base", 0.5), dtype=float)


def prob_home_covers(line: float, mu_margin: float, sigma: float) -> float:
    if sigma <= 0 or pd.isna(mu_margin) or pd.isna(line):
        return np.nan
    return norm_cdf((mu_margin + line) / sigma)


def prob_away_covers(line: float, mu_margin: float, sigma: float) -> float:
    if sigma <= 0 or pd.isna(mu_margin) or pd.isna(line):
        return np.nan
    return norm_cdf(-(mu_margin + line) / sigma)


def prob_over(total: float, mu_total: float, sigma: float) -> float:
    if sigma <= 0 or pd.isna(mu_total) or pd.isna(total):
        return np.nan
    return 1.0 - norm_cdf((total - mu_total) / sigma)


def prob_under(total: float, mu_total: float, sigma: float) -> float:
    if sigma <= 0 or pd.isna(mu_total) or pd.isna(total):
        return np.nan
    return norm_cdf((total - mu_total) / sigma)


# ─────────────────────────────────────────────────────────────────────────────
# Odds fetching from The Odds API
# ─────────────────────────────────────────────────────────────────────────────

_ODDS_SPORT_KEYS = {"nba": "basketball_nba", "ncaab": "basketball_ncaab"}


def fetch_odds_basketball(
    api_key: str,
    sport: str,
    market: str = "h2h",
    regions: str = "us",
    odds_format: str = "american",
    bookmakers: Optional[str] = None,
) -> pd.DataFrame:
    _ssl_bootstrap()
    sport_key = _ODDS_SPORT_KEYS.get(sport.lower(), sport)
    params: Dict[str, str] = {
        "apiKey": api_key, "markets": market, "regions": regions, "oddsFormat": odds_format,
    }
    if bookmakers:
        params["bookmakers"] = bookmakers

    r = requests.get(
        f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/",
        params=params, timeout=30,
    )
    r.raise_for_status()

    canon_fn = canonical_nba_team if sport.lower() == "nba" else canonical_ncaab_team
    rows = []
    for ev in r.json():
        home = canon_fn(ev.get("home_team", ""))
        away = canon_fn(ev.get("away_team", ""))
        for bm in ev.get("bookmakers", []):
            for m in bm.get("markets", []):
                for out in m.get("outcomes", []):
                    rows.append({
                        "odds_event_id":  ev.get("id"),
                        "commence_dt":    ev.get("commence_time"),
                        "home_team":      home,
                        "away_team":      away,
                        "book_key":       bm.get("key"),
                        "book_title":     bm.get("title"),
                        "market_key":     m.get("key"),
                        "outcome_name":   str(out.get("name", "")),
                        "price_american": out.get("price"),
                        "line":           out.get("point"),
                    })
    return pd.DataFrame(rows)


def prep_odds_bball(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    """Devig and classify odds rows (is_home_outcome, is_over_outcome)."""
    if df.empty:
        return df
    canon_fn = canonical_nba_team if sport.lower() == "nba" else canonical_ncaab_team
    out = df.copy()
    out["commence_dt"] = pd.to_datetime(out["commence_dt"], errors="coerce", utc=True)
    # Canonicalize team names so they match the schedule (abbreviations for NBA, upper for NCAAB)
    out["home_team"] = out["home_team"].apply(canon_fn)
    out["away_team"] = out["away_team"].apply(canon_fn)

    def _is_home(row) -> bool:
        n = str(row["outcome_name"]).strip().lower()
        if n in ("home", "away", "over", "under"):
            return n == "home"
        return canon_fn(row["outcome_name"]) == row["home_team"]

    out["is_home_outcome"] = out.apply(_is_home, axis=1)
    out["is_over_outcome"] = out["outcome_name"].str.strip().str.lower().eq("over")
    out["odds_decimal"] = out["price_american"].map(
        lambda x: american_to_decimal(x) if pd.notna(x) else np.nan
    )
    out["book_p_raw"] = out["price_american"].map(
        lambda x: implied_prob_from_american(x) if pd.notna(x) else np.nan
    )
    grp_keys = ["odds_event_id", "book_key", "market_key"]
    out["vig_sum"] = out.groupby(grp_keys)["book_p_raw"].transform("sum")
    out["book_p_devig"] = (out["book_p_raw"] / out["vig_sum"]).clip(lower=1e-6, upper=1 - 1e-6)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Schedule ↔ Odds merging
# ─────────────────────────────────────────────────────────────────────────────

def _pair_key(home: str, away: str) -> str:
    return "|".join(sorted([str(home), str(away)]))


def merge_odds_with_schedule_bball(odds: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """
    Match odds rows to schedule rows by (team pair, date proximity).
    Falls back to pair-only if no time-window match found.
    """
    if odds.empty or sched.empty:
        return pd.DataFrame()

    o = odds.copy()
    s = sched.copy()
    o["pair_key"] = o.apply(lambda r: _pair_key(r["home_team"], r["away_team"]), axis=1)
    s["pair_key"] = s.apply(lambda r: _pair_key(r["home_team"], r["away_team"]), axis=1)
    o["commence_dt"] = pd.to_datetime(o["commence_dt"], errors="coerce", utc=True)
    s["game_dt"]     = pd.to_datetime(s["game_dt"],     errors="coerce", utc=True)

    # Time-window merge (match within ±3 days)
    oo = o.reset_index(drop=False).rename(columns={"index": "oid"})
    cand = oo.merge(
        s[["pair_key", "game_id", "season", "home_team", "away_team", "game_dt"]],
        on="pair_key", how="left",
    )
    cand["dt_diff_h"] = np.where(
        cand["commence_dt"].notna() & cand["game_dt"].notna(),
        (cand["commence_dt"] - cand["game_dt"]).dt.total_seconds().abs() / 3600.0,
        np.inf,
    )
    best = cand.loc[cand.groupby("oid")["dt_diff_h"].idxmin()].copy()
    matched = best[best["dt_diff_h"] <= 72.0].copy()  # 3 days

    if matched.empty:
        # Pair-only fallback
        s_uniq = s.drop_duplicates(subset=["pair_key"], keep="last")
        matched = o.merge(
            s_uniq[["pair_key", "game_id", "season", "home_team", "away_team"]],
            on="pair_key", how="inner",
        )
        return matched.reset_index(drop=True)

    if matched.empty:
        # Last resort: create pseudo-schedule entries from odds data
        pseudo = o.drop_duplicates(subset=["odds_event_id"]).copy()
        pseudo["game_id"] = pseudo["odds_event_id"]
        pseudo["season"]  = datetime.date.today().year
        return pseudo.reset_index(drop=True)

    # Resolve home_team_x/home_team_y → home_team (prefer schedule version _y)
    for side in ["home_team", "away_team"]:
        if f"{side}_y" in matched.columns:
            matched[side] = matched[f"{side}_y"]
        elif f"{side}_x" in matched.columns:
            matched[side] = matched[f"{side}_x"]

    keep = [
        "odds_event_id", "commence_dt", "game_dt", "home_team", "away_team", "book_key", "book_title",
        "market_key", "outcome_name", "price_american", "line", "odds_decimal",
        "book_p_raw", "vig_sum", "book_p_devig", "season", "game_id",
        "is_home_outcome", "is_over_outcome",
    ]
    return matched[[c for c in keep if c in matched.columns]].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Picks assembly
# ─────────────────────────────────────────────────────────────────────────────

def compose_picks_bball(
    g: pd.DataFrame,
    market_friendly: str,
    bankroll: float = 10000.0,
    kelly_fraction_use: float = 0.5,
) -> pd.DataFrame:
    if g.empty:
        return g
    df = g.copy()

    if market_friendly == "totals":
        cond = df.get("is_over_outcome", df["outcome_name"].str.lower().eq("over"))
        df["side"] = np.where(cond, "OVER", "UNDER")
    else:
        cond = df.get("is_home_outcome", df["outcome_name"].str.lower().eq("home"))
        df["side"] = np.where(cond, "HOME", "AWAY")

    df["p_true"]      = pd.to_numeric(df["p_true"], errors="coerce")
    df["odds_decimal"] = pd.to_numeric(df["odds_decimal"], errors="coerce")
    df["ev_per_$1"]   = df["p_true"] * (df["odds_decimal"] - 1.0) - (1.0 - df["p_true"])
    df["kelly_frac"]  = (
        df.apply(
            lambda r: _kelly_fraction(r["p_true"], r["odds_decimal"])
            if pd.notna(r["p_true"]) and pd.notna(r["odds_decimal"]) else 0.0,
            axis=1,
        ).clip(lower=0.0) * float(kelly_fraction_use)
    )
    df["stake_usd"]  = (df["kelly_frac"] * float(bankroll)).round(2)
    df["market_key"] = market_friendly
    if "book_p_devig" in df.columns:
        df["edge_vs_book_devig"] = df["p_true"] - df["book_p_devig"]

    keep = [
        "game_dt", "home_team", "away_team", "book_title", "market_key", "side",
        "line", "price_american", "odds_decimal", "book_p_devig", "p_true",
        "ev_per_$1", "kelly_frac", "stake_usd", "edge_vs_book_devig",
    ]
    return (
        df[[c for c in keep if c in df.columns]]
        .copy()
        .rename(columns={"price_american": "odds_american", "ev_per_$1": "ev_per_usd"})
    )


def assemble_picks_bball(
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
    mk_map = {"moneyline": "h2h", "spreads": "spreads", "totals": "totals"}
    picks_all = []

    for mk in markets:
        sub = merged[merged["market_key"].eq(mk_map[mk])].copy()
        if sub.empty:
            continue
        sub = sub.merge(p_home_df, on="game_id", how="left")

        cond_home = sub.get("is_home_outcome", sub["outcome_name"].str.lower().eq("home"))

        if mk == "moneyline":
            sub["p_true"] = np.where(cond_home, sub["p_home_model"], 1.0 - sub["p_home_model"])

        elif mk == "spreads":
            sub["mu_margin"] = sub["game_id"].map(mu_margin_map)
            sub["p_true"] = np.where(
                cond_home,
                sub.apply(lambda r: prob_home_covers(r["line"], r["mu_margin"], s_margin), axis=1),
                sub.apply(lambda r: prob_away_covers(r["line"], r["mu_margin"], s_margin), axis=1),
            )

        else:  # totals
            sub["mu_total"] = sub["game_id"].map(mu_total_map)
            cond_over = sub.get("is_over_outcome", sub["outcome_name"].str.lower().eq("over"))
            sub["p_true"] = np.where(
                cond_over,
                sub.apply(lambda r: prob_over(r["line"], r["mu_total"], s_total), axis=1),
                sub.apply(lambda r: prob_under(r["line"], r["mu_total"], s_total), axis=1),
            )

        # Keep only the best-priced book for each game/side before sizing
        if "odds_decimal" in sub.columns and "game_id" in sub.columns:
            sub = (
                sub.sort_values("odds_decimal", ascending=False)
                   .drop_duplicates(subset=["game_id", "outcome_name"], keep="first")
            )

        picks_mkt = compose_picks_bball(sub, mk, bankroll=bankroll, kelly_fraction_use=kelly_fraction_use)
        if not picks_mkt.empty:
            picks_all.append(picks_mkt)

    if not picks_all:
        return pd.DataFrame()
    return pd.concat(picks_all, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sport-specific defaults
# ─────────────────────────────────────────────────────────────────────────────

SPORT_DEFAULTS = {
    "nba": {
        "pyth_exp":             14.0,
        "default_mu_total":    220.0,
        "default_sigma_margin": 12.0,
        "default_sigma_total":  16.0,
    },
    "ncaab": {
        "pyth_exp":             11.5,
        "default_mu_total":    140.0,
        "default_sigma_margin": 10.0,
        "default_sigma_total":  14.0,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_picks_basketball(
    sport: str,
    season: int,
    markets: Optional[List[str]] = None,
    books: str = "fanduel,draftkings,betmgm",
    bankroll: float = 10000.0,
    kelly_fraction: float = 0.5,
    top_n: int = 20,
    save_dir: str = "./data",
    cache_dir: str = "./data/bball_cache",
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    api_key = api_key or os.getenv("THE_ODDS_API_KEY")
    if not api_key:
        raise ValueError("No API key provided (set THE_ODDS_API_KEY or pass --api_key)")

    sport   = sport.lower()
    markets = markets or ["moneyline", "spreads", "totals"]
    defs    = SPORT_DEFAULTS.get(sport, SPORT_DEFAULTS["nba"])

    train_seasons = [season - 2, season - 1]
    all_seasons   = sorted(set(train_seasons + [season]))

    print(f"[{sport.upper()}] Loading schedule for seasons {all_seasons} ...")
    sched_all = load_schedule(sport, all_seasons, cache_dir=cache_dir, include_upcoming=True)
    if sched_all.empty:
        raise RuntimeError(f"No schedule data loaded for {sport.upper()} seasons {all_seasons}")

    print("Building rolling team stats ...")
    team_stats = build_team_rolling_stats(sched_all)
    feats_all  = build_bball_game_features(sched_all, team_stats, pyth_exp=defs["pyth_exp"])

    train_df = make_training_set_bball(feats_all[feats_all["season"].isin(train_seasons)].copy())
    print(f"Training on {len(train_df)} game rows ...")

    model_ml,   cv_acc          = fit_logistic_bball(train_df)
    margin_mod, s_margin, cv_mm = fit_margin_model_bball(
        train_df, default_sigma=defs["default_sigma_margin"]
    )
    total_mod,  s_total,  cv_tm = fit_total_model_bball(
        train_df,
        default_mu=defs["default_mu_total"],
        default_sigma=defs["default_sigma_total"],
    )

    # Fetch live odds
    api_map = {"moneyline": "h2h", "spreads": "spreads", "totals": "totals"}
    frames = []
    for mk in markets:
        try:
            od = fetch_odds_basketball(api_key, sport=sport, market=api_map[mk], bookmakers=books or None)
            if isinstance(od, pd.DataFrame) and not od.empty:
                frames.append(od)
        except Exception as e:
            print(f"[{sport.upper()}] Warning fetching {mk} odds: {e}")

    if not frames:
        raise RuntimeError("No odds returned from The Odds API.")

    odds_live = prep_odds_bball(pd.concat(frames, ignore_index=True), sport=sport)
    sched_target = sched_all[sched_all["season"] == season].copy()
    merged = merge_odds_with_schedule_bball(odds_live, sched_target)
    if merged.empty:
        raise RuntimeError("No market rows matched the schedule.")

    # Attach features to merged rows
    game_ids = merged["game_id"].unique()
    feats_target = feats_all[feats_all["game_id"].isin(game_ids)].copy()
    present = [c for c in BBALL_FEATURE_COLS if c in feats_target.columns]
    if "game_id" in feats_target.columns and present:
        merged = merged.merge(
            feats_target[["game_id"] + present].drop_duplicates("game_id"),
            on="game_id", how="left",
        )

    game_feats = merged.drop_duplicates(subset=["game_id"]).copy().reset_index(drop=True)
    for c in BBALL_FEATURE_COLS:
        if c not in game_feats.columns:
            game_feats[c] = np.nan

    X_full    = game_feats[BBALL_FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    p_home    = predict_home_prob(model_ml,  _align_features(model_ml,  X_full))
    mu_margin = margin_mod.predict(_align_features(margin_mod, X_full))
    mu_total  = total_mod.predict( _align_features(total_mod,  X_full))

    p_home_df     = pd.DataFrame({"game_id": game_feats["game_id"].values, "p_home_model": p_home})
    mu_margin_map = dict(zip(game_feats["game_id"], mu_margin))
    mu_total_map  = dict(zip(game_feats["game_id"], mu_total))

    df_out = assemble_picks_bball(
        merged, markets, p_home_df, mu_margin_map, mu_total_map,
        s_margin, s_total, bankroll, kelly_fraction,
    )
    if df_out.empty:
        raise RuntimeError("No picks produced.")

    df_out = df_out.sort_values(["ev_per_usd", "edge_vs_book_devig"], ascending=False).head(top_n)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"picks_{sport}_season{season}.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\nTop {top_n} picks ({sport.upper()}, Season {season}):")
    print(df_out.to_string(index=False))
    print(f"\nSaved → {out_path}")
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward backtest
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest_basketball(
    sched_all: pd.DataFrame,
    eval_seasons: List[int],
    sport: str = "nba",
    advanced_stats: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    defs        = SPORT_DEFAULTS.get(sport.lower(), SPORT_DEFAULTS["nba"])
    all_seasons = sorted({int(s) for s in sched_all["season"].dropna().unique()})
    eval_seasons = sorted({int(s) for s in eval_seasons})

    team_stats = build_team_rolling_stats(sched_all)
    # Use provided advanced stats; only fetch internally if not supplied
    if advanced_stats is None and sport.lower() == "nba":
        try:
            advanced_stats = load_nba_advanced_stats(all_seasons)
        except Exception:
            advanced_stats = pd.DataFrame()
    adv_stats = advanced_stats if advanced_stats is not None else pd.DataFrame()
    feats_all  = build_bball_game_features(
        sched_all, team_stats, pyth_exp=defs["pyth_exp"], advanced_stats=adv_stats
    )
    results: List[pd.DataFrame] = []

    for target_season in eval_seasons:
        prior = [s for s in all_seasons if s < target_season]
        if not prior:
            continue

        train_df = make_training_set_bball(feats_all[feats_all["season"].isin(prior)].copy())
        if train_df["y_home_win"].notna().sum() < 20:
            continue

        model_ml,   _         = fit_logistic_bball(train_df)
        margin_mod, s_margin, _ = fit_margin_model_bball(
            train_df, default_sigma=defs["default_sigma_margin"]
        )
        total_mod,  s_total,  _ = fit_total_model_bball(
            train_df,
            default_mu=defs["default_mu_total"],
            default_sigma=defs["default_sigma_total"],
        )

        eval_df   = make_training_set_bball(feats_all[feats_all["season"] == target_season].copy())
        completed = eval_df[eval_df["y_home_win"].notna()].reset_index(drop=True)
        if completed.empty:
            continue

        train_feats = make_training_set_bball(feats_all[feats_all["season"].isin(prior)].copy())
        for c in BBALL_FEATURE_COLS:
            if c not in completed.columns:
                completed[c] = np.nan

        X = completed[BBALL_FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
        feature_means = train_feats[BBALL_FEATURE_COLS].apply(pd.to_numeric, errors="coerce").mean()
        X = X.fillna(feature_means)
        p_home    = predict_home_prob(model_ml,  _align_features(model_ml,  X))
        mu_margin = margin_mod.predict(_align_features(margin_mod, X))
        mu_total  = total_mod.predict( _align_features(total_mod,  X))

        rec = pd.DataFrame({
            "season":          completed["season"].values,
            "game_id":         completed["game_id"].values,
            "home_team":       completed.get("home_team",  pd.Series(np.nan, index=completed.index)).values,
            "away_team":       completed.get("away_team",  pd.Series(np.nan, index=completed.index)).values,
            "p_home_pred":     p_home,
            "actual_home_win": completed["y_home_win"].values,
            "pred_margin":     mu_margin,
            "actual_margin":   completed["y_margin"].values,
            "pred_total":      mu_total,
            "actual_total":    completed["y_total"].values,
        })
        results.append(rec)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)
