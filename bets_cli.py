#!/usr/bin/env python3
"""
Manage bets in SQLite: add, update-clv, settle, list-games.

Examples:
  python bets_cli.py --db ./data/bets.db add --game_id <GAME_ID> --market spreads \
    --side HOME --entry_line -2.5 --book_key draftkings --price -110 --stake 150
  python bets_cli.py --db ./data/bets.db update-clv --all
  python bets_cli.py --db ./data/bets.db settle --season 2025
  python bets_cli.py --db ./data/bets.db list-games
"""

import os, sys, argparse, sqlite3, datetime as dt
from typing import Optional

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

import pandas as pd
import nfl_data_py as nfl

from fetch_nfl_data import american_to_decimal, fetch_closing_from_history

def _now_utc() -> str:
    try:
        from datetime import UTC
        return dt.datetime.now(UTC).isoformat(timespec="seconds")
    except Exception:
        return dt.datetime.utcnow().isoformat(timespec="seconds")

def _conn(db: str):
    if os.path.dirname(db): os.makedirs(os.path.dirname(db), exist_ok=True)
    return sqlite3.connect(db)

def _ensure_schema(cur):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bets(
          bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_utc TEXT, game_id TEXT, season INTEGER, week INTEGER,
          odds_event_id TEXT, book_key TEXT, market_key TEXT, side TEXT,
          entry_line REAL, entry_odds_american REAL, entry_odds_decimal REAL,
          stake_usd REAL, notes TEXT, status TEXT DEFAULT 'open',
          closing_odds_american REAL, closing_odds_decimal REAL,
          closing_line REAL, clv_prob_diff REAL, clv_return_pct REAL, clv_points REAL,
          result TEXT, payout_usd REAL, settled_utc TEXT
        )
    """)
    for col, dtype in [
        ("closing_odds_american","REAL"),("closing_odds_decimal","REAL"),("closing_line","REAL"),
        ("clv_prob_diff","REAL"),("clv_return_pct","REAL"),("clv_points","REAL"),
        ("result","TEXT"),("payout_usd","REAL"),("settled_utc","TEXT"),
    ]:
        try: cur.execute(f"ALTER TABLE bets ADD COLUMN {col} {dtype}")
        except Exception: pass

def cmd_list_games(db: str):
    con = _conn(db)
    try:
        df = pd.read_sql_query("SELECT * FROM games ORDER BY season, week", con)
        print("No games in DB." if df.empty else df.to_string(index=False))
    finally:
        con.close()

def cmd_add(db: str, game_id: str, market: str, side: str, entry_line: Optional[float],
            book_key: str, price: float, stake: float, notes: str):
    con = _conn(db); cur = con.cursor()
    try:
        _ensure_schema(cur)
        row = cur.execute("SELECT season, week FROM games WHERE game_id=?", (game_id,)).fetchone()
        season, week = (row[0], row[1]) if row else (None, None)
        cur.execute("""
            INSERT INTO bets(created_utc,game_id,season,week,odds_event_id,book_key,market_key,side,
                             entry_line,entry_odds_american,entry_odds_decimal,stake_usd,notes,status)
            VALUES (?,?,?,?,NULL,?,?,?,?,?,?,?,?,'open')
        """, (_now_utc(), game_id, season, week, book_key, market.lower(), side.upper(),
              entry_line, float(price), float(american_to_decimal(price)), float(stake), notes))
        con.commit(); print("Bet added.")
    finally:
        con.close()

def cmd_update_clv(db: str, api_key: Optional[str], only_bet_id: Optional[int], update_all: bool):
    api_key = api_key or os.getenv("THE_ODDS_API_KEY")
    if not api_key:
        print("Missing THE_ODDS_API_KEY", file=sys.stderr); sys.exit(2)
    con = _conn(db); cur = con.cursor()
    try:
        _ensure_schema(cur)
        q = "SELECT bet_id,odds_event_id,book_key,market_key,side,entry_odds_decimal,entry_line FROM bets WHERE status='open'"
        params = ()
        if only_bet_id:
            q += " AND bet_id=?"; params = (only_bet_id,)
        rows = cur.execute(q, params).fetchall()
        if not rows: print("No open bets to update."); return
        updated = 0
        for bet_id, odds_event_id, book_key, market_key, side, entry_dec, entry_line in rows:
            mk = (market_key or "moneyline").lower()
            if mk not in {"moneyline","spreads","totals"}: mk = "moneyline"
            api_market = "h2h" if mk == "moneyline" else mk
            try:
                info = fetch_closing_from_history(api_key, str(odds_event_id), str(book_key), market_key=api_market)
                if mk == "moneyline":
                    closing_am  = info["home_price"] if side.upper()=="HOME" else info["away_price"]
                    closing_dec = american_to_decimal(closing_am) if closing_am is not None else None
                    clv_prob    = (1.0/closing_dec)-(1.0/entry_dec) if (closing_dec and entry_dec) else None
                    clv_ret     = (closing_dec/entry_dec-1.0)       if (closing_dec and entry_dec) else None
                    cur.execute("UPDATE bets SET closing_odds_american=?,closing_odds_decimal=?,clv_prob_diff=?,clv_return_pct=? WHERE bet_id=?",
                                (_f(closing_am),_f(closing_dec),_f(clv_prob),_f(clv_ret),int(bet_id)))
                elif mk == "spreads":
                    closing_am   = info["home_price"] if side.upper()=="HOME" else info["away_price"]
                    closing_line = info["home_line"]  if side.upper()=="HOME" else info["away_line"]
                    closing_dec  = american_to_decimal(closing_am) if closing_am is not None else None
                    clv_pts = (closing_line-entry_line) if (closing_line is not None and entry_line is not None) else None
                    cur.execute("UPDATE bets SET closing_odds_american=?,closing_odds_decimal=?,closing_line=?,clv_points=? WHERE bet_id=?",
                                (_f(closing_am),_f(closing_dec),_f(closing_line),_f(clv_pts),int(bet_id)))
                else:
                    closing_line = info["total_line"]
                    closing_am   = info["over_price"] if side.upper()=="OVER" else info["under_price"]
                    closing_dec  = american_to_decimal(closing_am) if closing_am is not None else None
                    mult    = -1.0 if side.upper()=="OVER" else 1.0
                    clv_pts = (closing_line-entry_line)*mult if (closing_line is not None and entry_line is not None) else None
                    cur.execute("UPDATE bets SET closing_odds_american=?,closing_odds_decimal=?,closing_line=?,clv_points=? WHERE bet_id=?",
                                (_f(closing_am),_f(closing_dec),_f(closing_line),_f(clv_pts),int(bet_id)))
                updated += 1
            except Exception as e:
                print(f"CLV update failed for bet_id {bet_id}: {e}", file=sys.stderr)
        con.commit(); print(f"Updated CLV for {updated} bet(s).")
    finally:
        con.close()

def _f(v): return float(v) if v is not None else None

def cmd_settle(db: str, season: int):
    con = _conn(db); cur = con.cursor()
    try:
        _ensure_schema(cur)
        sched = nfl.import_schedules([int(season)])[["game_id","home_score","away_score"]].dropna()
        wins  = {r["game_id"]: ("HOME" if r["home_score"]>r["away_score"] else "AWAY")
                 for _,r in sched.iterrows() if r["home_score"]!=r["away_score"]}
        rows  = cur.execute(
            "SELECT bet_id,game_id,side,stake_usd,entry_odds_decimal FROM bets WHERE status='open' AND season=?",
            (int(season),)).fetchall()
        settled = 0
        for bet_id, game_id, side, stake, entry_dec in rows:
            if game_id in wins:
                result = "win" if wins[game_id]==side else "loss"
                payout = stake*(entry_dec-1.0) if result=="win" else -stake
                cur.execute("UPDATE bets SET status='settled',result=?,payout_usd=?,settled_utc=? WHERE bet_id=?",
                            (result, float(payout), _now_utc(), int(bet_id)))
                settled += 1
        con.commit(); print(f"Settled {settled} bet(s) for season {season}.")
    finally:
        con.close()

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Manage bets in SQLite")
    p.add_argument("--db", default="./data/bets.db")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-games")

    s = sub.add_parser("add")
    s.add_argument("--game_id", required=True)
    s.add_argument("--market",  required=True, choices=["moneyline","spreads","totals"])
    s.add_argument("--side",    required=True)
    s.add_argument("--entry_line", type=float, default=None)
    s.add_argument("--book_key",   required=True)
    s.add_argument("--price",  type=float, required=True)
    s.add_argument("--stake",  type=float, required=True)
    s.add_argument("--notes",  default="")

    s = sub.add_parser("update-clv")
    s.add_argument("--api_key", default=None)
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--bet_id", type=int)
    g.add_argument("--all", action="store_true")

    s = sub.add_parser("settle")
    s.add_argument("--season", type=int, required=True)

    return p.parse_args()

def main():
    args = _parse_args()
    if   args.cmd == "list-games":  cmd_list_games(args.db)
    elif args.cmd == "add":         cmd_add(args.db, args.game_id, args.market, args.side, args.entry_line, args.book_key, args.price, args.stake, args.notes)
    elif args.cmd == "update-clv":  cmd_update_clv(args.db, args.api_key, getattr(args,"bet_id",None), getattr(args,"all",False))
    elif args.cmd == "settle":      cmd_settle(args.db, args.season)
    else: print("Unknown command", file=sys.stderr); sys.exit(2)

if __name__ == "__main__":
    main()
