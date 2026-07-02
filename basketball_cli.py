#!/usr/bin/env python3
"""
Basketball picks CLI — NBA & NCAAB.
Mirrors cli.py for the NFL tool.

Examples
--------
  # NBA picks for the current (2024-25) season:
  python basketball_cli.py --sport nba --season 2025

  # NCAAB picks, FanDuel only, half-Kelly:
  python basketball_cli.py --sport ncaab --season 2025 --books fanduel --kelly_fraction 0.5

  # Moneyline and totals only, top 10:
  python basketball_cli.py --sport nba --season 2025 --markets moneyline,totals --top_n 10
"""

import os
import sys
import argparse

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from fetch_basketball_data import run_picks_basketball


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate basketball picks (NBA or NCAAB) using The Odds API."
    )
    p.add_argument(
        "--sport",
        required=True,
        choices=["nba", "ncaab"],
        help="Sport: 'nba' or 'ncaab'",
    )
    p.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season end-year (e.g. 2025 = 2024-25 season)",
    )
    p.add_argument(
        "--markets",
        default="moneyline,spreads,totals",
        help="Comma-separated markets (default: moneyline,spreads,totals)",
    )
    p.add_argument(
        "--books",
        default="fanduel,draftkings,betmgm",
        help="Comma-separated bookmaker keys",
    )
    p.add_argument("--bankroll",       type=float, default=10000.0)
    p.add_argument("--kelly_fraction", type=float, default=0.5)
    p.add_argument("--top_n",          type=int,   default=20)
    p.add_argument("--save_dir",       default="./data")
    p.add_argument(
        "--cache_dir",
        default="./data/bball_cache",
        help="Directory for caching ESPN schedule data (NCAAB especially)",
    )
    p.add_argument("--api_key", default=None, help="The Odds API key (overrides env var)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_picks_basketball(
        sport=args.sport,
        season=args.season,
        markets=[m.strip().lower() for m in args.markets.split(",") if m.strip()],
        books=args.books,
        bankroll=args.bankroll,
        kelly_fraction=args.kelly_fraction,
        top_n=args.top_n,
        save_dir=args.save_dir,
        cache_dir=args.cache_dir,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
