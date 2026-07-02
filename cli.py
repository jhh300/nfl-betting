#!/usr/bin/env python3
"""NFL picks CLI.

  python cli.py picks --season 2025           # pull odds, train, print top picks
  python cli.py track --seasons 2024,2025     # score model vs Vegas closing lines
                                              # (no odds-API key needed)

`python cli.py --season 2025` (no subcommand) still works and means `picks`.
"""

import os, sys, argparse

try:
    import nfl_data_py  # noqa: F401
except Exception as e:
    print(f"\n[ImportError] Could not import 'nfl_data_py'.\nPython: {sys.executable}\nError: {e!r}", file=sys.stderr)
    print("\nFix:\n  python3 -m venv .venv && source .venv/bin/activate\n  pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from fetch_nfl_data import run_picks, run_market_eval, format_market_summary

def _parse_args():
    p = argparse.ArgumentParser(description="NFL picks and model-vs-Vegas tracking")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("picks", help="Pull odds, train models, print top picks")
    s.add_argument("--season",         type=int,   required=True)
    s.add_argument("--markets",        default="moneyline,spreads",
                   help="Comma-separated; totals excluded by default (backtests show it loses)")
    s.add_argument("--books",          default="fanduel,draftkings,betmgm")
    s.add_argument("--bankroll",       type=float, default=10000.0)
    s.add_argument("--kelly_fraction", type=float, default=0.5)
    s.add_argument("--top_n",          type=int,   default=20)
    s.add_argument("--min_edge_pts",   type=float, default=2.0,
                   help="Skip spread/total picks where model and book differ by less than this")
    s.add_argument("--save_dir",       default="./data")
    s.add_argument("--api_key",        default=None)

    t = sub.add_parser("track", help="Backtest and score the model against Vegas closing lines")
    t.add_argument("--seasons", required=True, help="Comma-separated seasons to evaluate, e.g. 2023,2024")
    t.add_argument("--save_csv", default=None, help="Optional path to save the per-game comparison CSV")

    argv = sys.argv[1:]
    if argv and argv[0] not in ("picks", "track", "-h", "--help"):
        argv = ["picks"] + argv
    return p.parse_args(argv)

def main():
    args = _parse_args()
    if args.cmd == "track":
        seasons = [int(s) for s in str(args.seasons).split(",") if s.strip()]
        per_game, summary = run_market_eval(seasons)
        print(format_market_summary(summary))
        if args.save_csv:
            per_game.to_csv(args.save_csv, index=False)
            print(f"\nPer-game comparison saved to {args.save_csv}")
        return
    run_picks(
        season=args.season,
        markets=[m.strip().lower() for m in args.markets.split(",") if m.strip()],
        books=args.books,
        bankroll=args.bankroll,
        kelly_fraction=args.kelly_fraction,
        top_n=args.top_n,
        min_edge_pts=args.min_edge_pts,
        save_dir=args.save_dir,
        api_key=args.api_key,
    )

if __name__ == "__main__":
    main()
