#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd

try:
    import nfl_data_py as nfl
except Exception as e:
    raise SystemExit(
        "Missing dependency 'nfl-data-py'. Install with:\n"
        "  pip install --upgrade nfl-data-py pandas numpy pyarrow\n"
        f"Original error: {e}"
    )

OUT_DIR = "./data_uploads"

def _ensure_cols(df: pd.DataFrame, cols_and_defaults: list) -> pd.DataFrame:
    for c, default in cols_and_defaults:
        if c not in df.columns: df[c] = default
    return df

def _to_int_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def _group_sum(df: pd.DataFrame, by: list, cols: list) -> pd.DataFrame:
    return df.groupby(by, dropna=False)[cols].sum(min_count=1).reset_index()

def _qb_rating(attempts, completions, yards, touchdowns, interceptions) -> float:
    if attempts <= 0: return np.nan
    def clamp(v): return max(0.0, min(float(v), 2.375))
    a = clamp(((completions/attempts)-0.3)*5)
    b = clamp(((yards/attempts)-3)*0.25)
    c = clamp((touchdowns/attempts)*20)
    d = clamp(2.375-((interceptions/attempts)*25))
    return ((a+b+c+d)/6)*100

def load_pbp(years: List[int]) -> pd.DataFrame:
    pbp = nfl.import_pbp_data(years)
    needs = [
        ("season",np.nan),("week",np.nan),("game_id",""),("posteam",""),("defteam",""),
        ("pass_attempt",0),("complete_pass",0),("rush_attempt",0),
        ("dropback",0),("qb_hit",0),("sack",0),("interception",0),("fumble_lost",0),
        ("yards_gained",0.0),("epa",0.0),("yardline_100",np.nan),("drive",np.nan),
        ("pass_touchdown",0),("rush_touchdown",0),("play_type",""),
        ("passer_player_id",""),("passer_player_name",""),
    ]
    pbp = _ensure_cols(pbp, needs)
    pbp["season"] = pd.to_numeric(pbp["season"], errors="coerce")
    pbp["week"]   = pd.to_numeric(pbp["week"],   errors="coerce")
    pbp = pbp[pbp["season"].notna() & pbp["week"].notna()].copy()
    pbp["season"] = pbp["season"].astype(int)
    pbp["week"]   = pbp["week"].astype(int)
    return pbp

def build_turnovers(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp.copy()
    df["turnovers_off"] = (
        (pd.to_numeric(df["interception"], errors="coerce").fillna(0) > 0).astype(int) +
        (pd.to_numeric(df["fumble_lost"],  errors="coerce").fillna(0) > 0).astype(int)
    )
    df["int_def"]  = (pd.to_numeric(df["interception"], errors="coerce").fillna(0) > 0).astype(int)
    df["fuml_def"] = (pd.to_numeric(df["fumble_lost"],  errors="coerce").fillna(0) > 0).astype(int)
    off  = _group_sum(df, ["season","week","posteam"], ["turnovers_off"]).rename(columns={"posteam":"team"})
    take = _group_sum(df, ["season","week","defteam"],  ["int_def","fuml_def"])
    take["takeaways_def"] = take["int_def"].fillna(0)+take["fuml_def"].fillna(0)
    take = take.drop(columns=["int_def","fuml_def"]).rename(columns={"defteam":"team"})
    merged = off.merge(take, on=["season","week","team"], how="outer")
    merged["turnovers_off"]  = merged["turnovers_off"].fillna(0)
    merged["takeaways_def"]  = merged["takeaways_def"].fillna(0)
    merged["turnover_diff"]  = merged["takeaways_def"]-merged["turnovers_off"]
    return merged[["team","season","week","turnover_diff"]].sort_values(["season","week","team"])

def build_redzone(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp.copy()
    df["rz_play"]    = pd.to_numeric(df["yardline_100"], errors="coerce") <= 20
    df["off_td_flag"] = ((pd.to_numeric(df["rush_touchdown"], errors="coerce") > 0) |
                         (pd.to_numeric(df["pass_touchdown"], errors="coerce") > 0)).astype(int)
    g = df.groupby(["season","week","game_id","posteam","drive"], dropna=False).agg(
        rz_trip=("rz_play","max"), off_td=("off_td_flag","max")).reset_index()
    rz_off = g[g["rz_trip"]==True].groupby(["season","week","posteam"]).agg(  # noqa: E712
        rz_trips=("rz_trip","size"), rz_tds=("off_td","sum")).reset_index().rename(columns={"posteam":"team"})
    rz_off["rz_off_td_pct"] = np.where(rz_off["rz_trips"]>0, rz_off["rz_tds"]/rz_off["rz_trips"], np.nan)
    g_def = df.groupby(["season","week","game_id","defteam","drive"], dropna=False).agg(
        rz_trip=("rz_play","max"), off_td=("off_td_flag","max")).reset_index().rename(columns={"defteam":"team"})
    rz_def = g_def[g_def["rz_trip"]==True].groupby(["season","week","team"]).agg(
        rz_trips_allowed=("rz_trip","size"), rz_tds_allowed=("off_td","sum")).reset_index()
    rz_def["rz_def_td_pct"] = np.where(rz_def["rz_trips_allowed"]>0, rz_def["rz_tds_allowed"]/rz_def["rz_trips_allowed"], np.nan)
    out = rz_off.merge(rz_def[["season","week","team","rz_def_td_pct"]], on=["season","week","team"], how="outer")
    return out[["team","season","week","rz_off_td_pct","rz_def_td_pct"]].sort_values(["season","week","team"])

def build_qb(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp.copy()
    passes = df[pd.to_numeric(df["pass_attempt"], errors="coerce").fillna(0) > 0].copy()
    passes = _ensure_cols(passes, [("complete_pass",0),("yards_gained",0.0),("pass_touchdown",0),("interception",0)])
    grp = passes.groupby(["season","week","game_id","posteam","passer_player_id","passer_player_name"], dropna=False).agg(
        att=("pass_attempt","sum"), comp=("complete_pass","sum"), yards=("yards_gained","sum"),
        td=("pass_touchdown","sum"), ints=("interception","sum"), epa_sum=("epa","sum")).reset_index()
    idx = grp.groupby(["season","week","game_id","posteam"])["att"].idxmax()
    primary = grp.loc[idx].copy()
    primary["qb_rating"]   = primary.apply(lambda r: _qb_rating(r["att"],r["comp"],r["yards"],r["td"],r["ints"]), axis=1)
    primary["qb_ypa"]      = np.where(primary["att"]>0, primary["yards"]/primary["att"], np.nan)
    primary["qb_comp_pct"] = np.where(primary["att"]>0, primary["comp"]/primary["att"],  np.nan)
    primary["qb_epa_play"] = np.where(primary["att"]>0, primary["epa_sum"]/primary["att"], np.nan)
    out = primary.groupby(["season","week","posteam"], dropna=False).agg(
        qb_rating=("qb_rating","mean"), qb_ypa=("qb_ypa","mean"),
        qb_comp_pct=("qb_comp_pct","mean"), qb_epa_play=("qb_epa_play","mean")).reset_index().rename(columns={"posteam":"team"})
    return out[["team","season","week","qb_rating","qb_ypa","qb_comp_pct","qb_epa_play"]].sort_values(["season","week","team"])

def build_ol_dl(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp.copy()
    for c in ["dropback","sack","qb_hit","rush_attempt"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    off_pass = df[df["dropback"]>0].copy()
    off_grp  = off_pass.groupby(["season","week","posteam"], dropna=False).agg(
        dropbacks=("dropback","sum"), sacks=("sack","sum"), qbh=("qb_hit","sum")).reset_index().rename(columns={"posteam":"team"})
    off_grp["ol_pass_block"] = np.where(off_grp["dropbacks"]>0, 1.0-(off_grp["sacks"]+off_grp["qbh"])/off_grp["dropbacks"], np.nan)
    rush    = df[df["rush_attempt"]>0].copy()
    run_grp = rush.groupby(["season","week","posteam"], dropna=False).agg(
        rush_plays=("rush_attempt","sum"), rush_epa=("epa","sum")).reset_index().rename(columns={"posteam":"team"})
    run_grp["ol_run_block"] = np.where(run_grp["rush_plays"]>0, run_grp["rush_epa"]/run_grp["rush_plays"], np.nan)
    def_grp = df[df["dropback"]>0].groupby(["season","week","defteam"], dropna=False).agg(
        opp_dropbacks=("dropback","sum"), sacks=("sack","sum"), qbh=("qb_hit","sum")).reset_index().rename(columns={"defteam":"team"})
    def_grp["dl_pressure_rate"] = np.where(def_grp["opp_dropbacks"]>0, (def_grp["sacks"]+def_grp["qbh"])/def_grp["opp_dropbacks"], np.nan)
    out = (off_grp[["season","week","team","ol_pass_block"]]
           .merge(run_grp[["season","week","team","ol_run_block"]], on=["season","week","team"], how="outer")
           .merge(def_grp[["season","week","team","dl_pressure_rate"]], on=["season","week","team"], how="outer"))
    return out[["team","season","week","ol_pass_block","ol_run_block","dl_pressure_rate"]].sort_values(["season","week","team"])

def build_special_teams(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp.copy()
    if "special_teams_play" in df.columns:
        st_df = df[df["special_teams_play"]==1].copy()
    else:
        st_df = df[df["play_type"].astype(str).str.contains(r"kickoff|punt|field_goal|extra_point|return", case=False, regex=True)].copy()
    if st_df.empty: return pd.DataFrame(columns=["team","season","week","st_efficiency"])
    off  = st_df.groupby(["season","week","posteam"], dropna=False)["epa"].mean().reset_index().rename(columns={"posteam":"team","epa":"st_epa_off"})
    deff = st_df.groupby(["season","week","defteam"], dropna=False)["epa"].mean().reset_index().rename(columns={"defteam":"team","epa":"st_epa_def"})
    out  = off.merge(deff, on=["season","week","team"], how="outer")
    out["st_efficiency"] = out["st_epa_off"].fillna(0)-out["st_epa_def"].fillna(0)
    return out[["team","season","week","st_efficiency"]].sort_values(["season","week","team"])

def build_dvoa_template(pbp: pd.DataFrame) -> pd.DataFrame:
    teams = pbp[["season","week","posteam"]].dropna().drop_duplicates().rename(columns={"posteam":"team"})
    teams = teams.sort_values(["season","week","team"]).reset_index(drop=True)
    for c in ["off_dvoa","def_dvoa","st_efficiency"]:
        teams[c] = np.nan
    return teams[["team","season","week","off_dvoa","def_dvoa","st_efficiency"]]

def main(years: List[int]):
    os.makedirs(OUT_DIR, exist_ok=True)
    pbp = load_pbp(years)
    tasks = [
        ("qb.csv",            build_qb(pbp)),
        ("turnovers.csv",     build_turnovers(pbp)),
        ("redzone.csv",       build_redzone(pbp)),
        ("ol_dl.csv",         build_ol_dl(pbp)),
        ("special_teams.csv", build_special_teams(pbp)),
        ("dvoa_template.csv", build_dvoa_template(pbp)),
    ]
    for fname, df in tasks:
        df.to_csv(os.path.join(OUT_DIR, fname), index=False)
    print(f"Done. CSVs written to {OUT_DIR}/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, required=True, help="e.g. --seasons 2022 2023 2024")
    main(ap.parse_args().seasons)
