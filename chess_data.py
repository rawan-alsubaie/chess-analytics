"""
Data pipeline for the Lichess chess dashboard.

Reproduces the cleaning and feature engineering from the EDA notebook
(chess_eda_updated.ipynb) so the app works with either the raw `games.csv`
from Kaggle or the `clean_data.csv` the notebook exports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Category orders (shared by the app)
# ---------------------------------------------------------------------------
RESULT_ORDER = ["white", "black", "draw"]
STATUS_ORDER = ["mate", "resign", "outoftime", "draw"]
SKILL_ORDER = ["Beginner (<=1400)", "Intermediate (1400-1790)", "Advanced (>1790)"]
TIME_ORDER = ["Short (<10 min)", "Medium (10 min)", "Long (>10 min)"]
GAMETYPE_ORDER = ["Bullet", "Blitz", "Rapid", "Classical"]
DIFF_ORDER = [
    "Black much stronger", "Black stronger", "Even (Black slightly higher)",
    "Even (White slightly higher)", "White stronger", "White much stronger",
]
GAP_ORDER = ["1-50", "51-100", "101-200", "201-400", ">400"]

REQUIRED_COLS = {
    "id", "rated", "created_at", "last_move_at", "turns", "victory_status",
    "winner", "increment_code", "white_id", "white_rating", "black_id",
    "black_rating", "moves", "opening_eco", "opening_name", "opening_ply",
}


def _to_datetime(col: pd.Series) -> pd.Series:
    """Raw file stores epoch milliseconds; clean_data.csv stores date strings."""
    if pd.api.types.is_numeric_dtype(col):
        return pd.to_datetime(col, unit="ms", errors="coerce")
    return pd.to_datetime(col, errors="coerce")


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Run the notebook pipeline. Returns (clean df, cleaning log)."""
    missing = REQUIRED_COLS - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    log = [{"step": "Raw file", "rows": len(raw), "removed": 0,
            "note": "Lichess games as loaded"}]

    exact = int(raw.duplicated().sum())
    df = raw.drop_duplicates(subset="id", keep="first").copy()
    log.append({"step": "Remove repeated game ids", "rows": len(df),
                "removed": len(raw) - len(df),
                "note": f"{exact} exact copies + {len(raw) - len(df) - exact} same game with different timestamps"})

    df["created_at"] = _to_datetime(df["created_at"])
    df["last_move_at"] = _to_datetime(df["last_move_at"])
    df["duration_min"] = (df["last_move_at"] - df["created_at"]).dt.total_seconds() / 60

    before = len(df)
    df = df[df["turns"] >= 3].copy()
    log.append({"step": "Remove games with < 3 turns", "rows": len(df),
                "removed": before - len(df),
                "note": "Mostly instant resigns / timeouts, not real play"})

    if df["rated"].dtype != bool:
        df["rated"] = df["rated"].astype(str).str.lower().isin(["true", "1"])

    # Ratings
    df["rating_diff"] = df["white_rating"] - df["black_rating"]
    df["rating_diff_cat"] = pd.cut(
        df["rating_diff"], bins=[-np.inf, -200, -50, 0, 50, 200, np.inf], labels=DIFF_ORDER)
    df["white_win"] = (df["winner"] == "white").astype(int)
    df["black_win"] = (df["winner"] == "black").astype(int)
    df["is_draw"] = (df["winner"] == "draw").astype(int)
    # White's score from White's point of view (1 / 0.5 / 0)
    df["white_score"] = df["white_win"] + 0.5 * df["is_draw"]

    # Time control
    parts = df["increment_code"].astype(str).str.split("+", expand=True)
    df["base_min"] = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype(int)
    df["increment_sec"] = pd.to_numeric(parts[1], errors="coerce").fillna(0).astype(int)
    df["est_time_sec"] = df["base_min"] * 60 + 40 * df["increment_sec"]
    df["game_type"] = pd.cut(df["est_time_sec"], bins=[-np.inf, 179, 479, 1499, np.inf],
                             labels=GAMETYPE_ORDER)
    df["time_group"] = pd.cut(df["base_min"], bins=[-np.inf, 9, 10, np.inf], labels=TIME_ORDER)

    # Skill
    df["avg_rating"] = (df["white_rating"] + df["black_rating"]) / 2
    df["skill_level"] = pd.cut(df["avg_rating"], bins=[-np.inf, 1400, 1790, np.inf],
                               labels=SKILL_ORDER)

    # Openings
    df["opening_family"] = df["opening_name"].astype(str).str.split(":").str[0].str.strip()
    df["opening_variation"] = (df["opening_name"].astype(str).str.split(":", n=1).str[1]
                               .str.strip().fillna("(main line)"))
    top10 = df["opening_family"].value_counts().head(10).index
    df["opening_top"] = df["opening_family"].where(df["opening_family"].isin(top10), "Other")

    # Higher-rated player view
    abs_diff = df["rating_diff"].abs()
    higher_is_white = df["rating_diff"] > 0
    df["higher_rated_result"] = np.select(
        [df["rating_diff"] == 0, df["winner"] == "draw",
         (df["winner"] == "white") == higher_is_white],
        ["Equal rating", "Draw", "Higher-rated won"], default="Lower-rated won")
    df["gap_group"] = pd.cut(abs_diff, bins=[0, 50, 100, 200, 400, np.inf], labels=GAP_ORDER)

    df["date"] = df["created_at"].dt.date
    df = df.reset_index(drop=True)
    log.append({"step": "Final clean dataset", "rows": len(df), "removed": 0,
                "note": f"{df.shape[1]} columns incl. engineered features"})
    return df, log


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def white_advantage_test(df: pd.DataFrame, threshold: int) -> dict:
    """Binomial test of White vs Black wins among games with |diff| <= threshold."""
    sub = df[df["rating_diff"].abs() <= threshold]
    w = int((sub["winner"] == "white").sum())
    b = int((sub["winner"] == "black").sum())
    d = int((sub["winner"] == "draw").sum())
    n = w + b
    if n == 0:
        return {"threshold": threshold, "games": len(sub), "white_wins": w, "black_wins": b,
                "draws": d, "white_share": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "p_value": np.nan}
    res = stats.binomtest(w, n=n, p=0.5, alternative="two-sided")
    ci = res.proportion_ci(confidence_level=0.95)
    return {"threshold": threshold, "games": len(sub), "white_wins": w, "black_wins": b,
            "draws": d, "white_share": w / n, "ci_low": ci.low, "ci_high": ci.high,
            "p_value": res.pvalue}


def wilson_ci(successes: np.ndarray, n: np.ndarray, z: float = 1.96):
    successes = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = successes / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def opening_stats(df: pd.DataFrame, col: str = "opening_family", top_n: int = 15,
                  min_games: int = 50) -> pd.DataFrame:
    g = df.groupby(col, observed=True).agg(
        games=("id", "size"), white_wins=("white_win", "sum"),
        black_wins=("black_win", "sum"), draws=("is_draw", "sum"),
        white_score=("white_score", "mean"), avg_rating=("avg_rating", "mean"),
        avg_turns=("turns", "mean"))
    g = g[g["games"] >= min_games].sort_values("games", ascending=False).head(top_n)
    g["white_rate"] = g["white_wins"] / g["games"]
    g["black_rate"] = g["black_wins"] / g["games"]
    g["draw_rate"] = g["draws"] / g["games"]
    g["ci_low"], g["ci_high"] = wilson_ci(g["white_wins"], g["games"])
    return g.reset_index()


def elo_expected(white_rating: float, black_rating: float) -> float:
    """Expected score for White under the Elo model."""
    return 1 / (1 + 10 ** ((black_rating - white_rating) / 400))


def similar_games(df: pd.DataFrame, white_rating: int, black_rating: int,
                  window: int = 75) -> pd.DataFrame:
    return df[(df["white_rating"].sub(white_rating).abs() <= window)
              & (df["black_rating"].sub(black_rating).abs() <= window)]


def fit_logistic(df: pd.DataFrame):
    """P(White wins | decisive game) ~ rating_diff. Returns (intercept, slope per 100 pts)."""
    from scipy.optimize import minimize

    dec = df[df["winner"] != "draw"]
    if len(dec) < 30:
        return None
    x = dec["rating_diff"].to_numpy() / 100.0
    y = (dec["winner"] == "white").to_numpy().astype(float)

    def nll(beta):
        z = beta[0] + beta[1] * x
        return np.sum(np.logaddexp(0, z) - y * z)

    res = minimize(nll, x0=np.zeros(2), method="BFGS")
    return float(res.x[0]), float(res.x[1])


def player_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, game) so player-level stats are easy."""
    w = pd.DataFrame({
        "player": df["white_id"], "color": "white", "rating": df["white_rating"],
        "opp": df["black_id"], "opp_rating": df["black_rating"],
        "score": df["white_score"], "result": np.select(
            [df["winner"] == "white", df["winner"] == "draw"], ["win", "draw"], "loss"),
        "id": df["id"], "opening_family": df["opening_family"], "turns": df["turns"],
        "victory_status": df["victory_status"], "created_at": df["created_at"]})
    b = pd.DataFrame({
        "player": df["black_id"], "color": "black", "rating": df["black_rating"],
        "opp": df["white_id"], "opp_rating": df["white_rating"],
        "score": 1 - df["white_score"], "result": np.select(
            [df["winner"] == "black", df["winner"] == "draw"], ["win", "draw"], "loss"),
        "id": df["id"], "opening_family": df["opening_family"], "turns": df["turns"],
        "victory_status": df["victory_status"], "created_at": df["created_at"]})
    return pd.concat([w, b], ignore_index=True)


# ---------------------------------------------------------------------------
# Insights summary (mirrors the notebook's "Insights Summary", computed live)
# ---------------------------------------------------------------------------
STATUS_NAMES = {"resign": "resignation", "mate": "mate", "outoftime": "timeout", "draw": "draw"}
SHORT_SKILL = {SKILL_ORDER[0]: "beginners", SKILL_ORDER[1]: "intermediate players", SKILL_ORDER[2]: "advanced players"}


def _p(x, d=1):
    return f"{x * 100:.{d}f}%"


def insights(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return {group: [html sentence, ...]} following the notebook's three insight groups."""
    out: dict[str, list[str]] = {"Game results": [], "Game duration & endings": [], "Openings": []}
    if df.empty:
        return out
    vc = df["winner"].value_counts(normalize=True)
    levels = [s for s in SKILL_ORDER if (df["skill_level"] == s).any()]

    # ---- Game results -------------------------------------------------------
    g = out["Game results"]
    g.append(f"White won <b>{_p(vc.get('white', 0))}</b> of games vs. <b>{_p(vc.get('black', 0))}</b> "
             f"for Black (<b>{_p(vc.get('draw', 0))}</b> draws).")
    t = white_advantage_test(df, 50)
    if not pd.isna(t["white_share"]):
        if t["p_value"] < 0.05:
            g.append(f"A White advantage exists among evenly matched games (rating gap ≤ 50): White won "
                     f"<b>{_p(t['white_share'])}</b> of non-drawn games (p = {t['p_value']:.3f}). "
                     f"The effect is possible because White moves first.")
        else:
            g.append(f"Among evenly matched games (rating gap ≤ 50) White won <b>{_p(t['white_share'])}</b> of "
                     f"non-drawn games, which is not statistically significant here (p = {t['p_value']:.3f}).")
    if len(levels) >= 2:
        rr = pd.crosstab(df["skill_level"], df["winner"], normalize="index").reindex(levels)
        gaps = ((rr.get("white", 0) - rr.get("black", 0)) * 100).round(1)
        chain = " → ".join(f"{v:+.1f}" for v in gaps)
        if (gaps > 0).all() and gaps.iloc[-1] < gaps.iloc[0]:
            g.append(f"Skill and the White advantage: White wins more than Black at every skill level, but the "
                     f"gap looks smaller for stronger players (<b>{chain}</b> points, beginner → advanced).")
        else:
            g.append(f"Skill and the White advantage: the White − Black gap by skill level is <b>{chain}</b> points "
                     f"(beginner → advanced).")

    # ---- Duration & endings -------------------------------------------------
    d = out["Game duration & endings"]
    med = df.groupby("victory_status")["turns"].median()
    if len(med) >= 2:
        d.append(f"{STATUS_NAMES.get(med.idxmax(), med.idxmax()).capitalize()}s are the longest "
                 f"(median <b>{med.max():.0f}</b> turns), and {STATUS_NAMES.get(med.idxmin(), med.idxmin())}s "
                 f"the shortest (<b>{med.min():.0f}</b>).")
    cnt = df["victory_status"].value_counts()
    if len(cnt) >= 2:
        first, second = cnt.index[0], cnt.index[1]
        rest = cnt.iloc[2:]
        txt = (f"{STATUS_NAMES.get(first, first).capitalize()} is the most common ending (<b>{cnt.iloc[0]:,}</b> games), "
               f"followed by {STATUS_NAMES.get(second, second)} (<b>{cnt.iloc[1]:,}</b>)")
        if len(rest):
            names = " or by ".join(STATUS_NAMES.get(s, s) for s in rest.index)
            names = names.replace("by draw", "in a draw").replace("by timeout", "by timeout")
            txt += f", while <b>{rest.sum():,}</b> games ended {'in a draw' if rest.index.tolist() == ['draw'] else 'by ' + names}"
        d.append(txt.replace("by draw or by timeout", "in a draw or by timeout")
                    .replace("by timeout or by draw", "in a draw or by timeout") + ".")
    if len(levels) >= 2:
        sr = pd.crosstab(df["skill_level"], df["victory_status"], normalize="index").reindex(levels)
        lo, hi = levels[0], levels[-1]
        if "mate" in sr and "resign" in sr:
            m0, m1, r0, r1 = sr.loc[lo, "mate"], sr.loc[hi, "mate"], sr.loc[lo, "resign"], sr.loc[hi, "resign"]
            if m1 < m0 and r1 > r0:
                d.append("Experienced players often give up when they see their position is lost — even before "
                         "reaching checkmate — whereas beginners are more likely to play until the very end.")
            d.append(f"Checkmate endings {'decrease' if m1 < m0 else 'increase'} from <b>{_p(m0)}</b> among "
                     f"{SHORT_SKILL[lo]} to <b>{_p(m1)}</b> among {SHORT_SKILL[hi]}, while resignation endings "
                     f"{'increase' if r1 > r0 else 'decrease'} from <b>{_p(r0)}</b> to <b>{_p(r1)}</b>.")
        mt = df.groupby("skill_level", observed=True)["turns"].median().reindex(levels)
        one = {SKILL_ORDER[0]: "beginner", SKILL_ORDER[1]: "intermediate", SKILL_ORDER[2]: "advanced"}
        word = "last longer" if mt.iloc[-1] > mt.iloc[0] else "are shorter" if mt.iloc[-1] < mt.iloc[0] else "last as long"
        d.append(f"{one[hi].capitalize()} games tend to {word} than {one[lo]} games "
                 f"(median <b>{mt.iloc[-1]:.0f}</b> vs <b>{mt.iloc[0]:.0f}</b> turns).")

    # ---- Openings (top 10 families, as in the notebook) ---------------------
    o = out["Openings"]
    top10 = df["opening_family"].value_counts().head(10).index
    tg = df[df["opening_family"].isin(top10)]
    if len(top10) >= 2:
        wr = tg.groupby("opening_family")["white_win"].mean()
        o.append(f"<b>{wr.idxmax()}</b> has the highest White win rate (<b>{_p(wr.max())}</b>), and "
                 f"<b>{wr.idxmin()}</b> the lowest (<b>{_p(wr.min())}</b>).")
        lv = [s for s in (SKILL_ORDER[0], SKILL_ORDER[2]) if (tg["skill_level"] == s).any()]
        if len(lv) == 2:
            sh = pd.crosstab(tg["opening_family"], tg["skill_level"], normalize="columns")
            diff = (sh[SKILL_ORDER[2]] - sh[SKILL_ORDER[0]]).sort_values()
            up, down = diff.index[-1], diff.index[0]
            avg = tg.groupby("opening_family")["avg_rating"].mean()
            high_r = avg[up] > df["avg_rating"].mean()
            low_w = wr[up] < tg["white_win"].mean()
            fit = (f"which fits its high average rating (<b>{avg[up]:.0f}</b>)" if high_r
                   else f"with an average rating of <b>{avg[up]:.0f}</b>")
            fit += f" and {'low ' if low_w else ''}White win rate (<b>{_p(wr[up])}</b>)"
            o.append(f"Skill and openings: the <b>{up}</b> is {'much ' if diff[up] > .1 else ''}more common among "
                     f"advanced players (<b>{_p(sh.loc[up, SKILL_ORDER[2]])}</b>) than beginners "
                     f"(<b>{_p(sh.loc[up, SKILL_ORDER[0]])}</b>), {fit}.")
            if diff[down] < 0:
                o.append(f"The <b>{down}</b> is {'mostly' if diff[down] < -.1 else 'more of'} a beginner opening "
                         f"(<b>{_p(sh.loc[down, SKILL_ORDER[0]])}</b> of beginner games vs. "
                         f"<b>{_p(sh.loc[down, SKILL_ORDER[2]])}</b> of advanced games).")
    return out
