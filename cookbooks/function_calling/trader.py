"""A small trader assistant, and a cached TypeSafe client - the parts the cookbook does not
teach.

The ten functions are ordinary code with ordinary signatures. Their closed-set arguments are
``Literal``s, a set of them is a ``list[Literal[...]]``, and a switch is a ``bool``; that is all
the cookbook needs from them. Bars are mock, built from a seed on first import.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np
import polars as pl
from cooksafe import JsonCache
from typesafe_sdk import SystemOneResponse, TypeSafeClient

matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt  # noqa: E402

TYPESAFE_MODEL = "jev-1.12"
json_cache = JsonCache(Path(__file__).with_name("json_cache.json"))
_client = TypeSafeClient(
    api_key=os.environ.get(
        "TYPESAFE_API_KEY", "cache-only"
    ),  # keyless kernels replay the cache
    base_url=os.environ.get("TYPESAFE_ENDPOINT"),
    timeout=120.0,
)


@json_cache
def _system_one(model: str, state: str, questions: str) -> dict:
    response = _client.system_one(
        state=state, questions=json.loads(questions), model=model
    )
    return {
        "model": response.model,
        "answers": {
            key: answer.model_dump(mode="json")
            for key, answer in response.answers.items()
        },
        "usage": response.usage.model_dump(mode="json"),
    }


class Client:
    """A TypeSafe client whose every call is memoized to json_cache.json."""

    def system_one(
        self, state: Any, questions: dict, model: str = TYPESAFE_MODEL
    ) -> SystemOneResponse:
        raw = _system_one(model, str(state), json.dumps(questions, sort_keys=True))
        return SystemOneResponse.model_validate(raw)


client = Client()

BARS = Path(__file__).with_name("bars.parquet")
SESSION_MINUTES = 390  # 09:30 through 15:59
MARKET_VOL = 0.14
# annual drift, idiosyncratic vol, beta on the market factor, starting price, average volume
SPECS = {
    "SPY": (0.08, 0.03, 1.00, 612.0, 74_000),
    "NVDA": (0.34, 0.28, 1.60, 178.0, 210_000),
    "AMD": (0.20, 0.30, 1.50, 164.0, 96_000),
    "AAPL": (0.11, 0.16, 1.05, 229.0, 118_000),
    "MSFT": (0.13, 0.14, 0.95, 468.0, 61_000),
    "TSLA": (0.04, 0.45, 1.25, 331.0, 145_000),
}


def build_bars() -> pl.DataFrame:
    day, days = date(2026, 4, 27), []
    while day <= date(2026, 7, 28):
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    stamps = [
        datetime.combine(d, datetime.min.time()) + timedelta(hours=9, minutes=30 + m)
        for d in days
        for m in range(SESSION_MINUTES)
    ]
    rng, n = np.random.default_rng(20260728), len(days) * SESSION_MINUTES
    dt = 1.0 / (252 * SESSION_MINUTES)
    market = rng.standard_normal(n) * np.sqrt(dt) * MARKET_VOL
    minute = np.arange(n) % SESSION_MINUTES
    # busier and choppier at the open and the close; scaled so it leaves realized vol alone
    shape = 1.0 + 1.6 * np.exp(-minute / 45) + 1.1 * np.exp(-(389 - minute) / 40)
    intraday = shape / np.sqrt(np.mean(shape**2))
    frames = []
    for symbol, (drift, idio_vol, beta, price0, volume0) in SPECS.items():
        total = np.hypot(beta * MARKET_VOL, idio_vol)
        idio = rng.standard_normal(n) * np.sqrt(dt) * idio_vol
        close = price0 * np.exp(
            np.cumsum((drift - 0.5 * total**2) * dt + intraday * (beta * market + idio))
        )
        open_ = np.concatenate([[price0], close[:-1]])
        wick = np.abs(close) * total * np.sqrt(dt) * intraday
        frames.append(
            pl.DataFrame(
                {
                    "timestamp": stamps,
                    "symbol": [symbol] * n,
                    "open": np.round(open_, 4),
                    "high": np.round(
                        np.maximum(open_, close)
                        + np.abs(rng.standard_normal(n)) * wick,
                        4,
                    ),
                    "low": np.round(
                        np.minimum(open_, close)
                        - np.abs(rng.standard_normal(n)) * wick,
                        4,
                    ),
                    "close": np.round(close, 4),
                    "volume": (volume0 * intraday * rng.gamma(4.0, 0.25, n)).astype(
                        np.int64
                    ),
                }
            )
        )
    return pl.concat(frames).sort("timestamp", "symbol")


if not BARS.exists():
    build_bars().write_parquet(BARS, compression="zstd")


@cache
def load() -> pl.DataFrame:
    return pl.read_parquet(BARS)


Symbol = Literal["SPY", "NVDA", "AMD", "AAPL", "MSFT", "TSLA"]
Resolution = Literal["1m", "5m", "15m", "1h", "1d"]
Window = Literal["1d", "1w", "1mo", "3mo"]

SPAN = {"1w": timedelta(days=7), "1mo": timedelta(days=30), "3mo": timedelta(days=92)}
BARS_PER_DAY = 390


def within(window: Window, symbols: list[str] | None = None) -> pl.DataFrame:
    bars = load() if symbols is None else load().filter(pl.col("symbol").is_in(symbols))
    last = load()["timestamp"].max()
    if window == "1d":  # the final session, whole
        return bars.filter(pl.col("timestamp").dt.date() == last.date())
    return bars.filter(pl.col("timestamp") >= last - SPAN[window])


def resample(bars: pl.DataFrame, resolution: Resolution) -> pl.DataFrame:
    if resolution == "1m":
        return bars
    return (
        bars.sort("timestamp")
        .group_by_dynamic("timestamp", every=resolution, group_by="symbol")
        .agg(
            open=pl.col("open").first(),
            high=pl.col("high").max(),
            low=pl.col("low").min(),
            close=pl.col("close").last(),
            volume=pl.col("volume").sum(),
        )
    )


def _panel(title: str, height: float = 3.2):
    figure, axes = plt.subplots(figsize=(9, height))
    axes.set_title(title, fontsize=10)
    axes.grid(alpha=0.25, linewidth=0.5)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    return figure, axes


def list_symbols() -> str:
    """What data is loaded."""
    counts = load().group_by("symbol").len().sort("symbol")
    span = load()["timestamp"].min(), load()["timestamp"].max()
    return (
        f"{counts.height} symbols, 1-minute bars, {span[0]:%Y-%m-%d} to {span[1]:%Y-%m-%d}\n"
        + "\n".join(
            f"  {row['symbol']:<6}{row['len']:>8,} bars"
            for row in counts.iter_rows(named=True)
        )
    )


def market_summary(window: Window = "1d") -> str:
    """How every symbol moved over a window."""
    rows = (
        within(window)
        .sort("timestamp")
        .group_by("symbol")
        .agg(
            first=pl.col("close").first(),
            last=pl.col("close").last(),
            volume=pl.col("volume").sum(),
        )
        .with_columns(change=(pl.col("last") / pl.col("first") - 1) * 100)
        .sort("change", descending=True)
    )
    return f"the board over {window}\n" + "\n".join(
        f"  {r['symbol']:<6}{r['last']:>9.2f}{r['change']:>8.2f}%{r['volume']:>15,}"
        for r in rows.iter_rows(named=True)
    )


def summary_stats(symbol: Symbol, window: Window = "1mo") -> str:
    """Close-to-close statistics for one symbol."""
    bars = within(window, [symbol]).sort("timestamp")
    returns = bars.select(pl.col("close").pct_change()).drop_nulls().to_series()
    stats = {
        "bars": bars.height,
        "last": bars["close"][-1],
        "high": bars["high"].max(),
        "low": bars["low"].min(),
        "total return %": (bars["close"][-1] / bars["close"][0] - 1) * 100,
        "1m return sd bp": returns.std() * 10_000,
        "worst 1m bp": returns.min() * 10_000,
    }
    return f"{symbol} over {window}\n" + "\n".join(
        f"  {name:<18}{value:>14,.2f}" for name, value in stats.items()
    )


def volatility(symbol: Symbol, window: Window = "1mo", annualized: bool = True) -> str:
    """Realized volatility from one-minute returns."""
    returns = (
        within(window, [symbol])
        .sort("timestamp")
        .select(pl.col("close").pct_change())
        .drop_nulls()
        .to_series()
    )
    scale = (252 * BARS_PER_DAY) ** 0.5 if annualized else 1.0
    unit = "annualized" if annualized else "per one-minute bar"
    return (
        f"{symbol} realized volatility over {window}: {returns.std() * scale * 100:.2f}% "
        f"({unit}), from {returns.len():,} returns"
    )


def top_movers(
    window: Window = "1d",
    direction: Literal["gainers", "losers"] = "gainers",
    limit: int = 3,
) -> str:
    """The biggest movers over a window."""
    rows = (
        within(window)
        .sort("timestamp")
        .group_by("symbol")
        .agg(first=pl.col("close").first(), last=pl.col("close").last())
        .with_columns(change=(pl.col("last") / pl.col("first") - 1) * 100)
        .sort("change", descending=direction == "gainers")
        .head(limit)
    )
    return f"top {limit} {direction} over {window}\n" + "\n".join(
        f"  {r['symbol']:<6}{r['change']:>8.2f}%  ->  {r['last']:.2f}"
        for r in rows.iter_rows(named=True)
    )


def plot_price(
    symbol: Symbol,
    style: Literal["line", "candles"] = "line",
    resolution: Resolution = "15m",
    window: Window = "1w",
    include_volume: bool = False,
    moving_average: Literal["9", "20", "50"] | None = None,
    log_scale: bool = False,
):
    """One symbol's price, drawn as a line or as candles."""
    bars = resample(within(window, [symbol]), resolution).sort("timestamp")
    title = f"{symbol}  {style}  {resolution}  last {window}"
    if include_volume:
        figure, (axes, lower) = plt.subplots(
            2, 1, figsize=(9, 4.4), sharex=True, height_ratios=[3, 1]
        )
        axes.set_title(title, fontsize=10)
        lower.bar(bars["timestamp"], bars["volume"], width=0.4, color="#8899aa")
        lower.set_ylabel("volume", fontsize=8)
        for panel in (axes, lower):
            panel.grid(alpha=0.25, linewidth=0.5)
    else:
        figure, axes = _panel(title, 3.6)
    if style == "candles":
        width = (
            0.6 * (bars["timestamp"][1] - bars["timestamp"][0]).total_seconds() / 86400
        )
        for row in bars.iter_rows(named=True):
            colour = "#2f855a" if row["close"] >= row["open"] else "#c53030"
            axes.plot(
                [row["timestamp"]] * 2,
                [row["low"], row["high"]],
                color=colour,
                linewidth=0.6,
            )
            axes.bar(
                row["timestamp"],
                abs(row["close"] - row["open"]),
                bottom=min(row["open"], row["close"]),
                width=width,
                color=colour,
            )
    else:
        axes.plot(bars["timestamp"], bars["close"], linewidth=1.1, color="#2b6cb0")
    if moving_average is not None:
        smoothed = bars.select(
            pl.col("close").rolling_mean(int(moving_average))
        ).to_series()
        axes.plot(
            bars["timestamp"],
            smoothed,
            linewidth=1.2,
            color="#dd6b20",
            label=f"{moving_average}-bar average",
        )
        axes.legend(fontsize=8, frameon=False)
    if log_scale:
        axes.set_yscale("log")
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


def intraday_pattern(
    symbol: Symbol,
    window: Window = "1mo",
    metric: Literal["volume", "volatility", "return"] = "volume",
):
    """The average shape of a trading day, in fifteen-minute buckets."""
    bars = within(window, [symbol]).sort("timestamp")
    labels = {
        "volume": "mean volume",
        "volatility": "sd of 1m return (bp)",
        "return": "mean 1m return (bp)",
    }
    bucketed = (
        bars.with_columns(
            ret=pl.col("close").pct_change(),
            # cast first: dt.hour() is Int8, so hour * 60 overflows past 02:00
            bucket=(
                pl.col("timestamp").dt.hour().cast(pl.Int32) * 60
                + pl.col("timestamp").dt.minute()
            )
            // 15
            * 15,
        )
        .drop_nulls()
        .group_by("bucket")
        .agg(
            volume=pl.col("volume").mean(),
            volatility=pl.col("ret").std() * 10_000,
            **{"return": pl.col("ret").mean() * 10_000},
        )
        .sort("bucket")
    )
    figure, axes = _panel(
        f"{symbol}  average session by time of day, {metric}, last {window}"
    )
    hours = [f"{b // 60:02d}:{b % 60:02d}" for b in bucketed["bucket"]]
    axes.bar(hours, bucketed[metric], color="#2c7a7b")
    axes.set_ylabel(labels[metric], fontsize=8)
    axes.tick_params(axis="x", labelrotation=90, labelsize=7)
    figure.tight_layout()
    return figure


def compare_returns(
    symbols: list[Symbol], window: Window = "1mo", normalize: bool = True
):
    """Several symbols on one chart."""
    bars = resample(within(window, list(symbols)), "1h").sort("timestamp")
    figure, axes = _panel(f"{' vs '.join(symbols)}  last {window}")
    for symbol in symbols:
        series = bars.filter(pl.col("symbol") == symbol)
        values = series["close"]
        if normalize:
            values = (values / values[0] - 1) * 100
        axes.plot(series["timestamp"], values, linewidth=1.1, label=symbol)
    axes.set_ylabel("return from start (%)" if normalize else "price", fontsize=8)
    axes.legend(fontsize=8, frameon=False, ncols=len(symbols))
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


def rolling_correlation(
    symbol: Symbol,
    benchmark: Symbol = "SPY",
    window: Window = "1mo",
    resolution: Resolution = "1h",
):
    """Rolling correlation of returns between two symbols."""
    wide = (
        resample(within(window, [symbol, benchmark]), resolution)
        .pivot(on="symbol", index="timestamp", values="close")
        .sort("timestamp")
        .select(
            "timestamp", a=pl.col(symbol).pct_change(), b=pl.col(benchmark).pct_change()
        )
        .drop_nulls()
    )
    span = max(10, min(60, wide.height // 4))
    overall = wide.select(pl.corr("a", "b")).item()
    wide = wide.with_columns(
        rho=pl.rolling_corr("a", "b", window_size=span)
    ).drop_nulls()
    figure, axes = _panel(
        f"{symbol} vs {benchmark}  rolling correlation of {resolution} returns, last {window}"
    )
    axes.plot(wide["timestamp"], wide["rho"], linewidth=1.1, color="#6b46c1")
    axes.axhline(
        overall,
        linestyle="--",
        linewidth=0.9,
        color="#718096",
        label=f"whole window {overall:.2f}",
    )
    axes.set_ylim(-1.05, 1.05)
    axes.legend(fontsize=8, frameon=False)
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


def drawdown(symbol: Symbol, window: Window = "3mo", plot: bool = False):
    """Worst peak-to-trough fall inside a window."""
    curve = (
        within(window, [symbol])
        .sort("timestamp")
        .select("timestamp", "close", peak=pl.col("close").cum_max())
        .with_columns(drawdown=(pl.col("close") / pl.col("peak") - 1) * 100)
    )
    worst = curve.filter(pl.col("drawdown") == pl.col("drawdown").min()).row(
        0, named=True
    )
    text = (
        f"{symbol} worst drawdown over {window}: {worst['drawdown']:.2f}%, trough "
        f"{worst['timestamp']:%Y-%m-%d %H:%M} at {worst['close']:.2f} from {worst['peak']:.2f}"
    )
    if not plot:
        return text
    figure, axes = _panel(f"{symbol} drawdown from running peak, last {window}")
    axes.fill_between(
        curve["timestamp"], curve["drawdown"], 0, color="#c53030", alpha=0.35
    )
    axes.set_ylabel("%", fontsize=8)
    figure.autofmt_xdate()
    figure.tight_layout()
    print(text)
    return figure


TOOLS = {
    fn.__name__: fn
    for fn in (
        list_symbols,
        market_summary,
        plot_price,
        intraday_pattern,
        compare_returns,
        rolling_correlation,
        summary_stats,
        volatility,
        top_movers,
        drawdown,
    )
}
