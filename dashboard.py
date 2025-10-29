# ===========================================
# Minimal "Weekly Macro Rotation" Report (CI-safe)
# Writes report.html with last close + 1w change for a few tickers.
# No pandas_datareader, no notebook magics, no external assets.
# ===========================================

import sys, time
import datetime as dt
import io
import numpy as np
import pandas as pd
import yfinance as yf

# ---- Config ----
TICKERS = ["SPY", "EEM", "TLT", "HYG", "GLD", "DBC", "UUP", "BTC-USD", "ETH-USD"]
START   = "2023-01-01"
INTERVAL_WEEKLY = "1wk"
INTERVAL_DAILY  = "1d"

def fetch_weekly_closes(tickers, start):
    """Try weekly first; if empty, fall back to daily and resample to weekly (Fri close)."""
    # Try weekly direct
    data = yf.download(
        tickers=tickers,
        start=start,
        interval=INTERVAL_WEEKLY,
        auto_download=True,
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=True,
    )
    closes = None
    try:
        if isinstance(data.columns, pd.MultiIndex):
            # e.g. ('Close', 'SPY'), ('Close','EEM'), ...
            if "Close" in data.columns.get_level_values(0):
                closes = data["Close"].copy()
        else:
            # columns are ['Open','High','Low','Close',...] with each column as Series of arrays? (yfinance groups by column)
            closes = data.get("Close")
    except Exception:
        closes = None

    if closes is not None and isinstance(closes, pd.DataFrame) and not closes.empty:
        closes = closes.dropna(how="all")
        return closes

    # Fallback: daily -> weekly (last business week close, Friday)
    daily = yf.download(
        tickers=tickers,
        start=start,
        interval=INTERVAL_DAILY,
        auto_download=True,
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=True,
    )
    if isinstance(daily.columns, pd.MultiIndex):
        if "Close" in daily.columns.get_level_values(0):
            daily_close = daily["Close"].dropna(how="all")
        else:
            raise RuntimeError("Could not find Close prices in daily data.")
    else:
        daily_close = daily.get("Close")
        if daily_close is None:
            raise RuntimeError("Could not find Close prices in daily data.")

    # Resample to weekly (Friday)
    weekly = daily_close.resample("W-FRI").last().dropna(how="all")
    # Keep only requested tickers (if some missing, ignore)
    cols = [c for c in tickers if c in weekly.columns]
    if not cols:
        raise RuntimeError("No valid tickers after resampling.")
    return weekly[cols]

def build_html():
    closes = fetch_weekly_closes(TICKERS, START)
    returns = closes.pct_change()

    last_week = closes.index.max()
    prev_week = closes.index[closes.index < last_week].max() if closes.index.size > 1 else None

    last_close = closes.loc[last_week]
    last_change = returns.loc[last_week] if last_week in returns.index else pd.Series(index=closes.columns, dtype=float)

    # Format numbers
    def fmt_price(x):
        if pd.isna(x): return "—"
        # crude guess: crypto has bigger numbers; just format to 2 decimals
        return f"{x:,.2f}"

    def fmt_pct(x):
        if pd.isna(x): return "—"
        return f"{x*100:+.2f}%"

    rows = []
    for t in TICKERS:
        if t in last_close.index:
            rows.append((t, fmt_price(last_close.get(t, np.nan)), fmt_pct(last_change.get(t, np.nan))))
        else:
            rows.append((t, "—", "—"))

    # Minimal CSS + HTML, built as a list to avoid triple-quote pitfalls
    built = []
    built.append("<!doctype html>")
    built.append("<meta charset='utf-8'>")
    built.append("<title>Weekly Macro Snapshot</title>")
    built.append("""
<style>
body{font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; color:#111; background:#fff; margin:24px;}
h1{font-size:20px; margin:0 0 8px 0;}
.small{color:#555; font-size:12px; margin-bottom:16px;}
table{border-collapse:collapse; width:100%; max-width:820px;}
th,td{border:1px solid #e5e7eb; padding:8px 10px; text-align:right;}
th{background:#111; color:#fff; font-weight:600;}
td.symbol{text-align:left;}
tr:nth-child(even){background:#f9fafb;}
.badge{display:inline-block; padding:2px 8px; background:#111; color:#fff; border-radius:999px; font-size:11px; margin-left:8px;}
.footer{margin-top:16px; font-size:12px; color:#555;}
</style>
    """)
    ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    built.append(f"<h1>Weekly Macro Snapshot <span class='badge'>{ts}</span></h1>")
    if prev_week is not None:
        built.append(f"<div class='small'>Last close: {last_week.date()} &nbsp;·&nbsp; 1-week change vs {prev_week.date()}</div>")
    else:
        built.append(f"<div class='small'>Last close: {last_week.date()}</div>")

    built.append("<table>")
    built.append("<tr><th style='text-align:left'>Ticker</th><th>Last Close</th><th>1w %</th></tr>")
    for sym, px, chg in rows:
        built.append(f"<tr><td class='symbol'>{sym}</td><td>{px}</td><td>{chg}</td></tr>")
    built.append("</table>")

    built.append("""
<div class='footer'>
This is a minimal sanity-check report generated from Yahoo Finance weekly (fallback: daily→weekly).<br>
Once this runs clean in CI and emails correctly, we can layer back richer factors (FRED, momentum tiles, etc.).
</div>
    """)

    return "\n".join(built)

if __name__ == "__main__":
    try:
        html = build_html()
        if not html or len(html) < 300:
            raise RuntimeError("Generated HTML looks too small.")
    except Exception as e:
        # Always emit *something* so CI can publish a page
        html = f"""<!doctype html><meta charset='utf-8'>
        <title>Weekly Macro Snapshot — Build Error</title>
        <pre style="font-family:monospace;white-space:pre-wrap">{e}</pre>"""
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("✅ wrote report.html", len(html), "bytes")
