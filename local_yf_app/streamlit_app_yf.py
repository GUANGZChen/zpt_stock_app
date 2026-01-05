import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def normalize_ohlcv(df):
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        out = df.copy()
        out.columns = [c[0] for c in out.columns]
        return out
    return df


def add_ma(df, fast=25, slow=90):
    out = df.copy()
    out["MA25"] = out["Close"].rolling(fast).mean()
    out["MA90"] = out["Close"].rolling(slow).mean()
    return out


def add_macd(df, fast=12, slow=26, signal=9):
    out = df.copy()
    ema_fast = out["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["Close"].ewm(span=slow, adjust=False).mean()
    out["DIF"] = ema_fast - ema_slow
    out["DEA"] = out["DIF"].ewm(span=signal, adjust=False).mean()
    out["MACD_hist"] = out["DIF"] - out["DEA"]
    return out


def compute_signals(df, touch_zero_band=0.02):
    cross_up = (df["MA25"] > df["MA90"]) & (df["MA25"].shift(1) <= df["MA90"].shift(1))
    cross_dn = (df["MA25"] < df["MA90"]) & (df["MA25"].shift(1) >= df["MA90"].shift(1))
    underwater = cross_up & (df["Close"] < df["MA90"])

    above_red = df["MA25"] > df["MA90"]
    band = df["MA90"] * touch_zero_band
    touch_red = (df["Close"] - df["MA90"]).abs() <= band

    in_pos = False
    buy_idx = []
    sell_idx = []

    for i in range(len(df)):
        if (not in_pos) and above_red.iloc[i]:
            in_pos = True
            buy_idx.append(i)
        elif in_pos and ((not above_red.iloc[i]) or touch_red.iloc[i]):
            in_pos = False
            sell_idx.append(i)

    return cross_up, cross_dn, underwater, buy_idx, sell_idx


def make_chart(df, ticker, interval, touch_zero_band):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=(f"{ticker} | MA25/MA90", "MACD (DIF/DEA)"),
    )

    x = np.arange(len(df))
    ma25_color = "#4f8cff"
    ma90_color = "#ff6b6b"
    hist_pos = "#f1c40f"
    hist_neg = "#e74c3c"
    dif_color = "#2ecc71"
    dea_color = "#3498db"

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#2ecc71",
            decreasing_line_color="#e74c3c",
            increasing_fillcolor="rgba(46,204,113,0.35)",
            decreasing_fillcolor="rgba(231,76,60,0.35)",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=x, y=df["MA25"], mode="lines", name="MA25", line=dict(color=ma25_color, width=2.2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=df["MA90"], mode="lines", name="MA90", line=dict(color=ma90_color, width=2.2, dash="dash")),
        row=1,
        col=1,
    )

    _, _, underwater, buy_idx, sell_idx = compute_signals(df, touch_zero_band=touch_zero_band)

    if underwater.any():
        fig.add_trace(
            go.Scatter(
                x=x[underwater],
                y=df["MA25"][underwater],
                mode="markers",
                name="Underwater Golden Cross",
                marker=dict(symbol="star", size=12, color="#7b2cbf", line=dict(width=1, color="#f5f5f5")),
                hovertemplate="Underwater Cross<br>Time=%{text}<extra></extra>",
                text=df.index.strftime("%Y-%m-%d %H:%M")[underwater],
            ),
            row=1,
            col=1,
        )

    if buy_idx:
        fig.add_trace(
            go.Scatter(
                x=x[buy_idx],
                y=df["Close"].iloc[buy_idx],
                mode="text",
                name="Buy",
                text=["💗"] * len(buy_idx),
                textfont=dict(color="#ff5fa2", size=16),
                hovertemplate="Buy<br>Time=%{customdata}<br>Price=%{y:.2f}<extra></extra>",
                customdata=df.index.strftime("%Y-%m-%d %H:%M").values[buy_idx],
            ),
            row=1,
            col=1,
        )

    if sell_idx:
        fig.add_trace(
            go.Scatter(
                x=x[sell_idx],
                y=df["Close"].iloc[sell_idx],
                mode="text",
                name="Sell",
                text=["♡"] * len(sell_idx),
                textfont=dict(color="#ffffff", size=16),
                hovertemplate="Sell<br>Time=%{customdata}<br>Price=%{y:.2f}<extra></extra>",
                customdata=df.index.strftime("%Y-%m-%d %H:%M").values[sell_idx],
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x[sell_idx],
                y=df["Close"].iloc[sell_idx],
                mode="text",
                name="Sell Crack",
                text=["✕"] * len(sell_idx),
                textfont=dict(color="#777777", size=12),
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    hist = pd.to_numeric(df["MACD_hist"], errors="coerce").fillna(0.0)
    hist_pos_vals = hist.where(hist >= 0, 0.0)
    hist_neg_vals = hist.where(hist < 0, 0.0)
    fig.add_trace(
        go.Bar(x=x, y=hist_pos_vals, name="Hist+", marker_color=hist_pos, opacity=0.9),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=x, y=hist_neg_vals, name="Hist-", marker_color=hist_neg, opacity=0.9),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=df["DIF"], mode="lines", name="DIF", line=dict(color=dif_color, width=1.6)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=df["DEA"], mode="lines", name="DEA", line=dict(color=dea_color, width=1.6, dash="dot")),
        row=2,
        col=1,
    )

    step = max(len(df) // 8, 1)
    tickvals = x[::step]
    ticktext = df.index.strftime("%Y-%m-%d %H:%M")[::step]

    fig.update_layout(
        title=f"{ticker} | {interval} | touch_band={touch_zero_band:.3f}",
        xaxis_rangeslider_visible=False,
        height=950,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=80, b=40),
        barmode="relative",
        bargap=0.05,
    )
    fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, showgrid=False, row=2, col=1)
    fig.update_xaxes(showgrid=False, row=1, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor="rgba(255,255,255,0.08)")

    max_hist = float(np.nanmax(np.abs(hist.values))) if len(hist) else 0.0
    max_line = float(
        np.nanmax(
            np.abs(
                pd.concat(
                    [
                        pd.to_numeric(df["DIF"], errors="coerce"),
                        pd.to_numeric(df["DEA"], errors="coerce"),
                    ]
                ).values
            )
        )
    )
    price_scale = float(pd.to_numeric(df["Close"], errors="coerce").mean()) if len(df) else 0.0
    min_range = max(price_scale * 0.0005, 0.01)
    max_hist = max(max_hist, max_line, min_range)
    fig.update_yaxes(
        title_text="MACD",
        row=2,
        col=1,
        gridcolor="rgba(255,255,255,0.08)",
        range=[-max_hist * 1.2, max_hist * 1.2],
    )

    return fig


@st.cache_data(ttl=300)
def latest_trade_date(symbol):
    df = yf.download(symbol, period="10d", interval="1d", auto_adjust=True, progress=False)
    df = normalize_ohlcv(df)
    if df is None or df.empty:
        return datetime.utcnow().date()
    return df.index[-1].date()


def main():
    st.set_page_config(page_title="ZPT Stock App (Local YF)", layout="wide")
    st.title("ZPT Stock App (Local YF)")

    with st.sidebar:
        ticker = st.text_input("Ticker", value="AAPL")
        interval = st.selectbox("Interval", ["1m", "2m", "5m", "15m", "30m", "60m", "1d", "1wk"], index=0)
        default_date = latest_trade_date(ticker) if ticker else datetime.utcnow().date()
        date_range = st.date_input(
            "Date range",
            value=(default_date, default_date),
        )
        touch_zero_band = st.slider("Touch Band (±%)", min_value=0.0, max_value=0.2, value=0.0, step=0.005)
        auto_refresh = st.checkbox("Auto refresh", value=False)
        refresh_sec = st.slider("Refresh (sec)", min_value=5, max_value=120, value=15, step=5)
        if st.button("Refresh data"):
            st.cache_data.clear()

    if auto_refresh:
        st_autorefresh = getattr(st, "autorefresh", None)
        if st_autorefresh:
            st_autorefresh(interval=refresh_sec * 1000, key="auto_refresh")
        else:
            time.sleep(refresh_sec)
            st.experimental_rerun()

    if not ticker:
        st.warning("Please enter a ticker.")
        return

    start_date = None
    end_date = None
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = datetime.combine(date_range[0], datetime.min.time())
        end_date = datetime.combine(date_range[1], datetime.max.time()) + timedelta(days=1)

    if start_date and end_date and start_date >= end_date:
        st.error("Invalid date range.")
        return

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    df = normalize_ohlcv(df)

    if df is None or df.empty:
        st.error("No data returned. Try another ticker or timeframe.")
        return

    df = add_ma(df)
    df = add_macd(df)
    fig = make_chart(df, ticker, interval, touch_zero_band)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": True, "responsive": True},
    )


if __name__ == "__main__":
    main()
