import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# =========================
# INDICATORS
# =========================

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def ATR(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def volume_rsi(volume, period=14):
    delta = volume.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# =========================
# FEATURE ENGINEERING
# =========================

def add_features(df):
    df = df.copy()

    # Returns
    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_3"] = df["Close"].pct_change(3)
    df["ret_5"] = df["Close"].pct_change(5)

    # EMAs
    df["ema_5"] = df["Close"].ewm(span=5).mean()
    df["ema_10"] = df["Close"].ewm(span=10).mean()
    df["ema_20"] = df["Close"].ewm(span=20).mean()

    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()

    # RSI
    df["rsi"] = RSI(df["Close"])
    df["rsi_delta"] = df["rsi"].diff()

    # ATR
    df["atr"] = ATR(df)
    df["atr_norm"] = df["atr"] / df["Close"]

    # Volume features
    df["vol_rsi"] = volume_rsi(df["Volume"])
    df["vol_change"] = df["Volume"].pct_change()

    # Price action
    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["co_body"] = (df["Close"] - df["Open"]) / df["Close"]

    return df.dropna()


# =========================
# STRATEGY BACKTEST
# =========================

def backtest_strategy(df, prob_long=0.6, prob_short=0.4):
    df = df.copy()

    # Positions
    df["position"] = 0
    df.loc[df["prob_up"] > prob_long, "position"] = 1
    df.loc[df["prob_up"] < prob_short, "position"] = -1

    # Returns
    df["strategy_ret"] = df["position"].shift(1) * df["ret_1"]
    df["cum_pnl"] = (1 + df["strategy_ret"]).cumprod()

    # Metrics
    sharpe = np.sqrt(252) * df["strategy_ret"].mean() / df["strategy_ret"].std()

    rolling_max = df["cum_pnl"].cummax()
    drawdown = df["cum_pnl"] / rolling_max - 1
    max_dd = drawdown.min()

    return df, sharpe, max_dd


# =========================
# MAIN PIPELINE
# =========================

def run_pipeline(csv_path):
    df = pd.read_csv(csv_path)

    # Fix column names
    df.columns = [c.capitalize() for c in df.columns]

    # Sort if datetime exists
    if "Datetime" in df.columns:
        df = df.sort_values("Datetime")

    df = add_features(df)

    # Target: next candle direction
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()

    features = [
        "ret_1", "ret_3", "ret_5",
        "ema_5", "ema_10", "ema_20",
        "macd", "macd_signal",
        "rsi", "rsi_delta",
        "atr_norm",
        "vol_rsi", "vol_change",
        "hl_range", "co_body"
    ]

    X = df[features]
    y = df["target"]

    split = int(0.7 * len(df))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    df.loc[X_test.index, "prob_up"] = model.predict_proba(X_test)[:, 1]

    bt_df, sharpe, max_dd = backtest_strategy(df.loc[X_test.index])

    print("Sharpe Ratio:", round(sharpe, 2))
    print("Max Drawdown:", round(max_dd * 100, 2), "%")
    print("Final PnL:", round(bt_df["cum_pnl"].iloc[-1], 2))

    return bt_df


# =========================
# RUN
# =========================

bt = run_pipeline("ircon.csv")
