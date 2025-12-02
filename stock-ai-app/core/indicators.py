import pandas as pd


def compute_indicators(hist: pd.DataFrame, info: dict):
    latest_price = hist["Close"].iloc[-1] if not hist.empty else None

    # --- 動能指標 ---
    mom = {}

    if len(hist) >= 22:
        mom["oneMonthReturn"] = latest_price / hist["Close"].iloc[-22] - 1
    else:
        mom["oneMonthReturn"] = None

    if len(hist) >= 66:
        mom["threeMonthReturn"] = latest_price / hist["Close"].iloc[-66] - 1
    else:
        mom["threeMonthReturn"] = None

    if "High" in hist:
        mom["high3m"] = hist["High"].tail(66).max()
    else:
        mom["high3m"] = None

    if "Low" in hist:
        mom["low3m"] = hist["Low"].tail(66).min()
    else:
        mom["low3m"] = None

    # 波動度
    mom["volatility3m"] = (
        hist["Close"].pct_change().std() if len(hist) > 2 else None
    )

    # --- 估值 ---
    valuation = {
        "latestPrice": latest_price,
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "priceToBook": info.get("priceToBook"),
    }

    return {
        "valuation": valuation,
        "momentum": mom,
    }
