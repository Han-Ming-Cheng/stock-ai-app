import yfinance as yf
import pandas as pd


def fetch_us_stock(symbol: str, period="3mo"):
    symbol = symbol.upper()
    ticker = yf.Ticker(symbol)

    # 價格資料
    hist = ticker.history(period=period)

    # 基本資訊
    basic_info = ticker.info or {}

    return {
        "symbol": symbol,
        "price_history": hist,
        "basic_info": basic_info,
        "fundamentals_raw": basic_info,
    }


def fetch_earnings_summary(symbol: str):
    ticker = yf.Ticker(symbol)
    try:
        return ticker.earnings or pd.DataFrame()
    except:
        return pd.DataFrame()


def fetch_financial_statements(symbol: str):
    ticker = yf.Ticker(symbol)

    try:
        income_q = ticker.quarterly_financials.T
    except:
        income_q = pd.DataFrame()

    def fix(df):
        if df is None or df.empty:
            return df
        df = df.reset_index()
        df.rename(columns={"index": "period"}, inplace=True)
        return df

    return {
        "income_q": fix(income_q),
    }
