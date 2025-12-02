from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components
import yfinance as yf  # 抓 Yahoo 股價

from core.data_fetch import (
    fetch_us_stock,
    fetch_earnings_summary,
    fetch_financial_statements,
)
from core.indicators import compute_indicators
from core.ai_analyzer import (
    generate_analysis,
    extract_earnings_insights,
    translate_transcript_paragraphs,
    analyze_earnings_transcript,
)

st.set_page_config(page_title="美股 AI 分析工具", layout="wide")

# ========= 初始化 Session State =========
if "analyzed" not in st.session_state:
    st.session_state["analyzed"] = False
    st.session_state["last_symbol"] = ""
    st.session_state["last_period"] = "3mo"


# ========= 近一小時 or 最近收盤價 =========
def fetch_last_1h_price(symbol: str):
    """
    先嘗試抓近一小時 1 分鐘線；若抓不到，改回傳最近收盤價與前一日比較。
    回傳 dict: {last, change, pct, source}
        source = "intraday" 或 "last_close"
    """
    try:
        ticker = yf.Ticker(symbol)

        # 1. 先試 intraday（可能有延遲）
        df = ticker.history(period="2h", interval="1m")
        if df is not None and not df.empty:
            if len(df) >= 60:
                last_hour = df.tail(60)
            else:
                last_hour = df

            last = float(last_hour["Close"].iloc[-1])
            first = float(last_hour["Close"].iloc[0])
            pct = (last - first) / first if first != 0 else 0.0

            return {
                "last": last,
                "change": last - first,
                "pct": pct,
                "source": "intraday",
            }

        # 2. 若抓不到 1 分鐘線，就改抓日線最近收盤
        daily = ticker.history(period="5d", interval="1d")
        if daily is None or daily.empty:
            return None
        last_close = float(daily["Close"].iloc[-1])
        if len(daily) >= 2:
            prev_close = float(daily["Close"].iloc[-2])
        else:
            prev_close = last_close
        change = last_close - prev_close
        pct = (last_close - prev_close) / prev_close if prev_close != 0 else 0.0

        return {
            "last": last_close,
            "change": change,
            "pct": pct,
            "source": "last_close",
        }
    except Exception:
        return None


# ========= 最近一個交易日的 MA / Volume =========
def fetch_last_daily_ma_volume(symbol: str):
    """
    抓最近一個交易日的 MA5 / MA10 / MA20 / 成交量。
    回傳 dict: {date, ma5, ma10, ma20, volume}
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="3mo", interval="1d")
        if df is None or df.empty:
            return None

        last_row = df.iloc[-1]
        close_series = df["Close"]

        def last_ma(window: int):
            if len(close_series) >= window:
                return float(close_series.rolling(window).mean().iloc[-1])
            else:
                return None

        ma5 = last_ma(5)
        ma10 = last_ma(10)
        ma20 = last_ma(20)
        volume = float(last_row["Volume"]) if "Volume" in last_row else None
        date = last_row.name.strftime("%Y-%m-%d")

        return {
            "date": date,
            "ma5": ma5,
            "ma10": ma10,
            "ma20": ma20,
            "volume": volume,
        }
    except Exception:
        return None


# ========= 專業版圖表（K 線綠漲紅跌 + MA 疊線 + 彩色畫線 + 多步 Undo/Redo） =========
def render_pro_chart(hist: pd.DataFrame, period: str):
    st.subheader(f"📉 股價走勢（{period}）")

    if hist is None or hist.empty:
        st.warning("⚠ 找不到股價資料。")
        return

    required_cols = ["Open", "High", "Low", "Close"]
    has_ohlc = all(col in hist.columns for col in required_cols)

    if has_ohlc:
        chart_type = st.radio(
            "圖表類型",
            ["收盤價折線圖", "K 線圖（蠟燭圖）"],
            horizontal=True,
        )
    else:
        st.info("⚠ 此股票缺少開高低收（OHLC）資料，無法顯示 K 線圖。")
        chart_type = "收盤價折線圖"

    # 👉 是否顯示 MA 線 的切換按鈕
    show_ma = st.checkbox("顯示 MA5 / MA10 / MA20", value=True)

    # 👉 計算 MA5 / MA10 / MA20
    ma_df = None
    if "Close" in hist.columns:
        close = hist["Close"]
        ma_df = pd.DataFrame(index=hist.index)
        ma_df["MA5"] = close.rolling(5).mean()
        ma_df["MA10"] = close.rolling(10).mean()
        ma_df["MA20"] = close.rolling(20).mean()

    # ---------- 建立 Plotly 圖 ----------
    if chart_type == "收盤價折線圖":
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist["Close"],
                mode="lines",
                name="收盤價",
                line=dict(color="#0050b3", width=2),  # ⭐ 收盤價深藍色
            )
        )

        # 把 MA 線疊到折線圖上（若有打勾）
        if show_ma and ma_df is not None:
            if ma_df["MA5"].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=ma_df.index,
                        y=ma_df["MA5"],
                        mode="lines",
                        name="MA5",
                        line=dict(color="#ffa500", width=1.5),  # ⭐ 橘
                    )
                )
            if ma_df["MA10"].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=ma_df.index,
                        y=ma_df["MA10"],
                        mode="lines",
                        name="MA10",
                        line=dict(color="#2ca02c", width=1.3),  # ⭐ 綠
                    )
                )
            if ma_df["MA20"].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=ma_df.index,
                        y=ma_df["MA20"],
                        mode="lines",
                        name="MA20",
                        line=dict(color="#9467bd", width=1.3),  # ⭐ 紫
                    )
                )

    else:
        # K 線圖
        try:
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=hist.index,
                        open=hist["Open"],
                        high=hist["High"],
                        low=hist["Low"],
                        close=hist["Close"],
                        name="K 線",
                        increasing_line_color="green",
                        increasing_fillcolor="green",
                        decreasing_line_color="red",
                        decreasing_fillcolor="red",
                    )
                ]
            )
        except Exception as e:
            ...
            # 這裡如果 fallback 成折線圖記得也改顏色
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=hist["Close"],
                    mode="lines",
                    name="收盤價",
                    line=dict(color="#0050b3", width=2),  # ⭐ 一樣深藍
                )
            )

        # 把 MA 線疊到 K 線圖上（若有打勾）
        if show_ma and ma_df is not None:
            if ma_df["MA5"].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=ma_df.index,
                        y=ma_df["MA5"],
                        mode="lines",
                        name="MA5",
                        line=dict(color="#ffa500", width=1.5),
                        yaxis="y",
                    )
                )
            if ma_df["MA10"].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=ma_df.index,
                        y=ma_df["MA10"],
                        mode="lines",
                        name="MA10",
                        line=dict(color="#2ca02c", width=1.3),
                        yaxis="y",
                    )
                )
            if ma_df["MA20"].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=ma_df.index,
                        y=ma_df["MA20"],
                        mode="lines",
                        name="MA20",
                        line=dict(color="#9467bd", width=1.3),
                        yaxis="y",
                    )
                )

    fig.update_layout(
        height=560,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
    )

    fig_json = fig.to_json()

    # 原生 Plotly.js + JS 控制畫線 / Undo / Redo
    html_code = f"""
<div id="plot" style="width: 100%; height: 560px;"></div>
<div style="margin-top: 8px;">
  <button id="undoBtn">↩ Undo</button>
  <button id="redoBtn">↪ Redo</button>
  <button id="clearBtn">🧹 Clear</button>
</div>

<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<script>
  const fig = {fig_json};
  const gd = document.getElementById('plot');

  const config = {{
    editable: true,
    displaylogo: false,
    modeBarButtonsToAdd: [
      {{
        name: '黑線',
        icon: Plotly.Icons.pencil,
        click: function(gd) {{
          Plotly.relayout(gd, {{
            'newshape.line.color': 'black',
            'newshape.line.width': 2
          }});
        }}
      }},
      {{
        name: '紅線',
        icon: Plotly.Icons.pencil,
        click: function(gd) {{
          Plotly.relayout(gd, {{
            'newshape.line.color': 'red',
            'newshape.line.width': 2
          }});
        }}
      }},
      {{
        name: '藍線',
        icon: Plotly.Icons.pencil,
        click: function(gd) {{
          Plotly.relayout(gd, {{
            'newshape.line.color': 'blue',
            'newshape.line.width': 2
          }});
        }}
      }},
      {{
        name: '綠線',
        icon: Plotly.Icons.pencil,
        click: function(gd) {{
          Plotly.relayout(gd, {{
            'newshape.line.color': 'green',
            'newshape.line.width': 2
          }});
        }}
      }},
      'drawline',
      'drawopenpath',
      'eraseshape'
    ]
  }};

  Plotly.newPlot(gd, fig.data, fig.layout, config);

  // ====== 多步 Undo / Redo / Clear ======
  let shapesHistory = [];
  let currentIndex = -1;

  function getCurrentShapes() {{
    return gd.layout.shapes || [];
  }}

  function applyShapesFromHistory() {{
    if (currentIndex >= 0 && currentIndex < shapesHistory.length) {{
      const shapes = JSON.parse(shapesHistory[currentIndex]);
      Plotly.relayout(gd, {{shapes: shapes}});
    }}
  }}

  function saveState() {{
    const shapes = getCurrentShapes();
    const s = JSON.stringify(shapes);
    if (shapesHistory.length === 0 || shapesHistory[shapesHistory.length - 1] !== s) {{
      if (currentIndex < shapesHistory.length - 1) {{
        shapesHistory = shapesHistory.slice(0, currentIndex + 1);
      }}
      shapesHistory.push(s);
      currentIndex = shapesHistory.length - 1;
    }}
  }}

  // 初始狀態
  saveState();

  gd.on('plotly_relayout', function(ev) {{
    const keys = Object.keys(ev);
    const touchedShape = keys.some(k => k === 'shapes' || k.startsWith('shapes['));
    if (touchedShape) {{
      saveState();
    }}
  }});

  document.getElementById('undoBtn').onclick = function() {{
    if (currentIndex > 0) {{
      currentIndex -= 1;
      applyShapesFromHistory();
    }}
  }};

  document.getElementById('redoBtn').onclick = function() {{
    if (currentIndex < shapesHistory.length - 1) {{
      currentIndex += 1;
      applyShapesFromHistory();
    }}
  }};

  document.getElementById('clearBtn').onclick = function() {{
    saveState();
    Plotly.relayout(gd, {{shapes: []}});
    saveState();
  }};
</script>
"""
    components.html(html_code, height=620)


# ========= 主程式 =========
def main():
    st.title("📈 美股智慧 AI 分析")
    st.caption("輸入股票代號（例如：AAPL、TSLA、NVDA）")

    col_input, col_period = st.columns([3, 1])
    with col_input:
        symbol = st.text_input(
            "股票代號（Ticker）",
            value=st.session_state["last_symbol"] or "AAPL",
        )
    with col_period:
        period = st.selectbox(
            "資料區間",
            ["3mo", "6mo", "1y"],
            index=["3mo", "6mo", "1y"].index(st.session_state["last_period"]),
        )

    analyze_clicked = st.button("開始分析", type="primary")

    if analyze_clicked:
        clean_symbol = symbol.strip().upper()
        if clean_symbol:
            st.session_state["analyzed"] = True
            st.session_state["last_symbol"] = clean_symbol
            st.session_state["last_period"] = period

    if st.session_state["analyzed"] and st.session_state["last_symbol"]:
        clean_symbol = st.session_state["last_symbol"]
        period = st.session_state["last_period"]

        try:
            with st.spinner(f"正在載入 {clean_symbol} 資料…"):
                data = fetch_us_stock(clean_symbol, period)
                hist = data["price_history"]
                indicators = compute_indicators(hist, data["fundamentals_raw"])
                financials = fetch_financial_statements(clean_symbol)
                earnings = fetch_earnings_summary(clean_symbol)

            basic = data["basic_info"]
            display_name = (
                basic.get("longName")
                or basic.get("shortName")
                or clean_symbol
            )

            left, right = st.columns([2.2, 1.8])

            # ================= 左邊：即時價 + MA/Volume + 基本 + 圖 + 指標 + 財報 =================
            with left:
                # 即時價區
                st.subheader("⏱ 近一小時 / 最近收盤價")
                rt = fetch_last_1h_price(clean_symbol)
                if rt is not None:
                    c1, c2 = st.columns(2)
                    label_price = (
                        "最新價（近一小時）" if rt["source"] == "intraday" else "最近收盤價"
                    )
                    with c1:
                        st.metric(
                            label_price,
                            f"{rt['last']:.2f}",
                            f"{rt['change']:+.2f}",
                        )
                    with c2:
                        st.metric(
                            "相對變化",
                            f"{rt['pct'] * 100:+.2f} %",
                        )
                else:
                    st.info("目前無法取得近一小時或收盤價（資料來源限制）。")

                # 最近一個交易日 MA / Volume
                st.subheader("📆 最近一個交易日：均線 / 成交量")
                ma_info = fetch_last_daily_ma_volume(clean_symbol)
                if ma_info is not None:
                    ma_table = pd.DataFrame(
                        {
                            "項目": ["日期", "MA5", "MA10", "MA20", "成交量"],
                            "數值": [
                                ma_info["date"],
                                f"{ma_info['ma5']:.2f}" if ma_info["ma5"] is not None else "N/A",
                                f"{ma_info['ma10']:.2f}" if ma_info["ma10"] is not None else "N/A",
                                f"{ma_info['ma20']:.2f}" if ma_info["ma20"] is not None else "N/A",
                                f"{ma_info['volume']:.0f}" if ma_info["volume"] is not None else "N/A",
                            ],
                        }
                    )
                    st.table(ma_table)
                else:
                    st.info("無法取得最近交易日的均線與成交量資訊。")

                st.subheader("📌 基本資訊")
                st.write(f"**{display_name} ({clean_symbol})**")
                st.write(
                    f"{basic.get('sector')} / {basic.get('industry')} | "
                    f"{basic.get('country')} | 貨幣：{basic.get('currency')}"
                )

                # 專業版圖表
                render_pro_chart(hist, period)

                # 指標摘要
                st.subheader("📊 指標摘要")
                val = indicators["valuation"]
                mom = indicators["momentum"]

                def pct(x):
                    return f"{x*100:.2f}%" if x is not None else "N/A"

                table = pd.DataFrame(
                    {
                        "指標": [
                            "現價",
                            "本益比 (Trailing PE)",
                            "預估本益比 (Forward PE)",
                            "股價淨值比 (P/B)",
                            "1M 報酬率",
                            "3M 報酬率",
                            "3M 波動度",
                            "3M 高點",
                            "3M 低點",
                        ],
                        "數值": [
                            val.get("latestPrice"),
                            val.get("trailingPE"),
                            val.get("forwardPE"),
                            val.get("priceToBook"),
                            pct(mom.get("oneMonthReturn")),
                            pct(mom.get("threeMonthReturn")),
                            pct(mom.get("volatility3m")),
                            mom.get("high3m"),
                            mom.get("low3m"),
                        ],
                    }
                )
                st.table(table)

                # 財報
                st.subheader("📑 最近四季損益表")
                if (
                    financials
                    and "income_q" in financials
                    and financials["income_q"] is not None
                    and not financials["income_q"].empty
                ):
                    st.dataframe(financials["income_q"])
                else:
                    st.info("找不到損益資料")

            # ================= 右邊：AI 分析 =================
            with right:
                st.subheader("🤖 AI 數據分析")

                # 這裡主分析會特別強調目前選的 period
                main_question = (
                    f"請針對目前取得的股價與基本面數據，"
                    f"特別聚焦在顯示的時間區間「{period}」做一份完整分析。"
                    "說明該期間內股價走勢、估值位置（例如本益比在產業中的相對高低）、"
                    "以及此期間可以觀察到的亮點與潛在風險。"
                )
                summary = generate_analysis(
                    symbol=clean_symbol,
                    indicators=indicators,
                    price_history=hist,
                    user_question=main_question,
                )
                st.markdown(summary)

                st.markdown("---")
                st.subheader("📊 財報亮點 / 風險 / 展望")
                insight = extract_earnings_insights(
                    symbol=clean_symbol,
                    earnings_data=earnings,
                    financials=financials,
                )
                st.markdown(insight)

                st.markdown("---")
                st.markdown("### 追問 AI（可針對特定季度或期間）")
                q = st.text_input(
                    "想問什麼？（例：請分析 2025 年第一季的表現、這一年股價波動與估值是否合理…）"
                )
                if st.button("送出追問"):
                    follow_up_question = (
                        f"目前圖上顯示的時間區間為「{period}」。"
                        f"請在這段期間的背景下，結合先前提供的數據，"
                        f"回答以下追問，並盡量以該時間範圍內的變化為主：\n\n{q}"
                    )
                    ans = generate_analysis(
                        symbol=clean_symbol,
                        indicators=indicators,
                        price_history=hist,
                        user_question=follow_up_question,
                    )
                    st.markdown("#### AI 回覆")
                    st.markdown(ans)

            # ================= 最下方：任意文字檔摘要 + 翻譯 + 防呆檢查（支援 PDF） =================
            st.markdown("---")
            with st.expander("📄 文字檔摘要 / 翻譯（新聞、財報、法說會逐字稿｜支援 txt / md / pdf）"):
                st.caption(
                    "上傳與此公司相關的文字檔（PDF / TXT / MD），例如新聞、財報說明、法說會逐字稿等。"
                )

                uploaded = st.file_uploader(
                    "上傳文字檔（txt / md / pdf）",
                    type=["txt", "md", "pdf"],
                )
                manual = st.text_area("或直接貼上內容")

                text = ""

                # -------- PDF / txt / md 處理 --------
                if uploaded is not None:
                    if uploaded.type == "application/pdf":
                        try:
                            import pdfplumber
                            with pdfplumber.open(uploaded) as pdf:
                                pages = [page.extract_text() or "" for page in pdf.pages]
                                text = "\n".join(pages)
                        except Exception as e:
                            st.error(f"PDF 解析失敗：{e}")
                            text = ""
                    else:
                        # txt/md
                        text = uploaded.read().decode("utf-8", "ignore")

                elif manual.strip():
                    text = manual.strip()

                # -------- 有文本才進行後續處理 --------
                if text:
                    if st.button("開始分析文字檔"):
                        # ---- 防呆：檢查是否真的像是這家公司的內容 ----
                        lower_text = text.lower()
                        keywords = set()
                        keywords.add(clean_symbol.lower())

                        dn = display_name.lower()
                        keywords.add(dn)
                        for tok in dn.replace(",", " ").split():
                            tok = tok.strip()
                            if len(tok) > 2:
                                keywords.add(tok)

                        matched = any(k in lower_text for k in keywords)

                        if not matched:
                            st.error(
                                f"這份文字檔看起來不像是關於 {display_name} ({clean_symbol}) 的內容，"
                                "請確認是否上傳錯誤公司。"
                            )
                        else:
                            with st.spinner("AI 正在進行翻譯與摘要…"):
                                paragraphs = translate_transcript_paragraphs(text)
                                transcript_summary = analyze_earnings_transcript(
                                    clean_symbol, text
                                )

                            st.subheader("逐段翻譯")
                            for en, zh in paragraphs:
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.markdown(en)
                                with c2:
                                    st.markdown(zh)

                            st.subheader("文字重點摘要 / 分析")
                            st.markdown(transcript_summary)

        except Exception as e:
            st.error(f"發生錯誤：{e}")
    else:
        st.info("請先輸入股票代號並按下「開始分析」。")


if __name__ == "__main__":
    main()
