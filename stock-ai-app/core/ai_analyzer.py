# core/ai_analyzer.py
# -------------------------------------------------
# 使用 Google Gemini 作為 LLM：
# - 讀取環境變數 GOOGLE_API_KEY
# - 有 key → 用 Gemini 回覆
# - 沒 key 或出錯 → fallback 規則版
# - 保留 Question Guard（review_question）
# -------------------------------------------------

from __future__ import annotations

import os
import re
import json
import textwrap
from typing import Dict, Any, List, Tuple, Optional

# =============== Google Gemini SDK ===============
# pip install google-generativeai
import google.generativeai as genai

GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)


# ================= 工具：清理文字 =================
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========== 內部：呼叫 Gemini 模型 ===========
def _call_llm(model: str, system_prompt: str, user_prompt: str) -> str:
    """
    共用 LLM 呼叫：
    - model: "gemini-2.0-flash" 或 "gemini-1.5-pro"
    - system_prompt: 系統角色
    - user_prompt: 使用者問題 + 數據
    """
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not found")

    gm = genai.GenerativeModel(model)

    # Gemini 不區分 system/user，我們直接組一個完整 prompt
    prompt = system_prompt.strip() + "\n\n=== 使用者輸入 ===\n" + user_prompt.strip()

    resp = gm.generate_content(prompt)
    # resp.text 是整段生成內容
    return (resp.text or "").strip()


# ========= 規則版分析（fallback 用） =========
def _rule_based_stock_analysis(
    symbol: str,
    indicators: dict,
    user_question: str | None,
) -> str:
    val = indicators.get("valuation", {})
    mom = indicators.get("momentum", {})

    latest_price = val.get("latestPrice")
    pe = val.get("trailingPE")
    fpe = val.get("forwardPE")

    one_m = mom.get("oneMonthReturn")
    three_m = mom.get("threeMonthReturn")

    def fmt_pct(x):
        if x is None:
            return "N/A"
        return f"{x * 100:.2f}%"

    trend = "-"
    if isinstance(three_m, (int, float)):
        if three_m > 0.05:
            trend = "明顯上升"
        elif three_m < -0.05:
            trend = "明顯下跌"
        else:
            trend = "相對震盪整理"

    question_part = f"\n\n你問的問題：「{user_question}」\n" if user_question else ""

    txt = f"""
    ## 📈 股票分析（規則版，未啟用 Gemini）

    **股票：{symbol}**

    ### 🔹 1. 基本估值
    - 現價：{latest_price}
    - 本益比（PE）：{pe}
    - 預估本益比（Forward PE）：{fpe}

    ### 🔹 2. 股價動能
    - 1 個月報酬：{fmt_pct(one_m)}（短期情緒參考）
    - 3 個月報酬：{fmt_pct(three_m)} → **{trend}**

    ### 🔹 3. 亮點（根據簡單規則推斷）
    - 3M 上漲視為多方氣氛較強。
    - Forward PE 若低於 PE，代表市場對未來成長有期待。

    ### 🔹 4. 風險
    - 若 3M 報酬率為負，須注意可能的下跌趨勢。
    - 若 PE 遠高於產業平均，可能有估值過高風險。

    ### 🔹 5. 說明
    ⚠ 目前尚未啟用 Google Gemini 模型，
    因此本分析為「規則 + 模板」自動生成。

    {question_part}
    """
    return textwrap.dedent(txt)


def _rule_based_earnings(symbol: str) -> str:
    txt = f"""
    ## 📝 財報亮點摘要（規則版）

    股票：{symbol}

    ### 🔹 可能的亮點
    - 最近季度營收高於前季，通常被視為正向訊號。
    - 毛利率提升代表成本控制較佳。

    ### 🔹 潛在風險
    - 淨利較上季下滑時，需留意獲利穩定度。
    - 若營業活動現金流連續下滑，可能埋有財務壓力。

    ### 🔹 提醒
    ⚠ 本段仍為規則運算，並非真正 LLM 解析逐字稿與財報。
    """
    return textwrap.dedent(txt)


# =============== 問題審查器（Question Guard） ===============
_FIN_KW_ZH = [
    "營收",
    "獲利",
    "毛利",
    "淨利",
    "成長",
    "估值",
    "本益比",
    "股價",
    "股息",
    "配息",
    "現金流",
    "財報",
    "季度",
    "展望",
    "風險",
]

_FIN_KW_EN = [
    "revenue",
    "profit",
    "margin",
    "guidance",
    "valuation",
    "dividend",
    "eps",
    "cash flow",
    "earnings",
    "quarter",
    "risk",
    "growth",
]

_YEAR_RE = re.compile(r"(19|20)\d{2}")


def review_question(
    question: str,
    symbol: str,
    price_history=None,
    financials=None,
) -> Dict[str, Any]:
    """
    回傳：
    {
      "level": "ok" | "warn" | "reject",
      "reason": "...",
      "message": "要顯示給使用者看的文字",
      "system_hint": "要塞進 system prompt 的補充說明（可為空字串）"
    }
    """
    q = clean_text(question)
    if not q:
        return {
            "level": "reject",
            "reason": "empty",
            "message": "❌ 問題內容是空的，請具體輸入想分析的重點或疑問。",
            "system_hint": "",
        }

    # 1) 太短直接拒絕
    if len(q) <= 3:
        return {
            "level": "reject",
            "reason": "too_short",
            "message": "❌ 問題太短了，請再具體一些（例如：想看哪一段期間、估值、財報或風險？）。",
            "system_hint": "",
        }

    # 2) 明顯亂打（大量標點 / 符號）
    alpha_num_zh = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", q)
    if len(alpha_num_zh) / len(q) < 0.35:
        return {
            "level": "reject",
            "reason": "gibberish",
            "message": "❌ 這個問題看起來像是隨機字元或無法判讀的內容，請重新敘述你的問題。",
            "system_hint": "",
        }

    # 3) 關鍵字檢查（沒有財經關鍵字 → warn 但允許）
    has_fin_kw = any(kw in q for kw in _FIN_KW_ZH) or any(
        kw in q.lower() for kw in _FIN_KW_EN
    )

    warn_msgs: List[str] = []
    system_hints: List[str] = []

    if not has_fin_kw:
        warn_msgs.append(
            "⚠ 這個問題沒有明顯的財經 / 股價 / 財報關鍵字，我會盡量從一般角度回答，"
            "但也可能提醒你這個工具主要是用來做股票與財報分析。"
        )
        system_hints.append(
            "若使用者提問與股票 / 財報 /金融無直接關聯，請先說明本工具主要用途，"
            "再視情況簡要回答；若完全無關，建議禮貌回覆無法回答。"
        )

    # 4) 年份範圍檢查（從問題抓出年份）
    years_in_q = [int(y) for y in _YEAR_RE.findall(q)] if _YEAR_RE.findall(q) else []

    data_years: List[int] = []
    if price_history is not None and hasattr(price_history, "index"):
        try:
            for idx in price_history.index:
                y = getattr(idx, "year", None)
                if y:
                    data_years.append(int(y))
        except Exception:
            pass

    if financials and isinstance(financials, dict):
        inc = financials.get("income_q")
        if inc is not None and not inc.empty and "period" in inc.columns:
            for p in inc["period"]:
                try:
                    y = getattr(p, "year", None)
                    if y:
                        data_years.append(int(y))
                except Exception:
                    try:
                        m = _YEAR_RE.search(str(p))
                        if m:
                            data_years.append(int(m.group()))
                    except Exception:
                        pass

    if data_years and years_in_q:
        min_y, max_y = min(data_years), max(data_years)
        out_of_range = [y for y in years_in_q if y < min_y or y > max_y]
        if out_of_range:
            warn_msgs.append(
                f"⚠ 問題中提到的年份 {sorted(set(out_of_range))} 超出目前資料範圍 "
                f"（約 {min_y} ~ {max_y}），回答時會盡量以可取得的年份說明，並提醒這一點。"
            )
            system_hints.append(
                "使用者問題涉及資料範圍以外的年份時，請先明確說明資料僅涵蓋的區間，"
                "再依現有資料做推論；對於沒有資料的年份，不要虛構具體數字或事件。"
            )

    if not warn_msgs:
        return {
            "level": "ok",
            "reason": "pass",
            "message": "",
            "system_hint": "",
        }

    return {
        "level": "warn",
        "reason": "warn",
        "message": "\n\n".join(warn_msgs),
        "system_hint": "\n".join(system_hints),
    }


# ============ Gemini 版：主分析（AI 數據分析） ============
def generate_analysis(
    symbol: str,
    indicators: dict,
    price_history,
    user_question: str | None = None,
    model: str | None = None,
    guard_hint: str | None = None,
) -> str:
    """
    若有 GOOGLE_API_KEY 且指定 model → 呼叫 Gemini
    否則自動 fallback 規則版。
    """
    if (model is None) or (not GOOGLE_API_KEY):
        return _rule_based_stock_analysis(symbol, indicators, user_question)

    val = indicators.get("valuation", {})
    mom = indicators.get("momentum", {})

    def safe_float(x):
        try:
            return float(x)
        except Exception:
            return None

    payload = {
        "symbol": symbol,
        "valuation": {
            "latestPrice": safe_float(val.get("latestPrice")),
            "trailingPE": safe_float(val.get("trailingPE")),
            "forwardPE": safe_float(val.get("forwardPE")),
            "priceToBook": safe_float(val.get("priceToBook")),
        },
        "momentum": {
            "oneMonthReturn": safe_float(mom.get("oneMonthReturn")),
            "threeMonthReturn": safe_float(mom.get("threeMonthReturn")),
            "volatility3m": safe_float(mom.get("volatility3m")),
            "high3m": safe_float(mom.get("high3m")),
            "low3m": safe_float(mom.get("low3m")),
        },
    }

    system_prompt = textwrap.dedent(
        f"""
        你是一位專門分析「美股」與「財報」的專業投資顧問，回答時請使用**繁體中文**，
        風格清楚、有條理、但不要過度艱深。

        核心原則：
        1. 僅能根據使用者提供的指標資料與一般常識進行推理，不可捏造具體數字、年份或事件。
        2. 若無法從資料中合理推論答案，要明確說「目前資料無法判斷」或「缺乏足夠資料」。
        3. 若問題與股票 / 財報 / 投資風險無關，先說本工具的用途，再視情況簡要回答或婉拒。
        4. 盡量給出「亮點」、「風險」、「需要關注的指標」三個層次的說明。
        5. 若有額外的 guard 說明，必須一併遵守。

        {guard_hint or ""}
        """
    )

    user_prompt = textwrap.dedent(
        f"""
        以下是關於股票 {symbol} 的指標資料（JSON）：

        {json.dumps(payload, ensure_ascii=False, indent=2, default=str)}

        請根據這些資料，給出一份結構化的分析報告，格式包含：
        1. 估值概況（本益比、股價淨值比等，大致是偏貴、偏便宜、還是合理區間）
        2. 近期股價動能（1M / 3M 報酬率、波動度與高低點的解讀）
        3. 亮點（列出 2–4 點）
        4. 風險與需要特別留意的項目（列出 2–4 點）
        5. 給一般投資人的提醒（不要當作投資建議）

        使用者目前的提問是：
        {user_question or "「沒有額外提問，只是想看這檔股票在目前區間的綜合分析。」"}
        """
    )

    try:
        return _call_llm(model=model, system_prompt=system_prompt, user_prompt=user_prompt)
    except Exception:
        return _rule_based_stock_analysis(symbol, indicators, user_question)


# ============ Gemini 版：財報亮點 ============
def extract_earnings_insights(
    symbol: str,
    earnings_data,
    financials,
    model: str | None = None,
) -> str:
    """
    傳入 yfinance earnings / financials，請 LLM 幫忙整理財報亮點。
    沒有模型或金鑰時就用規則版。
    """
    if (model is None) or (not GOOGLE_API_KEY):
        return _rule_based_earnings(symbol)

    income_q = None
    if financials and isinstance(financials, dict):
        income_q = financials.get("income_q")

    income_json = None
    if income_q is not None and not income_q.empty:
        try:
            income_json = income_q.head(4).to_dict(orient="records")
        except Exception:
            income_json = None

    data_payload = {
        "symbol": symbol,
        "earnings_table": getattr(earnings_data, "to_dict", lambda **k: None)(
            orient="index"
        )
        if hasattr(earnings_data, "to_dict")
        else None,
        "income_q": income_json,
    }

    system_prompt = textwrap.dedent(
        """
        你是一位專門閱讀美股財報與法說會資訊的分析師，回答以繁體中文。
        目標是從有限的 earnings / 損益表資訊中，整理出：
        1. 最近幾季的營收與獲利趨勢（成長或衰退、是否穩定）
        2. 毛利率 / 營業利益率是否改善或惡化（若有資料）
        3. 管理階層可能關注的重點與風險（根據數字合理推論）
        4. 給一般投資人的提醒（不是投資建議）

        若發現資料極少或欄位不足，請明確說明限制，不要胡亂猜測。
        """
    )

    user_prompt = textwrap.dedent(
        f"""
        下面是股票 {symbol} 最近的部分財報數據（可能不完整）：

        {json.dumps(data_payload, ensure_ascii=False, indent=2, default=str)}

        請整理成一段易讀的「財報亮點 / 風險 / 展望」說明，條列重點。
        """
    )

    try:
        return _call_llm(model=model, system_prompt=system_prompt, user_prompt=user_prompt)
    except Exception:
        return _rule_based_earnings(symbol)


# ============ Gemini 版：逐段翻譯 ============
def translate_transcript_paragraphs(
    text: str,
    model: str | None = None,
) -> List[Tuple[str, str]]:
    """
    將逐字稿切段 → 每段英文 → 中文翻譯。
    若無模型就用「假翻譯」。
    """
    text = text.replace("\r", "\n")
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    results: List[Tuple[str, str]] = []

    if (model is None) or (not GOOGLE_API_KEY):
        for p in parts:
            zh = f"（此段落的中文摘要示意）{p[:40]}..."
            results.append((p, zh))
        return results

    system_prompt = textwrap.dedent(
        """
        你是一位精通英文與繁體中文的財報口譯人員。
        對於每一段英文逐字稿，請給出：
        - 精準、自然的繁體中文翻譯（不要加自己評論）
        """
    )

    for p in parts:
        try:
            zh = _call_llm(
                model=model,
                system_prompt=system_prompt,
                user_prompt=p,
            )
        except Exception:
            zh = f"（翻譯失敗，以下為原文前 40 字）{p[:40]}..."
        results.append((p, zh))

    return results


# ============ Gemini 版：整份文字摘要 ============
def analyze_earnings_transcript(
    symbol: str,
    text: str,
    model: str | None = None,
) -> str:
    """
    針對整份文字檔做摘要。若沒有模型則給規則版摘要。
    """
    clean = clean_text(text)

    if (model is None) or (not GOOGLE_API_KEY):
        word_count = len(clean.split())
        key_terms = ["guidance", "revenue", "margin", "profit"]
        found = [k for k in key_terms if k in clean.lower()]

        txt = f"""
        ## 📘 文字摘要（規則版）

        - 文字長度：約 {word_count} 個英文單字或詞。
        - 偵測到的財務關鍵字：{', '.join(found) if found else '無明顯關鍵字'}

        ⚠ 未啟用 LLM，因此僅能提供非常粗略的資訊。
        """
        return textwrap.dedent(txt)

    system_prompt = textwrap.dedent(
        f"""
        你是一位專門閱讀財報逐字稿與財經新聞的分析師，請使用繁體中文回答。

        目標：針對股票 {symbol} 的這份文字內容，整理出：
        1. 主題與背景是什麼（1 段話）
        2. 正面亮點（2–5 點）
        3. 潛在風險或市場擔憂（2–5 點）
        4. 管理階層對未來的展望或指引（若有）
        5. 對一般投資人的提醒：僅作資訊參考，不是投資建議。

        嚴禁捏造不存在的具體數字；若原文沒有寫，就以「原文未明確提到」表達。
        """
    )

    user_prompt = clean[:15000]  # 避免 prompt 過長

    try:
        return _call_llm(model=model, system_prompt=system_prompt, user_prompt=user_prompt)
    except Exception:
        word_count = len(clean.split())
        txt = f"""
        ## 📘 文字摘要（規則版）

        - 文字長度：約 {word_count} 個英文單字或詞。
        - 由於 LLM 呼叫失敗，僅能給出長度資訊，無法產生完整內容摘要。
        """
        return textwrap.dedent(txt)
