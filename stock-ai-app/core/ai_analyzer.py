from groq import Groq

client = Groq()


# ============================================================
# 1) 基本 AI 數據解讀
# ============================================================
def generate_analysis(symbol, indicators, price_history, user_question=None):
    system_prompt = """
你是一位嚴謹、保守的美股財務分析助理。
你只能基於提供的資料進行解讀，不可提供買賣建議。
請用繁體中文回答。
"""

    summary = f"""
股票代號：{symbol}
指標資料：{indicators}
"""

    if user_question is None:
        user_question = "請根據以上數據做摘要分析。"

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": summary + "\n\n問題：" + user_question},
        ],
        temperature=0.2,
    )

    return res.choices[0].message.content.strip()


# ============================================================
# 2) 財報亮點／風險／展望
# ============================================================
def extract_earnings_insights(symbol, earnings_data, financials):
    question = f"""
請整理以下財報資訊，輸出：
1. 財報亮點（3–5 點）
2. 潛在風險（3–5 點）
3. 管理層展望（若有）
4. 最後加一句：此內容僅為資料解讀，非投資建議。

Earnings:
{earnings_data}

Financials:
{financials}
"""

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": question}],
        temperature=0.2,
    )

    return res.choices[0].message.content.strip()


# ============================================================
# 3) 逐字稿逐段翻譯
# ============================================================
def translate_transcript_paragraphs(text: str):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    results = []

    for p in paragraphs:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": f"翻譯成繁體中文：\n{p}"}
            ],
            temperature=0.2,
        )
        zh = res.choices[0].message.content.strip()
        results.append((p, zh))

    return results


# ============================================================
# 4) 法說會逐字稿摘要
# ============================================================
def analyze_earnings_transcript(symbol, transcript_text):
    question = f"""
請根據以下法說會逐字稿整理：
- 3–5 個重點亮點
- 3–5 個風險
- 管理層展望
- 最後補一句「以上僅為資料解讀，非投資建議。」

逐字稿全文：
{transcript_text}
"""

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": question}],
        temperature=0.2,
    )
    return res.choices[0].message.content.strip()
