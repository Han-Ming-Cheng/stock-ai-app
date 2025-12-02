📈 Stock AI App

使用網址: https://stock-ai-app-c6evappbapgfta6oozgqfx.streamlit.app/

基於 Groq Llama 3.1 的智慧股票分析工具，整合「即時股價」、「K 線圖」、「移動平均線」、「成交量」、「AI 財報分析」、「文件（PDF/TXT）摘要與翻譯」等功能。

🚀 主要功能
📌 1. 即時 & 歷史股價資訊

顯示最新股價（或最近收盤價）

基本資訊：產業、國家、貨幣、公司名稱

技術指標：

MA5 / MA10 / MA20

成交量 Volume

📊 2. 互動式圖表（含 K 線 + 自由畫線工具）

折線圖（Line Chart）

K 線圖（Candlestick）

漲 → 綠色

跌 → 紅色

MA 線可一鍵切換顯示 / 隱藏

自由畫線工具：

支援顏色：黑 / 紅 / 藍 / 綠

支援「Undo / Redo」不限次數

可用於自訂技術分析標記

🧠 3. AI 財務與股價分析（Groq · Llama 3.1）

AI 可生成：

公司與產業介紹

PE / PB / Forward PE 估值分析

股價動能與波動度分析

財報亮點（Strengths）

風險（Risks）

未來展望（Outlook）

支援使用者追問：

「請分析 2025 年第一季。」
「告訴我最近一年股價波動的原因。」
「這家公司跟 NVDA 相比估值如何？」

📑 4. PDF / TXT 文件上傳解析

支援上傳：法說會逐字稿、新聞文章、財報重點內容、分析師研究報告

AI 將自動：

萃取文字

判斷是否為該公司（防呆檢查）

逐段翻譯（中 / 英）

摘要亮點 / 風險 / 展望

若文件不是該公司 → 顯示錯誤訊息。

📂 專案結構

stock-ai-app/
│── app.py
│── requirements.txt
│
└── core/
    │── __init__.py
    │── data_fetch.py
    │── indicators.py
    │── ai_analyzer.py


⚠ 免責聲明

本工具提供的內容僅供教育與研究用途。
所有資料與分析皆非投資建議，請自行斟酌投資風險。
