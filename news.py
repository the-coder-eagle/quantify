"""
News scraper + sentiment for A-share stocks.

Uses akshare Eastmoney news API.  The keyword-based sentiment is a crude
fallback — prefer AI sentiment for strategy signals.  This module is for
display/conversational use only; do NOT feed keyword sentiment into
trading decisions.
"""
from loguru import logger

POS = ["利好","增长","盈利","突破","上涨","涨停","增持","买入","评级","超预期","分红","回购","签约","中标","获批"]
NEG = ["利空","下跌","亏损","跌停","减持","卖出","下调","调查","处罚","违规","退市","爆雷","造假","债务","违约"]


def fetch(code, max_items=10):
    """Fetch recent news for a specific stock from Eastmoney.

    Args:
        code: 6-digit A-share stock code (e.g. "600519")
        max_items: max news items to return

    Returns:
        list of dicts with 'title' and 'time' keys
    """
    code = str(code).zfill(6)
    try:
        import akshare as ak
        # Primary: stock-specific news
        df = ak.stock_news_em(symbol=code)
        if df is not None and not df.empty:
            news = []
            for _, row in df.head(max_items).iterrows():
                news.append({
                    "title": str(row.iloc[0])[:120],
                    "time": str(row.iloc[-1]) if len(row) > 1 else "",
                })
            if news:
                return news
    except Exception as e:
        logger.warning(f"News fetch failed for {code}: {e}")

    # Fallback: filter general market news by code mention
    try:
        import akshare as ak
        df = ak.stock_news_em()
        if df is not None and not df.empty:
            news = []
            for _, row in df.iterrows():
                title = str(row.iloc[0])
                if code in title or (len(row) > 1 and code in str(row.iloc[-1])):
                    news.append({
                        "title": title[:120],
                        "time": str(row.iloc[-1]) if len(row) > 1 else "",
                    })
                    if len(news) >= max_items:
                        break
            return news
    except Exception as e:
        logger.warning(f"News fallback failed for {code}: {e}")

    return []


def sentiment(news_list, ai_client=None):
    """Analyze sentiment of news titles.

    NOTE: Keyword-based sentiment is LOW CONFIDENCE and should NOT be
    used for strategy signals.  It is here for display purposes only.
    Use AI sentiment (via ai_client) for any signal-quality analysis.
    """
    if not news_list:
        return {"score": 0, "confidence": 0, "summary": "无新闻", "method": "none"}

    combined = " ".join(n.get("title", "") for n in news_list)

    # Try AI first
    if ai_client:
        try:
            resp = ai_client.ask_json(
                "分析股票新闻情绪，返回JSON: {\"score\":-1到1,\"confidence\":0到1,\"summary\":\"一句话\"}",
                combined,
            )
            if "score" in resp:
                return {**resp, "method": "ai"}
        except Exception as e:
            logger.debug(f"AI sentiment failed, falling back to keyword: {e}")

    # Keyword fallback — crude, for display only
    pos_cnt = sum(1 for w in POS if w in combined)
    neg_cnt = sum(1 for w in NEG if w in combined)
    total = pos_cnt + neg_cnt
    if total == 0:
        return {"score": 0, "confidence": 0, "summary": "中性", "method": "keyword"}

    score = (pos_cnt - neg_cnt) / total
    return {
        "score": round(score, 2),
        "confidence": round(min(total / 5, 0.5), 2),  # capped at 0.5 — low confidence
        "summary": f"{pos_cnt}利好/{neg_cnt}利空",
        "method": "keyword",
    }
