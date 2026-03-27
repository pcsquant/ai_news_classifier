import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import requests
import streamlit as st
from openai import OpenAI

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"


def get_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def safe_get(d: Dict[str, Any], key: str, default: Any = "N/A") -> Any:
    value = d.get(key, default)
    if value in (None, "", []):
        return default
    return value


def fmt_unix(ts: int | None) -> str:
    if not ts:
        return "N/A"
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return "N/A"


def date_str(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%d")


@st.cache_data(ttl=300, show_spinner=False)
def finnhub_get(endpoint: str, params: Dict[str, Any]) -> Any:
    api_key = get_env("FINNHUB_API_KEY")
    url = f"{FINNHUB_BASE_URL}/{endpoint}"
    final_params = dict(params)
    final_params["token"] = api_key

    response = requests.get(url, params=final_params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if isinstance(data, dict) and data.get("error"):
        raise ValueError(data["error"])
    return data


def get_company_profile(symbol: str) -> Dict[str, Any]:
    return finnhub_get("company-profile2", {"symbol": symbol.upper()})


def get_quote(symbol: str) -> Dict[str, Any]:
    return finnhub_get("quote", {"symbol": symbol.upper()})


def get_company_news(symbol: str, days_back: int = 7) -> List[Dict[str, Any]]:
    return finnhub_get(
        "company-news",
        {
            "symbol": symbol.upper(),
            "from": date_str(days_back),
            "to": date_str(0),
        },
    )


def profile_text(profile: Dict[str, Any], quote: Dict[str, Any]) -> str:
    return f"""
Company Name: {safe_get(profile, 'name')}
Ticker: {safe_get(profile, 'ticker')}
Exchange: {safe_get(profile, 'exchange')}
Industry: {safe_get(profile, 'finnhubIndustry')}
Country: {safe_get(profile, 'country')}
IPO Date: {safe_get(profile, 'ipo')}
Currency: {safe_get(profile, 'currency')}
Market Capitalization: {safe_get(profile, 'marketCapitalization')}
Shares Outstanding: {safe_get(profile, 'shareOutstanding')}
Website: {safe_get(profile, 'weburl')}

Latest Quote:
Current Price: {safe_get(quote, 'c')}
Change: {safe_get(quote, 'd')}
Percent Change: {safe_get(quote, 'dp')}
High: {safe_get(quote, 'h')}
Low: {safe_get(quote, 'l')}
Open: {safe_get(quote, 'o')}
Previous Close: {safe_get(quote, 'pc')}
Timestamp: {fmt_unix(quote.get('t'))}
""".strip()


def news_text(news_items: List[Dict[str, Any]], max_items: int = 10) -> str:
    if not news_items:
        return "No recent news found."

    lines = []
    for i, item in enumerate(news_items[:max_items], start=1):
        lines.append(
            f"""{i}. Headline: {safe_get(item, 'headline')}
Source: {safe_get(item, 'source')}
Time: {fmt_unix(item.get('datetime'))}
Summary: {safe_get(item, 'summary')}
URL: {safe_get(item, 'url')}"""
        )
    return "\n\n".join(lines)


def ask_llm(symbol: str, question: str, profile: Dict[str, Any], quote: Dict[str, Any], news: List[Dict[str, Any]]) -> str:
    llm_api_key = get_env("LLM_API_KEY")
    model = os.getenv("LLM_MODEL", "gpt-5.4-mini")

    client = OpenAI(api_key=llm_api_key)

    system_prompt = f"""
You are a stock research assistant for an internal team.

You must answer using the supplied stock profile, latest quote, and recent company news.
Be clear, balanced, and practical.

Rules:
1. Separate facts from interpretation.
2. Do not invent missing information.
3. Do not guarantee future stock performance.
4. If asked buy/sell questions, provide a balanced view and mention risks.
5. Use concise headings and bullet points when useful.

Stock symbol: {symbol.upper()}

COMPANY PROFILE
{profile_text(profile, quote)}

RECENT NEWS
{news_text(news)}
""".strip()

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": question}],
            },
        ],
    )
    return response.output_text


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "loaded_symbol" not in st.session_state:
        st.session_state.loaded_symbol = ""
    if "profile" not in st.session_state:
        st.session_state.profile = None
    if "quote" not in st.session_state:
        st.session_state.quote = None
    if "news" not in st.session_state:
        st.session_state.news = []


st.set_page_config(page_title="Stock Chatbot", page_icon="📈", layout="wide")
init_state()

st.title("📈 Stock News Chatbot")
st.caption("Internal stock analysis app powered by Finnhub + LLM")

with st.sidebar:
    st.header("Settings")
    symbol = st.text_input("Stock Symbol", value=st.session_state.loaded_symbol or "AAPL").upper().strip()
    news_days = st.slider("News Lookback Days", min_value=1, max_value=30, value=7)
    if st.button("Load Stock Data", use_container_width=True):
        try:
            st.session_state.profile = get_company_profile(symbol)
            st.session_state.quote = get_quote(symbol)
            st.session_state.news = get_company_news(symbol, days_back=news_days)
            st.session_state.loaded_symbol = symbol
            st.session_state.messages = []
            st.success(f"Loaded {symbol}")
        except Exception as e:
            st.error(f"Could not load data: {e}")

    st.markdown("---")
    st.markdown("**Required env vars on Render**")
    st.code("FINNHUB_API_KEY\nLLM_API_KEY\nLLM_MODEL", language="bash")

left, right = st.columns([1, 1])

with left:
    st.subheader("Company Profile")
    if st.session_state.profile:
        p = st.session_state.profile
        q = st.session_state.quote or {}

        st.write(f"**Name:** {safe_get(p, 'name')}")
        st.write(f"**Ticker:** {safe_get(p, 'ticker')}")
        st.write(f"**Exchange:** {safe_get(p, 'exchange')}")
        st.write(f"**Industry:** {safe_get(p, 'finnhubIndustry')}")
        st.write(f"**Country:** {safe_get(p, 'country')}")
        st.write(f"**IPO:** {safe_get(p, 'ipo')}")
        st.write(f"**Website:** {safe_get(p, 'weburl')}")
        st.write(f"**Market Cap:** {safe_get(p, 'marketCapitalization')}")
        st.write(f"**Shares Outstanding:** {safe_get(p, 'shareOutstanding')}")

        logo = safe_get(p, "logo", "")
        if logo != "N/A":
            st.image(logo, width=90)

        st.markdown("---")
        st.subheader("Latest Quote")
        st.write(f"**Current Price:** {safe_get(q, 'c')}")
        st.write(f"**Change:** {safe_get(q, 'd')}")
        st.write(f"**Percent Change:** {safe_get(q, 'dp')}")
        st.write(f"**High:** {safe_get(q, 'h')}")
        st.write(f"**Low:** {safe_get(q, 'l')}")
        st.write(f"**Open:** {safe_get(q, 'o')}")
        st.write(f"**Previous Close:** {safe_get(q, 'pc')}")
        st.write(f"**Timestamp:** {fmt_unix(q.get('t'))}")
    else:
        st.info("Load a stock from the sidebar.")

with right:
    st.subheader("Recent News")
    if st.session_state.news:
        for item in st.session_state.news[:10]:
            headline = safe_get(item, "headline")
            url = safe_get(item, "url")
            source = safe_get(item, "source")
            summary = safe_get(item, "summary")
            timestamp = fmt_unix(item.get("datetime"))

            st.markdown(f"**[{headline}]({url})**")
            st.caption(f"{source} • {timestamp}")
            if summary != "N/A":
                st.write(summary)
            st.markdown("---")
    else:
        st.info("No news loaded yet.")

st.subheader("Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask about sentiment, risks, summary, bullish/bearish view...")

if question:
    if not st.session_state.profile or not st.session_state.quote:
        st.error("Please load a stock first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Analyzing..."):
                    answer = ask_llm(
                        st.session_state.loaded_symbol,
                        question,
                        st.session_state.profile,
                        st.session_state.quote,
                        st.session_state.news,
                    )
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"LLM error: {e}")
