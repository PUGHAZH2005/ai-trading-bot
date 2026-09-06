# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import json
import os
from openai import OpenAI

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Trading Dashboard", layout="wide", page_icon="📈")

# Use environment variable for security, or fallback to mock mode
API_KEY = os.environ.get("OPENAI_API_KEY", "mock-key")
client = OpenAI(api_key="xpl_45eaaa812fd1c31c38aa90bbb702f8e4d07b9ad5") if API_KEY != "mock-key" and API_KEY else None

# --- BOT TOOLS ---
def get_detailed_stock_analysis(ticker: str) -> str:
    try:
        df = yf.download(ticker, period="3mo", progress=False)
        if df.empty: return f"No data found for {ticker}"
        
        df['RSI'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df = pd.concat([df, macd], axis=1)
        
        latest = df.iloc[-1]
        return f"Price: ${latest['Close']:.2f} | RSI(14): {latest['RSI']:.2f} | MACD: {latest['MACD_12_26_9']:.4f} | Vol: {int(latest['Volume'])}"
    except Exception as e:
        return f"Error: {str(e)}"

def get_memory():
    # Note: Free cloud hosts reset local files on restart. 
    # For permanent cloud memory, we use Streamlit's session state or a free DB like Supabase later.
    if "trade_journal" not in st.session_state:
        st.session_state.trade_journal = []
    return st.session_state.trade_journal

def save_memory(ticker, action, result, reason):
    st.session_state.trade_journal.append({"ticker": ticker, "action": action, "result": result, "reason": reason})
    st.success("✅ Saved to Bot Memory! It will learn from this during this session.")

# --- DASHBOARD UI ---
st.title("📈 Multi-Agent AI Trading Dashboard")
st.caption("Accessible 24/7 from any device")

col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### ⚙️ Controls")
    ticker = st.text_input("Stock Ticker", "AAPL").upper().strip()
    
    if st.button("🤖 Run AI Analysis", type="primary"):
        with st.spinner("Bots are fetching data, calculating indicators, and checking memory..."):
            market_data = get_detailed_stock_analysis(ticker)
            memory = get_memory()
            memory_text = str(memory[-3:]) if memory else "No past trades in memory yet."
            
            prompt = f"""You are an expert AI Trading Bot. 
            Data for {ticker}: {market_data}
            Past Memory: {memory_text}
            
            Provide a strict BUY, SELL, or HOLD recommendation. 
            If memory shows recent losses on similar setups, be extra cautious.
            Include a Risk Plan: Position size for $10k account (2% max risk), exact Stop-Loss, and Take-Profit prices."""
            
            if client:
                response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
                ai_response = response.choices[0].message.content
            else:
                ai_response = f"⚠️ MOCK MODE (Add OPENAI_API_KEY in Secrets to enable real AI).\n\n📊 ANALYSIS: RSI indicates {ticker} is approaching oversold. MACD shows bullish crossover. Recommendation: BUY cautiously.\n\n🛡️ RISK PLAN: Max risk $200. Position: 15 shares. Stop-Loss: 5% below entry. Take-Profit: 10% above entry."
            
            st.session_state.last_analysis = ai_response
            st.session_state.last_ticker = ticker

with col2:
    st.markdown(f"### 📊 {st.session_state.get('last_ticker', ticker)} Chart")
    data = yf.download(st.session_state.get('last_ticker', ticker), period="3mo", progress=False)
    
    if not data.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close']
        )])
        fig.update_layout(xaxis_rangeslider_visible=False, height=400, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not fetch chart data.")

    if "last_analysis" in st.session_state:
        st.markdown("### 🧠 AI Bot Intelligence")
        st.info(st.session_state.last_analysis)
        
        st.markdown("---")
        st.markdown("##### 🧠 Teach the Bot (Auto-Learning)")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            outcome = st.selectbox("Trade Result:", ["Win ✅", "Loss ❌"])
        with res_col2:
            reason = st.text_input("Why?", "Followed plan / Market moved unexpectedly")
            
        if st.button("💾 Save to Bot Memory"):
            save_memory(st.session_state.last_ticker, "TRADE", outcome, reason)