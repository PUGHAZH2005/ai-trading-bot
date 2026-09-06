# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import os
from openai import OpenAI

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Trading Dashboard", layout="wide", page_icon="📈")

# Use environment variable for security, or fallback to mock mode
API_KEY = os.environ.get("OPENAI_API_KEY", "mock-key")
client = OpenAI(api_key="xpl_45eaaa812fd1c31c38aa90bbb702f8e4d07b9ad5") if API_KEY != "mock-key" and API_KEY else None

# --- BOT TOOLS (Pure Pandas, No pandas_ta needed) ---
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def get_detailed_stock_analysis(ticker: str) -> str:
    try:
        df = yf.download(ticker, period="3mo", progress=False)
        if df.empty or len(df) < 30: return f"Insufficient data found for {ticker}"
        
        # Calculate indicators using pure pandas
        df['RSI'] = calculate_rsi(df)
        macd, signal = calculate_macd(df)
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        
        latest = df.iloc[-1]
        rsi_val = latest['RSI'] if pd.notna(latest['RSI']) else 50.0
        macd_val = latest['MACD'] if pd.notna(latest['MACD']) else 0.0
        
        return f"Price: ${latest['Close']:.2f} | RSI(14): {rsi_val:.2f} | MACD: {macd_val:.4f} | Vol: {int(latest['Volume'])}"
    except Exception as e:
        return f"Error fetching data: {str(e)}"

def get_memory():
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