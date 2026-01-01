import streamlit as st
import asyncio

# Fix for "There is no current event loop" error in Streamlit
# keeping this at the top to ensure it runs before other imports that might use asyncio
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from agno.agent import Agent, RunOutput
from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools

def main():
    st.set_page_config(page_title="AI Option Recommender", page_icon="📈", layout="wide")

    st.title("AI Option Recommender 📈🤖")
    st.markdown("""
    This agent analyzes stock data (price, fundamentals, analyst recommendations) and searches for recent news 
    to predict weekly price targets and recommend potential option strategies.
    
    **Disclaimer:** This is for educational purposes only and is NOT financial advice.
    """)

    with st.sidebar:
        st.header("Configuration ⚙️")
        openai_api_key = st.text_input("OpenAI API Key 🔑", type="password")
        
        st.info("Get your API key from [OpenAI](https://platform.openai.com/account/api-keys)")

    # Main Input
    ticker = st.text_input("Enter Stock Ticker 📈 (e.g., NVDA, AAPL, TSLA)", value="NVDA")
    st.header("Strategy Preferences 🎯")
    strategy_type = st.selectbox(
        "Preferred Strategy ♟️",
        ["Any", "Covered Call", "Put", "Call", "Iron Condor", "Straddle", "Strangle", "Credit Spread", "Debit Spread"]
    )
    timeframe = st.selectbox(
        "Timeframe ⏳",
        ["1 week", "2 weeks", "1 month", "3 months", "6 months", "1 year"]
    )

    if st.button("Generate Prediction 🚀"):
        if not openai_api_key:
            st.error("Please enter your OpenAI API Key in the sidebar.")
            return
        
        if not ticker:
            st.error("Please enter a stock ticker.")
            return

        try:
            # Construct the strategy instruction dynamically
            if strategy_type == "Any":
                strategy_instruction = f"5. Recommended Strategy: Suggest ONE specific option strategy (e.g., Covered Call, Put, Iron Condor) that best fits the analysis, with a recommended duration/expiration of approximately {timeframe}."
            else:
                strategy_instruction = f"5. Recommended Strategy: Evaluate if a {strategy_type} strategy is suitable for the current market trend. If it is NOT suitable, recommend the best alternative strategy instead and explicitly explain why the user's preferred strategy ({strategy_type}) is risky or suboptimal. Specify the recommended duration/expiration of approximately {timeframe}."

            # Initialize the Agent
            agent = Agent(
                model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
                tools=[
                    YFinanceTools(),
                    DuckDuckGoTools()
                ],
                description="You are an expert financial analyst and options trader.",
                instructions=[
                    "1. **Summary Table**: Start with a markdown table containing: 'Price Target (Exact $)', 'Probability of Success %', 'AI Confidence Score %'. You MUST provide at least 5 distinct rows with specific price targets (NOT ranges).",
                    "2. 'Quickbite Overview': Stock Price, Trend (Bullish/Bearish), and Key Catalyst (1 sentence).",
                    "3. Key Metrics Table: Display P/E, 52-Wk Range, and Analyst Consensus only.",
                    "4. Market Sentiment: Summarize top 3 recent news items in 1 short bullet each.",
                    strategy_instruction,
                    "6. Prediction: Give a concise Price Target Range (1 week).",
                    "7. Important: Keep the entire response short, scannable, and avoid long paragraphs. Use full width for tables.",
                    "8. Disclaimer: This is for educational purposes only and is NOT financial advice.",
                    "9. Fun Style: Use relevant emojis throughout the response to make it engaging and fun! 🤩"
                ],
                markdown=True,
            )

            with st.spinner(f"Analyzing {ticker} market data and news..."):
                # Run the agent
                response: RunOutput = agent.run(
                    f"Analyze {ticker} and predict its weekly price target with confidence and probability. Suggest a {strategy_type} option strategy with a {timeframe} timeframe.",
                    stream=False
                )
                
                # Display Result
                st.markdown("### 📊 AI Analysis Report")
                st.markdown(response.content)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
