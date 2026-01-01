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

    # Sidebar for Configuration
    with st.sidebar:
        st.header("Configuration")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        st.info("Get your API key from [OpenAI](https://platform.openai.com/account/api-keys)")

    # Main Input
    ticker = st.text_input("Enter Stock Ticker (e.g., NVDA, AAPL, TSLA)", value="NVDA")

    if st.button("Generate Prediction"):
        if not openai_api_key:
            st.error("Please enter your OpenAI API Key in the sidebar.")
            return
        
        if not ticker:
            st.error("Please enter a stock ticker.")
            return

        try:
            # Initialize the Agent
            agent = Agent(
                model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
                tools=[
                    YFinanceTools(),
                    DuckDuckGoTools()
                ],
                description="You are an expert financial analyst and options trader.",
                instructions=[
                    "1. Provide a 'Quickbite Overview': Stock Price, Trend (Bullish/Bearish), and Key Catalyst (1 sentence).",
                    "2. Key Metrics Table: Display P/E, 52-Wk Range, and Analyst Consensus only.",
                    "3. Market Sentiment: Summarize top 3 recent news items in 1 short bullet each.",
                    "4. Prediction: Give a concise Price Target Range (1 week) with a Confidence Score %.",
                    "5. Recommended Strategy: Suggest ONE specific option strategy (e.g., Iron Condor) with a 1-sentence rationale.",
                    "6. Important: Keep the entire response short, scannable, and avoid long paragraphs.",
                    "7. Disclaimer: This is for educational purposes only and is NOT financial advice."
                ],
                markdown=True,
            )

            with st.spinner(f"Analyzing {ticker} market data and news..."):
                # Run the agent
                response: RunOutput = agent.run(
                    f"Analyze {ticker} and predict its weekly price target with confidence and probability. Suggest an option strategy.",
                    stream=False
                )
                
                # Display Result
                st.markdown("### 📊 AI Analysis Report")
                st.markdown(response.content)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
