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
                    "1. Get the current stock price, fundamentals, and analyst recommendations for the given ticker.",
                    "2. Search for the latest news, market sentiment, and any upcoming catalysts (earnings, events) for the company.",
                    "3. Analyze the data to predict a price target range for the next week/month.",
                    "4. Provide a 'Confidence Score' (0-100%) and a 'Probability' of the move.",
                    "5. Based on the prediction, suggest a potential option strategy (e.g., Call, Put, Straddle, Iron Condor) with reasoning.",
                    "6. Format your response in a clean, professional Markdown report.",
                    "7. Always include a disclaimer that this is not financial advice."
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
