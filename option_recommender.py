"""
AI Option Recommender Application.

This module provides a Streamlit interface for an AI-powered stock option recommender.
It uses OpenAI's GPT models via the Agno library to analyze stock data and suggest
option strategies.
"""

import asyncio
import streamlit as st
from agno.agent import Agent, RunOutput
from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools


# Fix for "There is no current event loop" error in Streamlit
# keeping this at the top to ensure it runs before other imports that might use asyncio
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


def main():
    """
    Main function to run the Streamlit application.
    """
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
        [
            "Any", "Covered Call", "Put", "Call", "Iron Condor", "Straddle", "Strangle",
            "Credit Spread", "Debit Spread"
        ]
    )
    timeframe = st.selectbox(
        "Timeframe ⏳",
        ["1 week", "2 weeks", "1 month", "3 months", "6 months", "1 year"]
    )
    premium_focus = st.checkbox("Focus on Collecting Premium 💰")

    if st.button("Generate Prediction 🚀"):
        if not openai_api_key:
            st.error("Please enter your OpenAI API Key in the sidebar.")
            return

        if not ticker:
            st.error("Please enter a stock ticker.")
            return

        try:
            # Construct the strategy instruction dynamically
            strategy_instruction = ""

            if premium_focus:
                strategy_instruction += (
                    "User wants to COLLECT PREMIUM (Theta Strategy). "
                    "Prioritize selling options "
                    "(e.g., Credit Spreads, Iron Condors, Covered Calls). "
                    "Suggest strikes that are Out-of-The-Money (OTM) "
                    "with high probability of expiring worthless. "
                )

            if strategy_type == "Any":
                strategy_instruction += (
                    f"5. Recommended Strategy: Suggest ONE specific option strategy that best fits "
                    f"the analysis (and premium focus if selected), with a recommended "
                    f"duration/expiration of approximately {timeframe}."
                )
            else:
                strategy_instruction += (
                    f"5. Recommended Strategy: Evaluate if a {strategy_type} strategy is suitable "
                    f"for the current market trend. If it is NOT suitable, recommend the best "
                    f"alternative strategy instead and explicitly explain why the user's preferred "
                    f"strategy ({strategy_type}) is risky or suboptimal. Specify the recommended "
                    f"duration/expiration of approximately {timeframe}."
                )

            # Initialize the Agent
            agent = Agent(
                model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
                tools=[
                    YFinanceTools(),
                    DuckDuckGoTools()
                ],
                description="You are an expert financial analyst and options trader.",
                instructions=[
                    "1. **Summary Table**: Start with a markdown table containing: "
                    "'Price Target (Exact $)', 'Probability of Success %', "
                    "'AI Confidence Score %'. "
                    "You MUST provide at least 5 distinct rows with specific price targets for "
                    "confidence always greater than 90%. IMMEDIATELY follow the table with this "
                    "italicized note: *'Note: AI Confidence Score reflects the model's certainty "
                    "based on the alignment of technical indicators, analyst consensus, "
                    "vs market sentiment.'*",

                    "2. 'Quickbite Overview': Stock Price, Trend (Bullish/Bearish), "
                    "and Key Catalyst (1 sentence).",

                    "3. Technical Analysis: State the **current RSI value** (e.g., 'RSI: 64'). "
                    "Analyze it: if RSI > 70, consider 'Overbought' (lean bearish). "
                    "If RSI < 30, consider 'Oversold' (lean bullish). "
                    "Use this to justify the strategy.",

                    "4. Key Metrics Table: Display P/E, 52-Wk Range, and Analyst Consensus only.",

                    "5. Market Sentiment: Summarize top 3 recent news items "
                    "in 1 short bullet each.",

                    strategy_instruction,

                    "6. Prediction: Give a concise Price Target Range (1 week).",

                    "7. Important: Keep the entire response short, scannable, and avoid "
                    "long paragraphs. Use full width for tables.",

                    "8. Disclaimer: This is for educational purposes only and is "
                    "NOT financial advice.",

                    "9. Fun Style: Use relevant emojis throughout the response to "
                    "make it engaging and fun! 🤩"
                ],
                markdown=True,
                debug_mode=True,
                debug_level="all"
            )

            with st.spinner(f"Analyzing {ticker} market data and news..."):
                # Run the agent
                response: RunOutput = agent.run(
                    f"Analyze {ticker} and predict its weekly price target with confidence and "
                    f"probability. Suggest a {strategy_type} option strategy with a {timeframe} "
                    f"timeframe.",
                    stream=False
                )

                # Display Result
                st.markdown("### 📊 AI Analysis Report")
                st.markdown(response.content)

        except Exception as e:  # pylint: disable=broad-exception-caught
            st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
