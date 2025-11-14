import os

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware

from agent.tools import TOOLS


def create_agent_with_memory(user_id):

    # --- LLM ---
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=os.getenv("GROQ_LLM"),  # or "llama-3.1-70b-versatile"
        temperature=0
    )

    agent = create_agent(
        model=llm,
        tools=TOOLS,
        middleware=[
            ToolCallLimitMiddleware(run_limit=15)
        ],
        system_prompt="""
        You are a financial analysis synthesizer.
        You receive multiple tool outputs containing ONLY numeric features.

        - produce short explanation (no indicator names)
        - produce concise numeric rationale (no indicator names)
        - compute confidence based on agreement across tools
        """,
    )

    return agent


async def run_agent(question, user):
    user_id = user.id

    agent = create_agent_with_memory(user_id)

    config = {"configurable": {"thread_id": user_id}}

    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config
    )

    return {
        "answer": response['messages'][-1].content,
    }