import os

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage

from agent.memory import load_memory, save_memory
from agent.tools import TOOLS


def create_agent_with_memory(user_id):

    # --- LLM ---
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=os.getenv("GROQ_LLM"),  # or "llama-3.1-70b-versatile"
        temperature=0
    ).bind_tools(TOOLS)

    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt="""
        You are a finance assistant. 
        - Always respond in under 3 tool calls unless absolutely necessary.
        - After getting data from tools, SUMMARIZE and respond to the user.
        - NEVER loop infinitely.
        """,
    )

    return agent, None


async def run_agent(question, user):
    user_id = user.id

    agent, old_mem = create_agent_with_memory(user_id)

    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
    )
    import logging
    logging.info(response)
    # answer = result["output"]

    # # --- Append new Q/A to memory ---
    # updated_mem = old_mem + [
    #     {"role": "user", "content": question},
    #     {"role": "assistant", "content": answer},
    # ]

    # save_memory(user_id, updated_mem)

    return {
        "answer": response['messages'][-1],
        # "tools_used": response['message'][1]
    }