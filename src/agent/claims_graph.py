from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

from src.agent.config import get_model
from src.tools.knowledge_base_tools import search_knowledge_base, get_article_detail
from src.tools.link_check_tools import check_links
from src.middleware.guardrails_middleware import GuardrailsMiddleware
from src.prompts.claims_agent_prompt import CLAIMS_AGENT_PROMPT

import logging

logger = logging.getLogger(__name__)

_guardrails = GuardrailsMiddleware()
_checkpointer = MemorySaver()


def create_claims_agent():
    model = get_model()
    
    tools = [
        search_knowledge_base,
        get_article_detail,
        check_links,
    ]
    
    agent = create_react_agent(
        model=model,
        tools=tools,
        state_modifier=CLAIMS_AGENT_PROMPT,
        checkpointer=_checkpointer,
    )
    
    return agent


graph = create_claims_agent()


async def chat(query: str, thread_id: str = "default"):
    messages = [HumanMessage(content=query)]
    
    is_allowed = await _guardrails._classify_query(messages)
    
    if is_allowed == "DENIED":
        yield "抱歉，我只能回答保险理赔相关的问题。如果您有理赔方面的疑问，请随时提问！"
        return
    
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }
    
    try:
        async for event in graph.astream_events(
            {"messages": messages},
            config=config,
            version="v2",
        ):
            if event["event"] == "on_chain_end":
                if event["name"] == "agent":
                    output = event["data"]["output"]
                    if "messages" in output:
                        for msg in output["messages"]:
                            if hasattr(msg, "content"):
                                yield msg.content
    except Exception as e:
        logger.error(f"Agent error: {e}")
        yield f"抱歉，系统出现错误：{str(e)}\n\n请稍后再试或联系人工客服。"


if __name__ == "__main__":
    import asyncio
    
    async def main():
        agent = create_claims_agent()
        
        test_queries = [
            "车险理赔需要哪些材料？",
            "医疗险理赔流程是什么？",
            "今天天气怎么样？",
        ]
        
        for query in test_queries:
            print(f"\n用户: {query}")
            print("助手: ", end="")
            async for response in chat(query):
                print(response, end="", flush=True)
            print("\n" + "="*50)
    
    asyncio.run(main())
