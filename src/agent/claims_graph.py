from langgraph.prebuilt import create_react_agent
from langchain.agents.middleware import apply_middleware
from langchain_core.messages import HumanMessage

from src.agent.config import get_model
from src.tools.knowledge_base_tools import search_knowledge_base, get_article_detail
from src.tools.link_check_tools import check_links
from src.middleware.guardrails_middleware import GuardrailsMiddleware
from src.middleware.retry_middleware import RetryMiddleware
from src.prompts.claims_agent_prompt import CLAIMS_AGENT_PROMPT

import logging

logger = logging.getLogger(__name__)


def create_claims_agent():
    model = get_model()
    
    tools = [
        search_knowledge_base,
        get_article_detail,
        check_links,
    ]
    
    guardrails_middleware = GuardrailsMiddleware()
    retry_middleware = RetryMiddleware(max_retries=3)
    
    agent = create_react_agent(
        model=model,
        tools=tools,
        state_modifier=CLAIMS_AGENT_PROMPT,
    )
    
    agent_with_middleware = apply_middleware(
        agent,
        middleware=[guardrails_middleware, retry_middleware],
    )
    
    return agent_with_middleware


graph = create_claims_agent()


async def chat(query: str, thread_id: str = "default"):
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }
    
    messages = [HumanMessage(content=query)]
    
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