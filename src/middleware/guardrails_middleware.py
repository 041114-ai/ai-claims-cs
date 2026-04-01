from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Optional
import logging

logger = logging.getLogger(__name__)

_GUARDRAILS_SYSTEM_PROMPT = """你是一个查询分类助手。你的任务是判断用户的查询是否与保险理赔相关。

保险理赔相关的主题包括：
- 理赔流程和步骤
- 理赔所需材料
- 理赔时效和进度
- 理赔金额计算
- 理赔被拒原因
- 保险条款解释
- 投保和续保问题
- 理赔案例咨询

判断标准：
- ALLOWED: 查询与保险理赔直接相关
- DENIED: 查询完全无关（如天气、娱乐、政治等）

请对以下查询进行分类。"""


class GuardrailsDecision(BaseModel):
    decision: str


class GuardrailsMiddleware(AgentMiddleware):
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.denied_queries = []

    async def _classify_query(self, messages: list) -> str | None:
        current_query = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                current_query = self._extract_message_text(msg)
                if current_query:
                    break

        if not current_query:
            return "ALLOWED"

        prior_queries = []
        for msg in messages[:-1]:
            if isinstance(msg, HumanMessage):
                text = self._extract_message_text(msg)
                if text:
                    prior_queries.append(text[:200])

        context_section = ""
        if prior_queries:
            recent = prior_queries[-3:]
            context_section = (
                "\n\n之前的对话:\n"
                + "\n".join(f"- {q}" for q in recent)
            )

        prompt = [
            SystemMessage(content=_GUARDRAILS_SYSTEM_PROMPT),
            HumanMessage(
                content=f"分类这个查询: {current_query}{context_section}"
            ),
        ]

        try:
            structured_llm = self.llm.with_structured_output(GuardrailsDecision)
            result: GuardrailsDecision = await structured_llm.ainvoke(
                prompt, config={"callbacks": [], "tags": ["guardrails"]}
            )
            return result.decision
        except Exception as e:
            logger.error(f"护栏分类错误: {e}")
            logger.info("护栏检查失败，允许查询通过...")
            return None

    def _extract_message_text(self, message) -> str:
        if isinstance(message.content, str):
            return message.content
        elif isinstance(message.content, list):
            text_parts = []
            for part in message.content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
            return " ".join(text_parts)
        return ""

    async def process(self, messages: list, config: dict, next_middleware):
        decision = await self._classify_query(messages)

        if decision == "DENIED":
            self.denied_queries.append({
                "query": self._extract_message_text(messages[-1]) if messages else "",
                "timestamp": config.get("timestamp", "unknown"),
            })
            
            return AIMessage(
                content="抱歉，我只能回答保险理赔相关的问题。如果您有理赔方面的疑问，请随时提问！"
            )

        return await next_middleware(messages, config)