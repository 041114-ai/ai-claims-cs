from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
import logging

logger = logging.getLogger(__name__)


class RetryMiddleware(AgentMiddleware):
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    async def process(self, messages: list, config: dict, next_middleware):
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return await next_middleware(messages, config)
            except Exception as e:
                last_error = e
                logger.warning(f"尝试 {attempt + 1}/{self.max_retries} 失败: {e}")
                
                if attempt < self.max_retries - 1:
                    continue
        
        logger.error(f"所有重试失败: {last_error}")
        return AIMessage(
            content="抱歉，系统暂时出现问题，请稍后再试。如果问题持续，请联系人工客服。"
        )