from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel
import os


def get_model() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "deepseek").lower()
    
    if provider == "anthropic":
        return ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0,
        )
    elif provider == "google":
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
        )
    elif provider == "deepseek":
        return ChatOpenAI(
            model="deepseek-chat",
            temperature=0,
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base="https://api.deepseek.com/v1",
        )
    else:
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )


def get_fallback_models() -> list[BaseChatModel]:
    return [
        ChatOpenAI(
            model="deepseek-chat",
            temperature=0,
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base="https://api.deepseek.com/v1",
        ),
        ChatOpenAI(model="gpt-4o-mini", temperature=0),
    ]