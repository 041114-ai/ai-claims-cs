import streamlit as st
from src.agent.claims_graph import create_claims_agent, chat
from langchain_core.messages import HumanMessage, AIMessage
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI理赔智能客服",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #e3f2fd;
    }
    .chat-message.assistant {
        background-color: #f5f5f5;
    }
    .chat-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
    }
    .chat-content {
        flex: 1;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    with st.spinner("正在初始化AI客服..."):
        st.session_state.agent = create_claims_agent()

st.title("🛡️ AI理赔智能客服")
st.markdown("---")

with st.sidebar:
    st.header("📋 使用说明")
    st.markdown("""
    ### 我可以帮您：
    - 🔍 解答理赔流程问题
    - 📝 指导准备理赔材料
    - ⏱️ 查询理赔进度
    - 💡 解决常见理赔问题
    - 📖 解释保险条款
    
    ### 理赔类型：
    - 🚗 车险理赔
    - 🏥 医疗险理赔
    - 🏠 财产险理赔
    - ⚡ 意外险理赔
    """)
    
    st.markdown("---")
    
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
        Powered by LangGraph + Chroma
    </div>
    """, unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("请输入您的理赔问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            async def get_response():
                responses = []
                async for response in chat(prompt, thread_id="streamlit_user"):
                    responses.append(response)
                return "".join(responses)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(get_response())
            loop.close()
            
            if not response:
                response = "抱歉，我暂时无法回答这个问题。请稍后再试或联系人工客服。"
            
            message_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            error_msg = f"抱歉，系统出现错误：{str(e)}\n\n请稍后再试或联系人工客服。"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 14px;'>
    💡 提示：本系统仅提供理赔咨询服务，具体理赔事宜请以保险公司官方回复为准
</div>
""", unsafe_allow_html=True)