# AI理赔智能客服

> 基于LangGraph的保险理赔智能咨询系统

[![LangGraph](https://img.shields.io/badge/Built%20with-LangGraph-blue)](https://langchain-ai.github.io/langgraph/)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 📖 项目简介

AI理赔智能客服是一个专业的保险理赔咨询系统，能够自动回答客户关于理赔流程、所需材料、理赔进度等问题。系统基于LangGraph构建，采用Chroma向量数据库作为知识库，实现了智能问答、护栏过滤、链接验证等功能。

## ✨ 核心功能

- **智能问答**：基于知识库自动回答理赔相关问题
- **知识库搜索**：使用Chroma向量数据库进行语义搜索
- **对话护栏**：自动过滤非理赔相关问题
- **链接验证**：验证回复中的链接有效性
- **多模型支持**：支持OpenAI、Anthropic、Google等LLM
- **重试机制**：自动重试失败的请求

## 🚀 快速开始

```bash
# 克隆仓库
git clone https://github.com/041114-ai/ai-claims-cs.git
cd ai-claims-cs

# 安装依赖
pip install -e . "langgraph-cli[inmem]"

# 配置环境变量
cp .env.example .env

# 构建知识库
python scripts/build_knowledge_base.py

# 启动服务
langgraph dev
```

## 📚 文档

详细文档请查看 [README.md](README.md)
