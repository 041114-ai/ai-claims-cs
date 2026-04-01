#!/usr/bin/env python3
"""
知识库构建脚本
用于将本地文档转换为向量数据库
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.knowledge_base_tools import build_knowledge_base
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    logger.info("="*60)
    logger.info("AI理赔客服 - 知识库构建工具")
    logger.info("="*60)
    
    knowledge_base_path = os.getenv("KNOWLEDGE_BASE_PATH", "./knowledge_base")
    chroma_path = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    
    logger.info(f"知识库路径: {knowledge_base_path}")
    logger.info(f"向量数据库路径: {chroma_path}")
    
    if not os.path.exists(knowledge_base_path):
        logger.error(f"知识库目录不存在: {knowledge_base_path}")
        logger.info("请创建知识库目录并添加文档")
        sys.exit(1)
    
    try:
        build_knowledge_base()
        logger.info("="*60)
        logger.info("知识库构建成功！")
        logger.info("="*60)
    except Exception as e:
        logger.error(f"知识库构建失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()