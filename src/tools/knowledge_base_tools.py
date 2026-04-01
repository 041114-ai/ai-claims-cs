from langchain.tools import tool
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import Optional
import os
import json
import logging

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "./knowledge_base")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")

_vectorstore: Optional[Chroma] = None


def get_embeddings():
    """获取 embedding 模型，优先使用本地模型"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        return OpenAIEmbeddings()
    else:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    
    embeddings = get_embeddings()
    
    if os.path.exists(CHROMA_PERSIST_DIR):
        _vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        )
    else:
        _vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
    
    return _vectorstore


@tool
def search_knowledge_base(
    query: str,
    category: str = "all",
    k: int = 5,
) -> str:
    """搜索保险理赔知识库，获取相关文章和信息。

    Args:
        query: 搜索查询，描述你想了解的理赔问题
        category: 理赔类型分类，可选值：
            - "all": 搜索所有分类（默认）
            - "车险": 车险理赔相关
            - "医疗险": 医疗险理赔相关
            - "财产险": 财产险理赔相关
            - "意外险": 意外险理赔相关
        k: 返回结果数量（默认5，最多10）

    Returns:
        JSON格式的搜索结果，包含文章标题、内容和分类
    """
    try:
        k = min(max(1, k), 10)
        
        vectorstore = get_vectorstore()
        
        filter_dict = None
        if category.lower() != "all":
            filter_dict = {"category": category}
        
        results = vectorstore.similarity_search(
            query,
            k=k,
            filter=filter_dict,
        )
        
        if not results:
            return json.dumps({
                "query": query,
                "category": category,
                "total": 0,
                "articles": [],
                "note": "未找到相关文章，请尝试不同的关键词",
            }, ensure_ascii=False, indent=2)
        
        articles = []
        for i, doc in enumerate(results, 1):
            articles.append({
                "id": str(i),
                "title": doc.metadata.get("title", "未命名文章"),
                "content": doc.page_content[:500],
                "category": doc.metadata.get("category", "未分类"),
                "source": doc.metadata.get("source", ""),
            })
        
        return json.dumps({
            "query": query,
            "category": category,
            "total": len(articles),
            "articles": articles,
            "note": "使用 get_article_detail 获取完整文章内容",
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        logger.error(f"知识库搜索失败: {e}")
        return json.dumps({
            "error": f"搜索失败: {str(e)}",
            "query": query,
        }, ensure_ascii=False, indent=2)


@tool
def get_article_detail(article_id: str) -> str:
    """获取特定文章的完整内容。

    Args:
        article_id: 文章ID（从 search_knowledge_base 获取）

    Returns:
        文章的完整内容
    """
    try:
        vectorstore = get_vectorstore()
        
        all_docs = vectorstore.similarity_search("", k=100)
        
        try:
            idx = int(article_id) - 1
            if 0 <= idx < len(all_docs):
                doc = all_docs[idx]
                return f"""文章ID: {article_id}
标题: {doc.metadata.get('title', '未命名')}
分类: {doc.metadata.get('category', '未分类')}
来源: {doc.metadata.get('source', '未知')}

完整内容:
{doc.page_content}
"""
        except (ValueError, IndexError):
            pass
        
        return f"未找到ID为 {article_id} 的文章"
    
    except Exception as e:
        logger.error(f"获取文章详情失败: {e}")
        return f"获取文章失败: {str(e)}"


def build_knowledge_base():
    """构建知识库向量索引"""
    logger.info("开始构建知识库...")
    
    embeddings = get_embeddings()
    
    if os.path.exists(CHROMA_PERSIST_DIR):
        import shutil
        shutil.rmtree(CHROMA_PERSIST_DIR)
    
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    categories = ["车险", "医疗险", "财产险", "意外险"]
    
    for category in categories:
        category_path = os.path.join(KNOWLEDGE_BASE_PATH, category)
        if not os.path.exists(category_path):
            logger.warning(f"分类目录不存在: {category_path}")
            continue
        
        logger.info(f"处理分类: {category}")
        
        for filename in os.listdir(category_path):
            filepath = os.path.join(category_path, filename)
            if not os.path.isfile(filepath):
                continue
            
            try:
                if filename.endswith('.pdf'):
                    loader = PyPDFLoader(filepath)
                elif filename.endswith('.txt') or filename.endswith('.md'):
                    loader = TextLoader(filepath, encoding='utf-8')
                else:
                    continue
                
                documents = loader.load()
                
                for doc in documents:
                    doc.metadata["category"] = category
                    doc.metadata["source"] = filename
                    doc.metadata["title"] = os.path.splitext(filename)[0]
                
                splits = text_splitter.split_documents(documents)
                vectorstore.add_documents(splits)
                
                logger.info(f"  已添加: {filename} ({len(splits)} 个片段)")
            
            except Exception as e:
                logger.error(f"  处理文件失败 {filename}: {e}")
    
    global _vectorstore
    _vectorstore = vectorstore
    
    logger.info("知识库构建完成！")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_knowledge_base()