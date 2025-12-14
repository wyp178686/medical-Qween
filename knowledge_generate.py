import json
from typing import List
from langchain_core.documents import Document


def process_cmb_data(file_path: str) -> List[Document]:
    """
    将 CMB 格式的 JSON 数据转换为 RAG 可用的 Document 列表。
    策略：Flatten（扁平化）处理。
    一条 JSON (1个病例 + N个问答) -> 拆分成 N 个 Document。
    每个 Document = 病例描述 + 当前的问 + 当前的答。
    """

    documents = []

    # 读取数据 (假设是 json 列表，或者 jsonl)
    with open(file_path, 'r', encoding='utf-8') as f:
        # 如果文件是整个列表 [{}, {}]
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # 如果文件是 jsonl (每行一个 json)
            f.seek(0)
            data = [json.loads(line) for line in f]

    for item in data:
        # 1. 提取公共上下文 (Context)
        case_id = item.get('id', 'unknown')
        title = item.get('title', '无标题')
        description = item.get('description', '')

        # 2. 遍历该病例下的所有问答对
        for qa in item.get('QA_pairs', []):
            question = qa.get('question', '')
            solution = qa.get('answer', '')

            # 3. 核心步骤：构造“富文本”块 (Chunk)
            # 我们使用 Markdown 格式拼接，这样语义结构最清晰
            page_content = (
                f"【病例标题】：{title}\n"
                f"【病例详情】：\n{description}\n"
                f"{'-' * 20}\n"  # 分隔线
                f"【临床问题】：{question}\n"
                f"【专家详解】：\n{solution}"
            )

            # 4. 构造元数据 (Metadata)
            # 这对后续检索过滤很有用，比如用户只想搜“治疗原则”
            metadata = {
                "source": "CMB_Case_Study",
                "original_id": case_id,
                "title": title,
                "type": "clinical_case",
                # 甚至可以尝试自动提取问题的类型
                "qa_type": "diagnosis" if "诊断" in question else "treatment" if "治疗" in question else "general"
            }

            # 5. 生成 Document 对象
            doc = Document(
                page_content=page_content,
                metadata=metadata
            )
            documents.append(doc)

    return documents


# --- 使用示例 ---

# 假设你把上面那个 JSON 保存为了 cmb_sample.json
docs = process_cmb_data("/nfs/home/9502_liuyu/wyp/medince/MedicalGPT-main/CMB_Clin/CMB-Clin/CMB-Clin-qa.json")

print(f"成功处理，共生成 {len(docs)} 个知识切片。")
print("-" * 30)
print("第一条数据切片预览：")
print(docs[0].page_content)
print("-" * 30)
print("元数据：", docs[0].metadata)

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. 定义 Embedding 模型
embeddings = HuggingFaceEmbeddings(
    model_name="/nfs/home/9502_liuyu/wyp/medince/bge-m3", # 强烈推荐用这个处理长文本
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

# 2. 存入 ChromaDB
print("正在存入向量数据库...")
db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./cmb_vector_db"
)
print("存入完成！")