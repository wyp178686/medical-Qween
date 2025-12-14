import json
import torch
import os
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# ================= 配置区域 =================
VECTOR_DB_PATH = "./cmb_vector_db"
EMBEDDING_MODEL_PATH = "/nfs/home/9502_liuyu/wyp/medince/bge-m3"
LLM_MODEL_PATH = "/nfs/home/9502_liuyu/wyp/medince/MedicalGPT-main/merged-sft"

# 输入文件 (格式见下文说明)
INPUT_FILE = "conversation_test.json"
# 输出文件
OUTPUT_FILE = "conversation_result.jsonl"


# ================= 1. 初始化系统 =================

def init_rag_chain():
    print(">> [Init] 正在加载模型和数据库...")

    # 1. Embedding
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. Vector DB
    vector_db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )

    # 3. LLM
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # 构建 Pipeline
    text_generation_pipeline = pipeline(
        "text_generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        repetition_penalty=1.1
    )
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    # 4. 记忆模块 (核心：带自动总结功能)
    # max_token_limit=1000: 当历史对话积攒超过1000个Token时，自动触发总结
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=1000,
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    # 5. Prompt 模板
    prompt_template = """你是一个专业的医疗AI助手。请结合【参考资料】和【历史对话】，回答患者的问题。

=== 历史对话摘要 ===
{chat_history}

=== 参考资料 ===
{context}

=== 患者问题 ===
{question}

专业的医生回答："""

    QA_PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["chat_history", "context", "question"]
    )

    # 6. 构建链
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )

    return qa_chain


# ================= 2. 批量处理函数 =================

def process_file_conversation(input_path, output_path):
    # 初始化链 (Memory 在这里被创建，并一直存在于 qa_chain 对象中)
    chain = init_rag_chain()

    # 读取数据
    if not os.path.exists(input_path):
        print(f"Error: 找不到输入文件 {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        # 假设输入是一个列表，代表按顺序发生的对话
        # 例如: [{"id": 1, "query": "头疼"}, {"id": 2, "query": "吃什么药?"}]
        data = json.load(f)

    print(f">> 开始处理对话，共 {len(data)} 轮...")

    # 打开输出文件 (使用 'w' 覆盖模式，或者 'a' 追加模式)
    with open(output_path, 'w', encoding='utf-8') as f_out:

        for item in tqdm(data, desc="Processing"):
            query = item.get("query") or item.get("question")

            # (可选) 如果你的文件里包含 "clear_memory": true 的标记
            # 可以在这里清空记忆，模拟开启一段新对话
            if item.get("new_session", False):
                chain.memory.clear()
                print("\n[Info] 检测到新会话标记，记忆已清空。")

            try:
                # === 核心调用 ===
                # LangChain 会自动读取 Memory -> 拼接 Prompt -> 检索 -> 生成 -> 更新 Memory
                result = chain.invoke({"question": query})

                # 获取结果
                answer = result["answer"]

                # 获取当前的“记忆摘要”（看看模型脑子里记住了什么）
                # moving_summary_buffer 是 SummaryBufferMemory 特有的属性
                current_summary = chain.memory.moving_summary_buffer

                # 构造输出记录
                record = {
                    "id": item.get("id"),
                    "question": query,
                    "model_answer": answer,
                    "memory_summary": current_summary,  # 把当前的总结也存下来，方便你检查
                    "retrieved_docs": [doc.page_content[:30] + "..." for doc in result["source_documents"]]
                }

                # 写入文件 (JSONL 格式)
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()  # 立即写入，防止程序崩溃丢失

            except Exception as e:
                print(f"Error processing '{query}': {e}")

    print(f"\n>> 处理完成！结果已保存至 {output_path}")


# ================= 执行 =================

if __name__ == "__main__":
    # 1. 如果没有输入文件，先生成一个测试用的
    if not os.path.exists(INPUT_FILE):
        print(">> 生成测试文件...")
        test_data = [
            {"id": 1, "query": "医生，我最近总是感觉右上腹隐痛，是什么原因？", "new_session": True},
            {"id": 2, "query": "这种疼痛通常在吃完油腻食物后加重。"},
            {"id": 3, "query": "这会是胆囊炎吗？"},
            {"id": 4, "query": "那我应该吃点什么药来缓解它？"},
            # 注意：这里的“它”指代上面的病，如果记忆功能生效，模型会知道是胆囊炎
            {"id": 5, "query": "如果我要做手术的话，需要多少钱？"}
        ]
        with open(INPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

    # 2. 运行
    process_file_conversation(INPUT_FILE, OUTPUT_FILE)