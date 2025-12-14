
import json
import torch
import os
from tqdm import tqdm  # 如果没有装，运行 pip install tqdm
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================= 配置区域 (请根据你的实际路径修改) =================
# 1. 向量数据库路径 (你刚才存的地方)
VECTOR_DB_PATH = "./cmb_vector_db"

# 2. Embedding 模型路径 (必须和存库时用的一模一样！)
EMBEDDING_MODEL_PATH = "/nfs/home/9502_liuyu/wyp/medince/bge-m3"

# 3. 你的微调模型路径 (SFT/DPO后的模型)
LLM_MODEL_PATH = "/nfs/home/9502_liuyu/wyp/medince/MedicalGPT-main/merged-sft"  # <--- 请修改这里


# 4. 输入文件路径 (假设是一个 JSON 列表)
INPUT_FILE = "test_questions.json"

# 5. 输出文件路径 (结果将保存为 jsonl 格式)
OUTPUT_FILE = "rag_inference_results.jsonl"

# ================= Step 1: 初始化资源 (模型加载) =================

print(">> 正在加载 Embedding 模型...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_PATH,
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

print(">> 正在加载向量数据库...")
db = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embeddings
)

print(f">> 正在加载 LLM 模型: {LLM_MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)


# ================= Step 2: 定义 RAG 核心函数 =================

def get_rag_prompt(query, context):
    prompt_template = f"""你是一个专业的医疗AI助手。请参考以下检索到的【相似病例】或【医学知识】，准确回答用户的问题。

=== 别的患者的参考信息开始 ===
{context}
=== 别的患者的参考信息结束 ===

用户问题：{query}
请基于上述别的患者参考信息，给出这个患者专业的诊断或建议，注意术语不要太专业，尽量让患者能够理解："""
    return prompt_template


def run_rag_single(query, top_k=3):
    """
    对单个问题执行 RAG，返回结果字典
    """
    # 1. 检索
    docs = db.similarity_search(query, k=top_k)

    # 记录检索到的来源（方便后续分析 RAG 效果）
    retrieved_info = []
    context_pieces = []

    for doc in docs:
        context_pieces.append(doc.page_content)
        retrieved_info.append({
            "content": doc.page_content,
            "metadata": doc.metadata
        })

    full_context = "\n\n".join(context_pieces)

    # 2. 构造 Prompt
    final_prompt_content = get_rag_prompt(query, full_context)

    messages = [
        {"role": "system", "content": "你是一个严谨、专业的医生。"},
        {"role": "user", "content": final_prompt_content}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 3. 生成
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True
        )

    # 解码
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response, retrieved_info


# ================= Step 3: 批量处理函数 =================

def batch_inference(input_path, output_path):
    # 1. 读取输入文件
    # 假设输入是一个 JSON 列表: [{"id": 1, "question": "..."}]
    # 或者每行一个 JSON 对象
    print(f">> 读取输入文件: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        # 尝试读取整个 JSON
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # 或者是 jsonl 格式
            f.seek(0)
            data = [json.loads(line) for line in f]

    print(f">> 共有 {len(data)} 条数据待处理")

    # 2. 打开输出文件 (追加模式或写入模式)
    # 使用 jsonl 格式保存：每行一个完整的 json 结果
    with open(output_path, 'w', encoding='utf-8') as f_out:

        # 使用 tqdm 显示进度条
        for item in tqdm(data, desc="RAG 推理中"):
            # 获取问题，兼容不同的 key 名
            query = item.get("question") or item.get("query") or item.get("q") or item.get("ask")

            if not query:
                continue  # 跳过没有问题的行

            try:
                # 执行推理
                model_answer, retrieved_docs = run_rag_single(query)

                # 构造结果对象
                result_item = {
                    "id": item.get("id", "unknown"),
                    "question": query,
                    "model_answer": model_answer,
                    #"retrieved_docs": retrieved_docs,  # 把检索到的东西也存下来，方便查错
                    #"gold_answer": item.get("answer", "") or item.get("solution", "")  # 如果原数据有标准答案，也带上
                }

                # 写入一行
                f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                # 强制刷新缓冲区，防止程序崩溃数据丢失
                f_out.flush()

            except Exception as e:
                print(f"\n[Error] 处理问题 '{query}' 时出错: {e}")
                continue

    print(f"\n>> 推理完成！结果已保存至: {output_path}")


# ================= 执行 =================

if __name__ == "__main__":
    # 在这里先创建一个测试用的输入文件 (如果你没有的话)
    if not os.path.exists(INPUT_FILE):
        print(">> 未找到输入文件，正在生成测试数据...")
        test_data = [
            {"id": 101, "question": "嵌顿性腹股沟斜疝怎么治疗？"},
            {"id": 102, "question": "我可以确定我不是嵌顿性腹股沟斜疝等病情，我现在右下腹痛并摸到包块是什么病？"},
            {"id": 103, "question": "高血压患者能吃甘草片吗？"},
            {"id": 104, "question": "患者右下腹痛并摸到一个包块3小时，有呕吐腹胀，体温37.8度，这可能是什么病？"}
        ]
        with open(INPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

    # 开始运行
    batch_inference(INPUT_FILE, OUTPUT_FILE)