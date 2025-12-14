import json
import torch
import os
import re
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM

# 引入搜索工具 (需 pip install duckduckgo-search)
try:
    from duckduckgo_search import DDGS

    HAS_WEB_SEARCH = True
except ImportError:
    HAS_WEB_SEARCH = False
    print("Warning: 未安装 duckduckgo-search，网络搜索功能将不可用。请运行 pip install duckduckgo-search")

# ================= 配置区域 =================
VECTOR_DB_PATH = "./cmb_vector_db"
EMBEDDING_MODEL_PATH = "/nfs/home/9502_liuyu/wyp/medince/bge-m3"
LLM_MODEL_PATH = "/nfs/home/9502_liuyu/wyp/medince/MedicalGPT-main/merged-sft"

INPUT_FILE = "test_questions.json"
OUTPUT_FILE = "rag_agent_results.jsonl"

# 【新增】相似度阈值 (L2距离)
# Chroma 默认 L2 距离：0 是完全一样，数字越大越不相关。
# 建议值：0.8 ~ 1.2 之间。如果发现稍微不相关的都被过滤了，调大这个数；如果太多垃圾文档，调小这个数。
SCORE_THRESHOLD = 0.8

# ================= Step 1: 初始化资源 =================

print(">> 正在加载 Embedding 模型...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_PATH,
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

print(">> 正在加载向量数据库...")
db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)

print(f">> 正在加载 LLM 模型: {LLM_MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)


# ================= Step 2: 辅助函数 (Agent 核心组件) =================

def local_llm_inference(prompt, max_tokens=1024):
    """通用的本地模型推理函数"""
    messages = [
        {"role": "system", "content": "你是一个专业的医疗AI助手。"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,  # 低温，保证逻辑稳定
            do_sample=True
        )

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()


def analyze_query_intent(query):
    """
    Agent 思考步：
    1. Router: 判断是否需要检索。
    2. Extractor: 提取关键词和否定词。
    """
    # 【修改】Prompt 增加了“是否检索”的判断
    analysis_prompt = f"""请分析用户输入。

用户输入：{query}

请判断并提取以下信息：
1. 【是否检索】：用户是在询问医学病情、药物、治疗方案吗？如果是，输出 YES；如果是闲聊、问候或明显无关的问题，输出 NO。
2. 【关键词】：检索数据库的核心症状或实体。
3. 【排除词】：用户明确说“不是”、“排除”的疾病。

格式要求：
是否检索：YES/NO
关键词：[词1, 词2]
排除词：[词1, 词2]
"""
    # 调用模型进行分析
    result_text = local_llm_inference(analysis_prompt, max_tokens=256)

    keywords = []
    negatives = []
    need_retrieval = True  # 默认兜底为 True

    try:
        # 1. 解析是否需要检索
        if "是否检索：NO" in result_text or "是否检索：no" in result_text:
            need_retrieval = False

        # 2. 解析关键词
        kw_match = re.search(r"关键词：\[(.*?)\]", result_text)
        if kw_match:
            keywords = [k.strip() for k in kw_match.group(1).split(",") if k.strip()]
        else:
            if need_retrieval: keywords = [query]  # 只有需要检索时才兜底

        # 3. 解析排除词
        neg_match = re.search(r"排除词：\[(.*?)\]", result_text)
        if neg_match:
            negatives = [n.strip() for n in neg_match.group(1).split(",") if n.strip()]

    except Exception as e:
        print(f"解析出错: {e}, 使用默认策略")
        keywords = [query]

    return need_retrieval, keywords, negatives, result_text


def web_search_tool(query_terms, negative_terms):
    """网络搜索工具 (DuckDuckGo)"""
    if not HAS_WEB_SEARCH:
        return []

    # 构造搜索词
    search_str = " ".join(query_terms)
    if negative_terms:
        search_str += " " + " ".join([f"-{n}" for n in negative_terms])

    search_str += " 鉴别诊断"

    print(f"   [Web Search] 正在联网搜索: {search_str}")
    try:
        results = DDGS().text(search_str, max_results=3)
        return [f"网络来源标题：{r['title']}\n内容摘要：{r['body']}" for r in results]
    except Exception as e:
        print(f"网络搜索失败: {e}")
        return []


# ================= Step 3: Agentic RAG 主流程 =================

def run_rag_agent(query, top_k=3):
    """
    包含 思考(路由) -> 检索(阈值截断) -> 过滤(否定词) -> 搜索回退 -> 生成 的完整流程
    """
    logs = []

    # --- 1. 意图分析 (Router & Analyzer) ---
    print(f"1. 分析意图: {query}")
    need_retrieval, keywords, negatives, analysis_raw = analyze_query_intent(query)
    logs.append(f"意图分析: {analysis_raw}")

    # 【新增】路由分支：如果不需要检索，直接让 LLM 回答
    if not need_retrieval:
        print("   -> 判定为非检索类问题 (Chat Mode)")
        direct_prompt = f"用户说：{query}\n请礼貌、自然地回答用户。如果是闲聊就正常聊天；如果是无法回答的问题请直说。"
        answer = local_llm_inference(direct_prompt)
        return answer, logs, "LLM内置知识(无检索)"

    # --- 2. 向量检索 (带阈值) ---
    search_query = " ".join(keywords)
    print(f"2. 向量检索: {search_query}")

    # 【修改】使用 similarity_search_with_score
    # 返回格式: [(doc, score), (doc, score)]
    docs_with_score = db.similarity_search_with_score(search_query, k=top_k * 2)

    # --- 3. 双重过滤 (分数 + 否定词) ---
    valid_docs = []
    dropped_by_score = 0.0
    dropped_by_neg = 0

    for doc, score in docs_with_score:
        # 3.1 分数阈值过滤 (L2距离: 越小越好)
        if score > SCORE_THRESHOLD:
            dropped_by_score += 1
            continue

        # 3.2 否定词过滤
        if negatives and any(neg in doc.page_content for neg in negatives):
            dropped_by_neg += 1
            continue

        valid_docs.append(doc.page_content)

    log_msg = f"检索 {len(docs_with_score)} 条 | 因分数过低丢弃: {dropped_by_score} | 因含排除词丢弃: {dropped_by_neg} | 剩余有效: {len(valid_docs)}"
    print(f"   {log_msg}")
    logs.append(log_msg)

    context_source = "内部知识库"
    final_context = []

    # --- 4. 决策与回退 (Decision & Fallback) ---
    # 如果有效文档为空 (可能是本来就没搜到，也可能是被过滤光了)
    if len(valid_docs) == 0:
        logs.append("内部知识库无有效匹配，切换至网络搜索。")
        print("3. 触发网络搜索...")
        web_docs = web_search_tool(keywords, negatives)
        final_context = web_docs
        context_source = "互联网搜索"
    else:
        # 截取 top_k
        final_context = valid_docs[:top_k]
        print("3. 使用内部知识库...")

    # --- 5. 最终生成 ---
    context_text = "\n\n".join(final_context)

    final_prompt = f"""你是一个专业的医生。请根据参考资料回答问题。

            【用户问题】：{query}
【用户明确排除】：{negatives}

【参考资料 ({context_source})】：
{context_text}

请给出通俗易懂的诊断建议。如果参考资料不足，请说明。"""

    answer = local_llm_inference(final_prompt)

    return answer, logs, context_source


# ================= Step 4: 批量处理函数 =================

def batch_inference(input_path, output_path):
    print(f">> 读取输入文件: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except:
            f.seek(0)
            data = [json.loads(line) for line in f]

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in tqdm(data, desc="Agent 推理中"):
            query = item.get("question")
            if not query: continue

            try:
                model_answer, thoughts, source = run_rag_agent(query)

                result_item = {
                    "id": item.get("id"),
                    "question": query,
                    "model_answer": model_answer,
                    "agent_thoughts": thoughts,
                    "source": source
                }

                f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                f_out.flush()

            except Exception as e:
                print(f"[Error] {e}")
                continue

    print(f"\n>> 完成！结果已保存至: {output_path}")


# ================= 执行 =================

if __name__ == "__main__":
    # 生成测试数据
    if not os.path.exists(INPUT_FILE):
        test_data = [
            {"id": 101, "question": "你好"},  # 测试 Router (应该不检索)
            {"id": 102, "question": "我可以确定我不是嵌顿性腹股沟斜疝、疝气，我现在右下腹痛并摸到包块是什么病？"},
            # 测试 RAG + 过滤
            {"id": 103, "question": "外星人感冒了怎么办？"}  # 测试 Threshold (应该检索不到 -> 联网/回答不知道)
        ]
        with open(INPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

    batch_inference(INPUT_FILE, OUTPUT_FILE)