import json
import torch
import os
import re
from typing import List, Dict
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM

# 引入搜索工具
try:
    from duckduckgo_search import DDGS

    HAS_WEB_SEARCH = True
except ImportError:
    HAS_WEB_SEARCH = False
    print("Warning: 未安装 duckduckgo-search。请运行 pip install -U duckduckgo_search")

# ================= 配置区域 =================
VECTOR_DB_PATH = "./cmb_vector_db"
EMBEDDING_MODEL_PATH = "/nfs/home/9502_liuyu/wyp/medince/bge-m3"
LLM_MODEL_PATH = "/nfs/home/9502_liuyu/wyp/medince/MedicalGPT-main/merged-sft"

# 输入输出文件
INPUT_FILE = "test_questions_memory.json"
OUTPUT_FILE = "rag_agent_output.jsonl"

# 阈值设置
SCORE_THRESHOLD = 1.2  # L2距离阈值 (越小越相似)，超过此值视为不相关
MAX_HISTORY_TURNS = 3  # 记忆窗口大小
FETCH_K = 15  # 初次检索数量 (用于过量检索策略)

# ================= Step 1: 初始化模型 (全局加载一次) =================

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


# ================= Step 2: 核心组件类定义 =================

def local_llm_inference(prompt, max_tokens=1024):
    """底层推理函数"""
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
            temperature=0.1,
            do_sample=True
        )

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()


class ConversationManager:
    """记忆管理器：负责存储历史、压缩摘要"""

    def __init__(self, session_id):
        self.session_id = session_id
        self.history: List[Dict[str, str]] = []
        self.summary: str = ""

    def add_turn(self, user_input: str, ai_response: str):
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": ai_response})

        # 触发摘要压缩
        if len(self.history) > MAX_HISTORY_TURNS * 2:
            self._compress_history()

    def _compress_history(self):
        retain_count = 2  # 保留最近1轮问答
        to_summarize = self.history[:-retain_count]
        self.history = self.history[-retain_count:]

        content_block = "\n".join([f"{msg['role']}: {msg['content']}" for msg in to_summarize])
        prompt = f"""请总结以下对话，保留关键医学信息（症状、已排除疾病、医生建议）。
                     旧摘要：{self.summary}
                     近期对话：
                    {content_block}

                    请输出新摘要："""
        self.summary = local_llm_inference(prompt, max_tokens=200)

    def get_context_str(self):
        context = ""
        if self.summary:
            context += f"【历史摘要】：{self.summary}\n"
        if self.history:
            context += "【近期对话】：\n"
            for msg in self.history:
                context += f"{msg['role']}: {msg['content']}\n"
        return context


class RagAgent:
    """Agent 逻辑封装"""

    def __init__(self):
        pass

    def rewrite_query(self, query, memory: ConversationManager):
        """利用记忆重写查询"""
        if not memory.history and not memory.summary:
            return query  # 无历史，无需重写

        prompt = f"""根据历史对话，将用户最新问题改写为指代清晰的完整句子。
                    {memory.get_context_str()}

                    用户问题：{query}

                    改写后的查询（直接输出结果）："""

        rewritten = local_llm_inference(prompt, max_tokens=128)
        # 简单的清洗
        rewritten = rewritten.replace("改写后的查询：", "").replace("改写为：", "").strip()
        return rewritten

    def analyze_intent(self, query):
        """意图与否定词提取"""
        prompt = f"""分析用户输入。
                    用户输入：{query}

                    请判断：
                    1. 是否检索：涉及病情、治疗输出YES；闲聊输出NO。
                    2. 关键词：搜索用的症状/疾病。
                    3. 排除词：用户说“不是/排除”的词。

                    格式：
                    是否检索：YES/NO
                    关键词：[A, B]
                    排除词：[C, D]"""

        res = local_llm_inference(prompt, max_tokens=200)

        need_retrieval = True
        if "是否检索：NO" in res or "是否检索：no" in res:
            need_retrieval = False

        keywords = [query]
        kw_match = re.search(r"关键词：\[(.*?)\]", res)
        if kw_match:
            keywords = [k.strip() for k in kw_match.group(1).split(",") if k.strip()]

        negatives = []
        neg_match = re.search(r"排除词：\[(.*?)\]", res)
        if neg_match:
            negatives = [n.strip() for n in neg_match.group(1).split(",") if n.strip()]

        return need_retrieval, keywords, negatives, res

    def web_search(self, keywords, negatives):
        if not HAS_WEB_SEARCH: return []
        q = " ".join(keywords)
        if negatives: q += " " + " ".join([f"-{n}" for n in negatives])
        q += " 鉴别诊断"
        try:
            # 使用 DDGS 上下文管理器，防止连接未关闭
            with DDGS() as ddgs:
                results = list(ddgs.text(q, max_results=3))
            return [f"来源：{r['title']}\n摘要：{r['body']}" for r in results]
        except Exception as e:
            return [f"网络搜索出错: {str(e)}"]

    def run(self, raw_query, memory: ConversationManager, top_k=3):
        logs = []

        # 1. 查询重写
        rewritten_query = self.rewrite_query(raw_query, memory)
        logs.append(f"原始输入: {raw_query}")
        if rewritten_query != raw_query:
            logs.append(f"记忆重写: {rewritten_query}")

        # 2. 意图分析
        need_retrieval, keywords, negatives, raw_analysis = self.analyze_intent(rewritten_query)
        logs.append(f"意图分析: {keywords} | 排除: {negatives}")

        # 分支：闲聊模式
        if not need_retrieval:
            prompt = f"{memory.get_context_str()}\n用户：{raw_query}\n请礼貌回答。"
            ans = local_llm_inference(prompt)
            memory.add_turn(raw_query, ans)
            return ans, logs, "LLM直出"

        # 3. 向量检索 (Over-fetching)
        search_q = " ".join(keywords)
        docs_with_score = db.similarity_search_with_score(search_q, k=FETCH_K)

        # 4. 过滤 (分数 + 否定词)
        valid_docs = []
        dropped_info = []

        for doc, score in docs_with_score:
            if len(valid_docs) >= top_k: break  # 够了就停

            # L2距离过滤
            if score > SCORE_THRESHOLD:
                dropped_info.append(f"score:{score:.2f}")
                continue

            # 否定词过滤
            if negatives and any(n in doc.page_content for n in negatives):
                dropped_info.append("negative_term")
                continue

            valid_docs.append(doc.page_content)

        logs.append(f"检索{len(docs_with_score)}条 -> 过滤后剩余{len(valid_docs)}条。丢弃原因: {dropped_info[:5]}...")

        # 5. 决策与回退
        context_source = "KnowledgeBase"
        final_context = valid_docs

        if not valid_docs:
            logs.append("知识库无有效结果，切换网络搜索")
            context_source = "WebSearch"
            final_context = self.web_search(keywords, negatives)

        # 6. 生成
        context_text = "\n\n".join(final_context)
        prompt = f"""你是一名医生。结合历史和资料回答。
                {memory.get_context_str()}

                【参考资料 ({context_source})】：
                {context_text}

                【用户明确排除】：{negatives}
                【用户当前问题】：{raw_query}

                请给出诊断建议。"""

        final_answer = local_llm_inference(prompt)

        # 更新记忆
        memory.add_turn(raw_query, final_answer)

        return final_answer, logs, context_source


# ================= Step 3: 批量文件处理 (主程序) =================

def process_file_batch():
    # 1. 生成测试数据
    if not os.path.exists(INPUT_FILE):
        print(">> 生成测试文件...")
        test_data = [
            {"session_id": "user_001", "id": 1, "question": "你好，你是谁？"},
            {"session_id": "user_001", "id": 2, "question": "我右下腹痛，有包块，但确定不是疝气，是什么？"},
            {"session_id": "user_001", "id": 3, "question": "这个病严重吗？需要手术吗？"},
            {"session_id": "user_002", "id": 4, "question": "高血压能吃甘草片吗？"},
            {"session_id": "user_003", "id": 5, "question": "外星人感冒吃什么药？"}
        ]
        with open(INPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

    agent = RagAgent()
    sessions: Dict[str, ConversationManager] = {}

    print(f">> 开始处理文件: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 写入 TXT 文件 (关键修改)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for item in tqdm(data, desc="Batch Inference"):

            sess_id = item.get("session_id", "default_session")
            query = item.get("question")

            if sess_id not in sessions:
                sessions[sess_id] = ConversationManager(sess_id)

            current_memory = sessions[sess_id]

            try:
                answer, thoughts, source = agent.run(query, current_memory)

                # ==========================================
                # 修改区域：格式化为纯文本块写入
                # ==========================================

                # 将思考过程列表转为带缩进的字符串
                thoughts_str = "\n".join([f"  > {t}" for t in thoughts])

                # 构造人类可读的文本块
                record_text = f"""
                ==================================================
                【ID】: {item.get('id')} | 【Session】: {sess_id}
                【问题】: {query}
                    --------------------------------------------------
                【Agent 思考过程】:
                {thoughts_str}
                --------------------------------------------------
                【最终回答】 (来源: {source}):
                {answer}
                ==================================================
                \n"""
                # 直接写入字符串，而不是 json.dump
                f_out.write(record_text)
                f_out.flush()

            except Exception as e:
                print(f"[Error] ID {item.get('id')} failed: {e}")

    print(f"\n>> 处理完成！结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_file_batch()