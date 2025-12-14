from transformers import AutoTokenizer
# 指向您下载的目录
tokenizer = AutoTokenizer.from_pretrained("/nfs/home/9502_liuyu/wyp/medince/Qwen3-7B", trust_remote_code=True)

# 打印看看它默认的模板长什么样
print("模板配置:", tokenizer.chat_template)

# 试着格式化一句对话看看
msg = [{"role": "user", "content": "你好"}]
print("格式化结果:", tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))