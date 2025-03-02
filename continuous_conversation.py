import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.trainer_utils import set_seed
from threading import Thread
import random
import os
from serpapi import GoogleSearch

device = "cuda"  # the device to load the model onto

# 获取当前脚本所在的目录
current_directory = os.path.dirname(os.path.abspath(__file__))

# 加载模型和分词器
def _load_model_tokenizer(checkpoint_path, cpu_only):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, resume_download=True)

    device_map = "cpu" if cpu_only else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype="auto",
        device_map=device_map,
        resume_download=True,
    ).eval()
    model.generation_config.max_new_tokens = 512  # For chat.

    return model, tokenizer

# 搜索引擎集成
def search_query(query, api_key):
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get("organic_results", [])

# 数据整合与分析
def clean_and_process_data(search_results):
    processed_data = []
    for result in search_results:
        title = result["title"].strip()
        snippet = result["snippet"].strip()
        processed_data.append({"title": title, "snippet": snippet})
    return processed_data

def analyze_data(processed_data):
    keyword_counts = {}
    for data in processed_data:
        snippet = data["snippet"]
        words = snippet.split()
        for word in words:
            if word in keyword_counts:
                keyword_counts[word] += 1
            else:
                keyword_counts[word] = 1
    return keyword_counts

def integrate_analysis_results(keyword_counts):
    analysis_result = "关键词出现次数：\n"
    for keyword, count in keyword_counts.items():
        analysis_result += f"{keyword}: {count}\n"
    return analysis_result

# 上下文管理
class DialogueContext:
    def __init__(self):
        self.context = []

    def add_context(self, user_input, response):
        self.context.append({"user_input": user_input, "response": response})

    def get_context(self):
        return "\n".join([f"用户：{item['user_input']}\nAI：{item['response']}" for item in self.context])

# 将搜索结果和上下文整合到模型中
def generate_response_with_search(model, tokenizer, user_input, api_key):
    # 调用搜索引擎
    search_results = search_query(user_input, api_key)
    processed_data = clean_and_process_data(search_results)
    keyword_counts = analyze_data(processed_data)
    analysis_result = integrate_analysis_results(keyword_counts)

    # 构建上下文
    context = "\n".join([f"标题: {result['title']}\n链接: {result['link']}\n摘要: {result['snippet']}" for result in search_results])
    prompt = f"以下是搜索结果：\n{context}\n\n用户问题：{user_input}\n请根据搜索结果回答用户的问题。"

    # 准备输入文本
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    # 使用模型生成响应
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors='pt',
    )
    inputs = inputs.to(model.device)
    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=inputs,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text

# 主循环
if __name__ == "__main__":
    checkpoint_path = current_directory
    seed = random.randint(0, 2**32 - 1)  # 随机生成一个种子
    set_seed(seed)  # 设置随机种子
    cpu_only = False

    model, tokenizer = _load_model_tokenizer(checkpoint_path, cpu_only)
    api_key = "你的 API 密钥"  # 替换为你的 SerpAPI 密钥

    history = []

    while True:
        query = input('User: ').strip()
        if query.lower() in ["exit", "quit"]:
            break

        print(f"\nUser: {query}")
        print(f"\nAssistant: ", end="")
        try:
            partial_text = ''
            for new_text in generate_response_with_search(model, tokenizer, query, api_key):
                print(new_text, end='', flush=True)
                partial_text += new_text
            print()
            history.append((query, partial_text))

        except KeyboardInterrupt:
            print('Generation interrupted')
            continue
