# search_integration.py

from qwen import QwenModel, QwenTokenizer
from serpapi import GoogleSearch
from collections import defaultdict

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
    keyword_counts = defaultdict(int)
    for data in processed_data:
        snippet = data["snippet"]
        words = snippet.split()
        for word in words:
            keyword_counts[word] += 1
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

# 将搜索结果和上下文整合到 Qwen 2 模型
def generate_response_with_search(model, tokenizer, user_input, api_key):
    search_results = search_query(user_input, api_key)
    processed_data = clean_and_process_data(search_results)
    keyword_counts = analyze_data(processed_data)
    analysis_result = integrate_analysis_results(keyword_counts)

    context = "\n".join([f"标题: {result['title']}\n链接: {result['link']}\n摘要: {result['snippet']}" for result in search_results])
    prompt = f"以下是搜索结果：\n{context}\n\n用户问题：{user_input}\n请根据搜索结果回答用户的问题。"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=512)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 用法
if __name__ == "__main__":
    model = QwenModel.from_pretrained("qwen-2")
    tokenizer = QwenTokenizer.from_pretrained("qwen-2")
    context_manager = DialogueContext()

    user_input = "请讲解量子纠缠的概念！"
    api_key = "你的 API 密钥"
    response = generate_response_with_search(model, tokenizer, user_input, api_key)
    print("AI：", response)
