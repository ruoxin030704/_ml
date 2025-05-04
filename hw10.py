import ollama

def translate_en_to_zh(text: str,
                       model: str = "llama3.2:3b") -> str:
    """
    使用 Ollama Llama 模型把英文翻譯成中文
    """
    messages = [
        {"role": "system", "content": "You are a professional translator from English to Traditional Chinese."},
        {"role": "user",   "content": text}
    ]
    resp = ollama.chat(model=model, messages=messages)
    return resp["message"]["content"].strip()

if __name__ == "__main__":
    # 範例英文句子
    sentences = [
        "Machine learning enables computers to learn from data.",
        "PyTorch is a popular deep learning framework.",
        "How do you translate this sentence into Chinese?"
    ]
    for s in sentences:
        zh = translate_en_to_zh(s)
        print(f"EN: {s}")
        print(f"ZH: {zh}\n")
