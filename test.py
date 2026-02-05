from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
text = "hello world"
tokens = tokenizer.encode(text)
print(f"文本: {text}")
print(f"Token数: {len(tokens)}")  # 实际token数量
print(f"Tokens: {tokens}")
