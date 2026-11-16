from sentrysearch.local_embedder import normalize_model_key
print(f'Qwen/Qwen3-VL-Embedding-8B -> {normalize_model_key("Qwen/Qwen3-VL-Embedding-8B")}')
print(f'qwen8b -> {normalize_model_key("qwen8b")}')
print(f'Qwen/Qwen3-VL-Embedding-2B -> {normalize_model_key("Qwen/Qwen3-VL-Embedding-2B")}')
print(f'custom/model -> {normalize_model_key("custom/model")}')