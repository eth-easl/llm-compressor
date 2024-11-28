import os
models = [
    'meta-llama/Llama-3.2-1B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct',
    'meta-llama/Llama-3.1-8B-Instruct',
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'meta-llama/Llama-2-7b-chat-hf',
    'meta-llama/Llama-2-13b-chat-hf',
]
schemes = [
    'W4A16',
    'W8A8_FP8',
    'W8A8_int8',
]
for model in models:
    for scheme in schemes:
        cmd = f"python src/cli/compress.py --model-id {model} --ds-id HuggingFaceH4/ultrachat_200k --ds-split train_sft --output-dir .outputs --org-id espressor --scheme {scheme}"
        cmd = f"ts -G 1 {cmd}"
        os.system(cmd)