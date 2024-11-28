import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from src.cli.utils import generate_readme

def upload_and_delete(org_id, model_id, local_path):
    cmd = f"huggingface-cli upload {org_id}/{model_id} {local_path} --repo-type model"
    os.system(cmd)
    os.system(f"rm -rf {local_path}")

def get_max_sequence_length(config):
    if "max_position_embeddings" in config:
        return config.max_position_embeddings
    else:
        raise ValueError("Could not determine maximum sequence length from model configuration")

def oneshot_with_recipe(
        model,
        ds,
        recipe, 
        max_seq_len,
        num_calibration_samples
    ):
    if recipe=='W4A16':
        recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
        oneshot(
            model=model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=max_seq_len,
            num_calibration_samples=num_calibration_samples
        )
    elif recipe == 'W8A8_FP8':
        recipe = QuantizationModifier(
            targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
        )
        oneshot(model=model, recipe=recipe)
    elif recipe == 'W8A8_int8':
        recipe = [
            SmoothQuantModifier(smoothing_strength=0.8),
            GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
        ]
        oneshot(
            model=model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=max_seq_len,
            num_calibration_samples=num_calibration_samples,
        )
    else:
        raise ValueError(f"Invalid recipe. Expected one of ['W4A16', 'W8A8_FP8', 'W8A8_int8'] but got {recipe}")
    return model
    
def compress(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    ds = load_dataset(args.ds_id, split=args.ds_split)
    ds  = ds.shuffle(seed=42).select(range(args.n_samples))
    if args.preprocessor == "chat":
        def preprocess(example):
            return {
                "text": tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                )
            }
        
    ds = ds.map(preprocess)
    if args.max_seq_length == -1:
        max_length = get_max_sequence_length(model.config)
    else:
        max_length = args.max_seq_length
    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )
    ds = ds.map(tokenize, remove_columns=ds.column_names)
    oneshot_with_recipe(
        model=model,
        ds=ds,
        recipe=args.scheme,
        max_seq_len=max_length,
        num_calibration_samples=args.n_samples
    )
    SAVE_DIR = args.model_id.replace("/", ".") + f"_{args.scheme}"
    SAVE_DIR = os.path.join(args.output_dir, SAVE_DIR)
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)
    try:
        message = [{"role":"user", "content":"Who is Alan Turing?"}]
        prompt = tokenizer.apply_chat_template(message, tokenize=False)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        output = model.generate(input_ids, max_new_tokens=128)
        output = tokenizer.decode(output[0])
    except Exception as e:
        prompt = "Could not generate output"
        output = str(e)
    
    readme = generate_readme({
        "model_id": args.model_id,
        "scheme": args.scheme,
        "dataset_id": args.ds_id,
        "dataset_split": args.ds_split,
        "n_samples": args.n_samples,
        "preprocessor": args.preprocessor,
        "max_seq_length": max_length,
        "prompt": prompt,
        "output": output   
    })
    with open(os.path.join(SAVE_DIR, "README.md"), "w") as f:
        f.write(readme)
    new_model_id = args.model_id.replace("/", ".") + f"_{args.scheme}"
    upload_and_delete(args.org_id, new_model_id, SAVE_DIR)
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compress a model")
    parser.add_argument("--model-id", type=str, help="Model ID to compress")
    parser.add_argument("--ds-id", type=str, help="Dataset ID to use for calibration")
    parser.add_argument("--ds-split", type=str, help="Dataset split to use for calibration", default="train")
    parser.add_argument("--n-samples", type=int, help="Number of samples to use for calibration", default=512)
    parser.add_argument("--max-seq-length", type=int, help="Maximum sequence length", default=-1)
    parser.add_argument("--preprocessor", type=str, help="Preprocessor to use", default="chat")
    parser.add_argument("--scheme", type=str, help="Quantization scheme to use", default="W4A16", choices=['W4A16', 'W8A8_FP8', 'W8A8_int8'])
    parser.add_argument("--output-dir", type=str, help="Output directory to save compressed model")
    parser.add_argument("--org-id", type=str, help="Organization ID to upload compressed model")
    compress(parser.parse_args())