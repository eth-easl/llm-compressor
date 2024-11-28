template = """This is a compressed model using [llmcompressor](https://github.com/vllm-project/llm-compressor).

## Compression Configuration

Base Model: {model_id}
Compression Scheme: {scheme}
Dataset: {dataset_id}
Dataset Split: {dataset_split}
Number of Samples: {n_samples}
Preprocessor: {preprocessor}
Maximum Sequence Length: {max_seq_length}

## Sample Output

#### Prompt: 

```
{prompt}
```

#### Output: 

```
{output}
```

## Evaluation

<TODO>

"""

def generate_readme(config) -> str:
    readme = template.format(
        model_id=config['model_id'],
        scheme=config['scheme'],
        dataset_id=config['dataset_id'],
        dataset_split=config['dataset_split'],
        n_samples=config['n_samples'],
        preprocessor=config['preprocessor'],
        max_seq_length=config['max_seq_length'],
        prompt=config['prompt'],
        output=config['output']
    )
    return readme