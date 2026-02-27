from kfp import dsl


@dsl.component(
    packages_to_install=["llmcompressor==0.9.0.2"],
)
def one_shot_model_compressor(
    model_id: str,
    quantization_scheme: str,
    quantization_ignore_list: list[str],
    dataset_id: str,
    dataset_split: str,
    output_model: dsl.Output[dsl.Artifact],
    num_calibration_samples: int = 512,
    max_sequence_length: int = 2048,
    seed: int = 42,
):
    """Compress a causal language model using one-shot quantization.

    Loads a Hugging Face causal LM and a calibration dataset, preprocesses and
    tokenizes the data, then applies one-shot quantization via llmcompressor.
    The compressed model and tokenizer are saved to the output artifact path.

    Args:
        model_id: Hugging Face model identifier (e.g. "meta-llama/Llama-3-8B").
        quantization_scheme: llmcompressor recipe(s) defining the compression strategy, e.g. W8A8, W4A16, NVFP4A16 etc.
        quantization_ignore_list: Layer names to exclude from quantization (e.g. ["lm_head"]).
        dataset_id: Hugging Face dataset identifier used for calibration.
        dataset_split: Dataset split to use (e.g. "train", "test").
        output_model: Output artifact where the compressed model and tokenizer
            are saved.
        num_calibration_samples: Number of dataset samples used for
            calibration (default: 512).
        max_sequence_length: Maximum token sequence length for truncation
            (default: 2048).
        seed: Random seed for dataset shuffling (default: 42).
    """
    from datasets import load_dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load dataset.
    ds = load_dataset(dataset_id, split=dataset_split)
    ds = ds.shuffle(seed=seed).select(range(num_calibration_samples))

    # Preprocess the data into the format the model is trained with.
    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }

    ds = ds.map(preprocess)

    # Tokenize the data (be careful with bos tokens -
    # we need add_special_tokens=False since the chat_template already added it).
    def tokenize(sample):
        return tokenizer(
            sample["text"], padding=False, max_length=max_sequence_length, truncation=True, add_special_tokens=False
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    # Load model.
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    recipe = QuantizationModifier(
        targets="Linear",
        scheme=quantization_scheme,
        ignore=quantization_ignore_list,
    )

    # Run one shot
    model = oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        tokenizer=tokenizer,
        max_seq_length=max_sequence_length,
        num_calibration_samples=num_calibration_samples,
    )

    # Save to output model
    model.save_pretrained(output_model.path)
    tokenizer.save_pretrained(output_model.path)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        one_shot_model_compressor,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
