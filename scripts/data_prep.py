import pandas as pd
import yaml
import os
import argparse
from datasets import Dataset


def load_prompt_content(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file does not exist: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def format_data(cfg):
    print(f"--- Preparing data ---")

    input_path = cfg["data"]["input_path"]
    print(f"Reading dataset: {input_path}")
    df = pd.read_csv(input_path)

    prompt_path = cfg["prompts"]["system_prompt_path"]
    print(f"Loading prompt from: {prompt_path}")
    system_prompt_content = load_prompt_content(prompt_path)

    template = cfg["prompts"]["format_template"]
    col_text = cfg["data"]["col_text"]
    col_label = cfg["data"]["col_label"]

    formatted_data = []
    print("Formatting entries...")
    for _, row in df.iterrows():
        # Constructing the final text for the LLM
        full_text = template.format(
            system_prompt=system_prompt_content,
            input_text=row[col_text],
            output_score=str(row[col_label]),
        )
        formatted_data.append({"text": full_text})

    output_path = cfg["data"]["processed_path"]
    dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
    dataset.to_json(output_path)
    print(f"Success! Formatted dataset saved to: {output_path}")
    print(f"Sample generated:\n{formatted_data[0]['text'][:300]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for training.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    format_data(config)
