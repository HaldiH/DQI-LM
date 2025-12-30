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


def process_split(df, system_prompt_content, col_speech, col_label, output_path):
    formatted_data = []
    print(f"Formatting {len(df)} entries...")

    for _, row in df.iterrows():
        # Constructing the final text for the LLM in messages format
        grade = str(int(float(row[col_label])))
        speech_text = row[col_speech]

        formatted_data.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt_content},
                    {"role": "user", "content": f"Text to analyse: '{speech_text}'"},
                    {"role": "assistant", "content": grade},
                ]
            }
        )

    dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
    dataset.to_json(output_path)
    print(f"Success! Formatted dataset saved to: {output_path}")
    return formatted_data


def format_data(cfg):
    print(f"--- Preparing data ---")

    prompt_path = cfg["prompts"]["system_prompt_path"]
    print(f"Loading prompt from: {prompt_path}")
    system_prompt_content = load_prompt_content(prompt_path)

    col_label = cfg["data"]["col_label"]
    col_speech = cfg["data"]["col_speech"]

    # Load speeches from merged_debates file
    merged_debates_path = cfg["data"]["merged_debates_path"]
    print(f"Reading speeches from: {merged_debates_path}")
    df_speeches = pd.read_csv(merged_debates_path)
    print(f"Loaded {len(df_speeches)} speeches")

    # Process Train, Validation, and Test splits
    for split_name, split_key in [
        ("Train", "train"),
        ("Validation", "val"),
        ("Test", "test"),
    ]:
        input_path = cfg["data"][f"{split_key}_path"]
        output_path = cfg["data"][f"processed_{split_key}_path"]

        print(f"Reading {split_name} dataset: {input_path}")
        df_split = pd.read_csv(input_path)

        print(f"Merging {split_name} data with speeches...")
        df_split = df_split.merge(
            df_speeches[["speech_id", "speech"]],
            left_on="speech_id_1",
            right_on="speech_id",
            how="left",
        )
        print(f"{split_name} dataset after merge: {len(df_split)} rows")

        split_data = process_split(
            df_split,
            system_prompt_content,
            col_speech,
            col_label,
            output_path,
        )

        if split_name == "Train":
            print(
                f"Sample generated ({split_name}):\n{split_data[0]['messages'][:300]}..."
            )


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
