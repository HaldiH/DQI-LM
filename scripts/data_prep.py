import pandas as pd
import yaml
import os
from datasets import Dataset

config_path = "configs/config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

def load_prompt_content(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file does not exist: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def format_data():
    print(f"--- Preparing data ---")
    
    # 1. Read the CSV
    input_path = config['data']['input_path']
    print(f"Reading dataset: {input_path}")
    df = pd.read_csv(input_path)
    
    # 2. Read the System Prompt from the text file
    prompt_path = config['prompts']['system_prompt_path']
    print(f"Loading prompt from: {prompt_path}")
    system_prompt_content = load_prompt_content(prompt_path)
    
    # 3. Retrieve the template and column names
    template = config['prompts']['format_template']
    col_text = config['data']['col_text']
    col_label = config['data']['col_label']
    
    # 4. Formatting
    formatted_data = []
    print("Formatting entries...")
    for _, row in df.iterrows():
        # Constructing the final text for the LLM
        full_text = template.format(
            system_prompt=system_prompt_content,
            input_text=row[col_text],
            output_score=str(row[col_label])
        )
        formatted_data.append({"text": full_text})
    
    # 5. Saving
    output_path = config['data']['processed_path']
    dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
    dataset.to_json(output_path)
    print(f"Success! Formatted dataset saved to: {output_path}")
    print(f"Sample generated:\n{formatted_data[0]['text'][:300]}...")

if __name__ == "__main__":
    format_data()