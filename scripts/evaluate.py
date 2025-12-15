import torch
from unsloth import FastLanguageModel
import pandas as pd
from tqdm import tqdm
import yaml
import re
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse


def extract_score(generated_text):
    """
    Cleans the LLM response to keep only the digit.
    Searches for the last digit generated after [/INST].
    """
    # Cut to keep only the response (after [/INST])
    response_part = generated_text.split("[/INST]")[-1]

    # Search for the first digit that appears (0, 1, 2 or 3)
    match = re.search(r"[0-3]", response_part)
    if match:
        return int(match.group(0))
    else:
        return -1  # Parsing error


def evaluate(config_path):
    # --- CONFIGURATION ---
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load system prompt
    with open(cfg["prompts"]["system_prompt_path"], "r") as f:
        system_prompt = f.read().strip()

    TEMPLATE = cfg["prompts"]["format_template"]

    # 1. Load fine-tuned model
    model_path = cfg["training"]["output_dir"]

    print(f"Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)  # Enable inference mode (faster)

    # 2. Load Test Set
    test_path = cfg["data"]["test_path"]
    df_test = pd.read_csv(test_path)
    y_true = df_test["label"].tolist()
    y_pred = []

    print("Starting inference on Test Set...")

    # 3. Prediction loop
    for text in tqdm(df_test["text"]):
        # Prepare prompt
        prompt = TEMPLATE.format(
            system_prompt=system_prompt,
            input_text=text,
            output_score="",  # Leave empty for model completion
        )

        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        # Generation
        outputs = model.generate(**inputs, max_new_tokens=4, use_cache=True)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        pred = extract_score(decoded)
        y_pred.append(pred)

    # 4. Calculate metrics
    # Filter parsing errors (-1)
    valid_indices = [i for i, x in enumerate(y_pred) if x != -1]
    y_true_clean = [y_true[i] for i in valid_indices]
    y_pred_clean = [y_pred[i] for i in valid_indices]

    parsing_errors = len(y_pred) - len(valid_indices)
    print(f"\n--- RESULTS ---")
    print(f"LLM formatting errors : {parsing_errors}/{len(y_pred)}")
    print(f"Accuracy : {accuracy_score(y_true_clean, y_pred_clean):.4f}")
    print(f"Mean Absolute Error : {mean_absolute_error(y_true_clean, y_pred_clean):.4f}")

    print("\nClassification Report :")
    print(
        classification_report(
            y_true_clean,
            y_pred_clean,
            target_names=["Level 0", "Level 1", "Level 2", "Level 3"],
        )
    )

    # 5. Visualization : Confusion Matrix
    cm = confusion_matrix(y_true_clean, y_pred_clean, labels=[0, 1, 2, 3])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1", "Pred 2", "Pred 3"],
        yticklabels=["True 0", "True 1", "True 2", "True 3"],
    )
    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix - DQI Justification")

    # Save graph
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/confusion_matrix.png")
    print("\nGraph saved to 'results/confusion_matrix.png'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    evaluate(args.config)
