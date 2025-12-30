import torch

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import pandas as pd
from tqdm import tqdm
import yaml
import re
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import weave
import wandb
from datasets import load_dataset
from weave import Evaluation, Model
import weave
import asyncio


def extract_score(generated_text, num_classes):
    """
    Extracts the first digit found in the generated text that is a valid class label.
    """
    # Create pattern for valid class indices (0 to num_classes-1)
    valid_digits = "|".join(str(i) for i in range(num_classes))
    pattern = f"[{valid_digits}]"
    match = re.search(pattern, generated_text)
    if match:
        return int(match.group(0))
    else:
        return -1  # Parsing error


def compute_metrics(all_expected, all_predictions):
    """
    Compute overall evaluation metrics.

    Args:
        all_expected: List of expected labels
        all_predictions: List of predicted labels

    Returns:
        Dictionary containing accuracy, mse, mae, and confusion matrix
    """
    cm = confusion_matrix(all_expected, all_predictions)
    acc = accuracy_score(all_expected, all_predictions)
    mse = mean_squared_error(all_expected, all_predictions)
    mae = mean_absolute_error(all_expected, all_predictions)

    return {"confusion_matrix": cm, "accuracy": acc, "mse": mse, "mae": mae}


def print_metrics_summary(metrics, cfg):
    """
    Print a summary of all computed metrics to console.

    Args:
        metrics: Dictionary returned by compute_metrics()
        cfg: Configuration dictionary
    """
    cm = metrics["confusion_matrix"]
    acc = metrics["accuracy"]
    mse = metrics["mse"]
    mae = metrics["mae"]

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    labels = cfg["data"]["labels"]
    all_expected = []  # Will be populated by caller if needed
    all_predictions = []
    # Note: classification_report is called separately in plot_metrics_per_class

    print(f"\nOverall Accuracy: {acc:.4f}")
    print(f"Overall MSE: {mse:.4f}")
    print(f"Overall MAE: {mae:.4f}")
    print("=" * 50 + "\n")


def plot_confusion_matrix(cm, cfg):
    """
    Create and log confusion matrix heatmap.

    Args:
        cm: Confusion matrix from sklearn
        cfg: Configuration dictionary

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=cfg["data"]["labels"],
        yticklabels=cfg["data"]["labels"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    weave.log({"confusion_matrix": fig})
    plt.close()
    return fig


def plot_metrics_per_class(all_expected, all_predictions, cfg):
    """
    Create and log precision, recall, and F1-score per class.

    Args:
        all_expected: List of expected labels
        all_predictions: List of predicted labels
        cfg: Configuration dictionary

    Returns:
        Figure object
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        all_expected, all_predictions, labels=range(len(cfg["data"]["labels"]))
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(cfg["data"]["labels"]))
    width = 0.25

    ax.bar([i - width for i in x], precision, width, label="Precision", alpha=0.8)
    ax.bar([i for i in x], recall, width, label="Recall", alpha=0.8)
    ax.bar([i + width for i in x], f1, width, label="F1-Score", alpha=0.8)

    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Precision, Recall, and F1-Score per Class")
    ax.set_xticks(x)
    ax.set_xticklabels(cfg["data"]["labels"])
    ax.legend()
    ax.set_ylim([0, 1])
    plt.tight_layout()

    weave.log({"metrics_per_class": fig})
    plt.close()

    # Also print classification report
    print("\nClassification Report:")
    print(
        classification_report(
            all_expected, all_predictions, target_names=cfg["data"]["labels"]
        )
    )

    return fig


def plot_overall_metrics(metrics):
    """
    Create and log overall evaluation metrics bar chart.

    Args:
        metrics: Dictionary returned by compute_metrics()

    Returns:
        Figure object
    """
    metrics_names = ["Accuracy", "MSE", "MAE"]
    metrics_values = [metrics["accuracy"], metrics["mse"], metrics["mae"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(metrics_names, metrics_values, color=["green", "orange", "red"], alpha=0.7)
    ax.set_ylabel("Score")
    ax.set_title("Overall Evaluation Metrics")
    ax.set_ylim([0, max(metrics_values) * 1.2])

    for i, v in enumerate(metrics_values):
        ax.text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom")

    plt.tight_layout()
    weave.log({"overall_metrics": fig})
    plt.close()

    return fig


def plot_prediction_distribution(all_expected, all_predictions, cfg):
    """
    Create and log histogram comparing true vs predicted label distribution.

    Args:
        all_expected: List of expected labels
        all_predictions: List of predicted labels
        cfg: Configuration dictionary

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        all_expected,
        bins=len(cfg["data"]["labels"]),
        alpha=0.5,
        label="True Labels",
        edgecolor="black",
    )
    ax.hist(
        all_predictions,
        bins=len(cfg["data"]["labels"]),
        alpha=0.5,
        label="Predictions",
        edgecolor="black",
    )
    ax.set_xlabel("Label")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of True Labels vs Predictions")
    ax.legend()
    plt.tight_layout()
    weave.log({"prediction_distribution": fig})
    plt.close()

    return fig


def create_all_visualizations(all_expected, all_predictions, metrics, cfg):
    """
    Create and log all evaluation visualizations.

    Args:
        all_expected: List of expected labels
        all_predictions: List of predicted labels
        metrics: Dictionary returned by compute_metrics()
        cfg: Configuration dictionary
    """
    print("\n📊 Creating visualizations...")

    plot_confusion_matrix(metrics["confusion_matrix"], cfg)
    plot_metrics_per_class(all_expected, all_predictions, cfg)
    plot_overall_metrics(metrics)
    plot_prediction_distribution(all_expected, all_predictions, cfg)

    print("✅ All visualizations sent to Weave!")


@weave.op()
def exact_match_scorer(expected: int, output: dict) -> dict:
    """Scorer that checks if the predicted label exactly matches the expected label."""
    return {"match": expected == output["predicted_label"]}


@weave.op()
def accuracy_score_op(expected: int, output: dict) -> dict:
    """Computes accuracy (1 if match, 0 otherwise)"""
    pred = output["predicted_label"]
    return {"accuracy": 1 if expected == pred else 0}


@weave.op()
def mse_score_op(expected: int, output: dict) -> dict:
    """Computes Mean Squared Error between expected and predicted"""
    pred = output["predicted_label"]
    return {"mse": (expected - pred) ** 2}


@weave.op()
def mae_score_op(expected: int, output: dict) -> dict:
    """Computes Mean Absolute Error between expected and predicted"""
    pred = output["predicted_label"]
    return {"mae": abs(expected - pred)}


class DQIModel(Model):
    def __init__(self, cfg: dict):
        super().__init__()
        model, tokenizer_or_processor = FastLanguageModel.from_pretrained(
            model_name=cfg["training"]["output_dir"],
            max_seq_length=cfg["model"]["max_seq_length"],
            dtype=None,
            load_in_4bit=cfg["model"]["load_in_4bit"],
        )
        FastLanguageModel.for_inference(model)
        text_tokenizer = (
            tokenizer_or_processor.tokenizer
            if hasattr(tokenizer_or_processor, "tokenizer")
            else tokenizer_or_processor
        )
        chat_tokenizer = get_chat_template(text_tokenizer)

        self.cfg = cfg
        self.model = model
        self.text_tokenizer = text_tokenizer
        self.chat_tokenizer = chat_tokenizer
        self.labels = cfg["data"]["labels"]
        self.num_classes = len(self.labels)

    @weave.op()
    def predict(self, messages: list[dict]):
        prompt = self.chat_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.text_tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=4, use_cache=True)
        new_tokens = outputs[0, inputs.input_ids.shape[1] :]
        decoded = self.text_tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred = extract_score(decoded, self.num_classes)
        return {"generated_text": decoded, "predicted_label": pred}


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    weave.init("intro-example")

    model = DQIModel(cfg)
    # Load HuggingFace dataset and convert to list for Weave
    hf_dataset = load_dataset(
        "json", data_files=cfg["data"]["processed_test_path"], split="train"
    )
    # Convert to list of dicts for Weave Evaluation
    # Extract messages (without answer) and keep expected output separate
    dataset = [
        {
            "messages": example["messages"][:-1],
            "expected": int(example["messages"][-1]["content"]),
        }
        for example in hf_dataset
    ]

    evaluation = Evaluation(
        dataset=dataset,
        scorers=[exact_match_scorer, accuracy_score_op, mse_score_op, mae_score_op],
    )

    results = asyncio.run(evaluation.evaluate(model))

    # Extract predictions and expected labels
    all_expected = [example["expected"] for example in dataset]
    all_predictions = [result.output["predicted_label"] for result in results.results]

    # Compute metrics
    metrics = compute_metrics(all_expected, all_predictions)

    # Print summary
    print_metrics_summary(metrics, cfg)

    # Create and log all visualizations
    create_all_visualizations(all_expected, all_predictions, metrics, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    main(args)
