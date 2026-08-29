#!/usr/bin/env python3
"""
scripts/evaluate.py
-------------------
Runs the full Ragas evaluation suite against the pipeline.

Expects a JSON file of test questions + ground-truth answers.
Format:
    [
      {
        "question": "What is the attention mechanism?",
        "ground_truth": "The attention mechanism allows..."
      },
      ...
    ]

Usage:
    python scripts/evaluate.py --test-file data/eval_questions.json
    python scripts/evaluate.py --test-file data/eval_questions.json --output results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console

from evaluation.ragas_eval import RagasEvaluator
from pipeline.graph import pipeline
from pipeline.state import initial_state
from config import cfg

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ragas evaluation on the GraphRAG pipeline")
    parser.add_argument(
        "--test-file",
        type=Path,
        required=True,
        help="JSON file with questions and ground-truth answers",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional: save results to a JSON file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to evaluate (useful for quick checks)",
    )
    return parser.parse_args()


def load_test_data(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Test file must be a JSON array of {question, ground_truth} objects")
    return data


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    cfg.validate()

    test_data = load_test_data(args.test_file)
    if args.limit:
        test_data = test_data[:args.limit]

    console.print(f"\n[bold cyan]Ragas Evaluation[/bold cyan] — {len(test_data)} questions\n")

    # Run pipeline for each question and collect results
    samples = []
    for i, item in enumerate(test_data):
        question    = item["question"]
        ground_truth = item["ground_truth"]

        console.print(f"  [{i+1}/{len(test_data)}] {question[:70]}…" if len(question) > 70 else f"  [{i+1}/{len(test_data)}] {question}")

        state = pipeline.invoke(initial_state(question))

        answer   = state.get("answer", "")
        contexts = state.get("graph_context", []) + state.get("vector_context", [])

        samples.append({
            "question":     question,
            "answer":       answer,
            "contexts":     contexts or ["No context retrieved"],
            "ground_truth": ground_truth,
        })

    console.print()
    console.print("[dim]Running Ragas metrics…[/dim]\n")

    evaluator = RagasEvaluator()
    scores = evaluator.evaluate(samples)
    evaluator.print_summary(scores)

    if args.output:
        out = {
            "scores": scores,
            "n_samples": len(samples),
            "test_file": str(args.test_file),
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        console.print(f"[green]Results saved to {args.output}[/green]")


if __name__ == "__main__":
    main()
