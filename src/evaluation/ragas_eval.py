"""
ragas_eval.py
-------------
Evaluates the GraphRAG pipeline using Ragas metrics.

Ragas provides model-based evaluation metrics for RAG systems:
  - faithfulness     : is the answer grounded in the retrieved context?
  - answer_relevancy : does the answer address the question?
  - context_recall   : did retrieval cover the ground-truth answer?
  - context_precision: how much of the retrieved context was actually useful?

Usage
-----
    from evaluation.ragas_eval import RagasEvaluator

    evaluator = RagasEvaluator()
    results = evaluator.evaluate(samples)
    evaluator.print_summary(results)

Input format (samples)
----------------------
Each sample must be a dict with keys:
  - question    : str
  - answer      : str  (from the pipeline)
  - contexts    : list[str]  (retrieved context strings)
  - ground_truth: str  (reference answer — needed for context_recall)
"""

import logging
from typing import Any

from config import cfg

logger = logging.getLogger(__name__)


def _get_ragas_llm():
    """Return the LLM ragas will use internally for its own scoring."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=cfg.ragas_llm,
        openai_api_key=cfg.openai_api_key,
        temperature=0.0,
    )


class RagasEvaluator:
    """
    Wraps Ragas evaluate() with sensible defaults and pretty output.

    Notes:
    - Ragas >= 0.1.x uses EvaluationDataset + evaluate()
    - It internally calls the LLM for faithfulness / answer_relevancy scoring
    - Keep test sets to ~50 questions max to avoid runaway API costs
    """

    def __init__(self) -> None:
        self._check_ragas_installed()

    @staticmethod
    def _check_ragas_installed() -> None:
        try:
            import ragas  # noqa: F401
        except ImportError:
            raise ImportError(
                "ragas is not installed. Run: pip install ragas"
            )

    def evaluate(
        self,
        samples: list[dict[str, Any]],
        metrics: list | None = None,
    ) -> dict[str, float]:
        """
        Run Ragas evaluation over a list of sample dicts.

        Parameters
        ----------
        samples : list of dicts with keys: question, answer, contexts, ground_truth
        metrics : optional list of ragas Metric objects (defaults to all 4)

        Returns
        -------
        dict mapping metric name -> mean score
        """
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
        from datasets import Dataset

        _metrics = metrics or [
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ]

        # Validate samples
        required_keys = {"question", "answer", "contexts", "ground_truth"}
        for i, s in enumerate(samples):
            missing = required_keys - set(s.keys())
            if missing:
                raise ValueError(f"Sample {i} is missing keys: {missing}")

        # Convert to HuggingFace Dataset format (Ragas < 0.2 API)
        dataset = Dataset.from_list(samples)

        logger.info(
            "Running Ragas evaluation on %d samples with metrics: %s",
            len(samples),
            [m.name for m in _metrics],
        )

        result = evaluate(dataset=dataset, metrics=_metrics)

        scores = {m.name: float(result[m.name]) for m in _metrics}
        logger.info("Ragas scores: %s", scores)
        return scores

    def evaluate_from_pipeline(
        self,
        questions: list[str],
        ground_truths: list[str],
        pipeline_fn,
    ) -> dict[str, float]:
        """
        Convenience method: runs each question through the pipeline, collects
        context + answer, then evaluates with Ragas.

        Parameters
        ----------
        questions      : list of question strings
        ground_truths  : list of reference answer strings (same order)
        pipeline_fn    : callable(question: str) -> PipelineState dict
        """
        from pipeline.state import initial_state

        samples = []
        for q, gt in zip(questions, ground_truths):
            logger.info("Running pipeline for: '%s'", q[:60])
            state = pipeline_fn(initial_state(q))

            answer   = state.get("answer", "")
            contexts = state.get("graph_context", []) + state.get("vector_context", [])

            samples.append({
                "question":     q,
                "answer":       answer,
                "contexts":     contexts or ["No context retrieved"],
                "ground_truth": gt,
            })

        return self.evaluate(samples)

    @staticmethod
    def print_summary(scores: dict[str, float]) -> None:
        """Pretty-print a scores dict."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title="Ragas Evaluation Results", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Score", justify="right", style="bold")
            table.add_column("Rating", justify="center")

            def _rating(s: float) -> str:
                if s >= 0.85: return "[green]Excellent[/green]"
                if s >= 0.70: return "[yellow]Good[/yellow]"
                if s >= 0.55: return "[orange3]Fair[/orange3]"
                return "[red]Poor[/red]"

            for metric, score in scores.items():
                table.add_row(metric, f"{score:.4f}", _rating(score))

            console.print(table)

        except ImportError:
            # Fallback if rich is not installed
            print("\n=== Ragas Evaluation Results ===")
            for metric, score in scores.items():
                print(f"  {metric:<25} {score:.4f}")
            print("================================\n")
