#!/usr/bin/env python3
"""
scripts/query.py
----------------
CLI entry point for running a single question through the GraphRAG pipeline.

Usage:
    python scripts/query.py --question "What is the attention mechanism?"
    python scripts/query.py -q "How does BERT differ from GPT?"
    python scripts/query.py --question "..." --verbose  # show full state trace
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from pipeline.graph import pipeline
from pipeline.state import initial_state
from config import cfg

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the GraphRAG pipeline")
    parser.add_argument(
        "-q", "--question",
        type=str,
        required=True,
        help="The question to answer",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print intermediate state after each node",
    )
    return parser.parse_args()


def display_result(state: dict, elapsed: float, verbose: bool) -> None:
    """Pretty-print the final pipeline state."""
    console.print()
    console.print(Rule("[bold cyan]GraphRAG Pipeline Result[/bold cyan]"))
    console.print()

    # Original vs rewritten query
    if state.get("rewritten_query") and state["rewritten_query"] != state["query"]:
        console.print(f"[dim]Original query :[/dim] {state['query']}")
        console.print(f"[dim]Rewritten query:[/dim] {state['rewritten_query']}")
        console.print()

    # Graph context used
    graph_ctx = state.get("graph_context", [])
    vector_ctx = state.get("vector_context", [])
    if verbose and graph_ctx:
        console.print("[bold]Graph context retrieved:[/bold]")
        for item in graph_ctx:
            console.print(f"  [cyan]•[/cyan] {item}")
        console.print()
    if verbose and vector_ctx:
        console.print("[bold]Vector context retrieved:[/bold]")
        for item in vector_ctx:
            console.print(f"  [yellow]•[/yellow] {item[:120]}…" if len(item) > 120 else f"  [yellow]•[/yellow] {item}")
        console.print()

    # Answer
    answer = state.get("answer", "No answer generated.")
    panel = Panel(
        Text(answer),
        title="[bold green]Answer[/bold green]",
        border_style="green",
    )
    console.print(panel)
    console.print()

    # Metadata
    confidence = state.get("confidence", 0.0)
    iterations = state.get("iterations", 0)
    low_conf   = state.get("low_confidence_warning", False)

    conf_color = "green" if confidence >= cfg.confidence_threshold else "red"
    console.print(
        f"[dim]Confidence:[/dim] [{conf_color}]{confidence:.2f}[/{conf_color}]  "
        f"[dim]|[/dim]  [dim]Iterations:[/dim] {iterations}  "
        f"[dim]|[/dim]  [dim]Time:[/dim] {elapsed:.1f}s"
    )
    if low_conf:
        console.print("[yellow]⚠  Low confidence warning — context may be insufficient[/yellow]")
    console.print()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg.validate()

    console.print(f"\n[bold]Question:[/bold] {args.question}\n")
    console.print("[dim]Running pipeline…[/dim]")

    start = time.time()
    state = pipeline.invoke(initial_state(args.question))
    elapsed = time.time() - start

    display_result(state, elapsed, args.verbose)


if __name__ == "__main__":
    main()
