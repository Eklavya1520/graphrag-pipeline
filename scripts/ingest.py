#!/usr/bin/env python3
"""
scripts/ingest.py
-----------------
CLI entry point for Agent A — reads .txt files from a directory,
extracts entities/relationships, and loads them into Neo4j.

Usage:
    python scripts/ingest.py --data-dir data/raw
    python scripts/ingest.py --data-dir data/raw --build-vector  # also builds FAISS index
    python scripts/ingest.py --file data/raw/transformers.txt    # single file
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src/ to path so local imports work without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from agents.graph_builder import GraphBuilder
from config import cfg
from graph.neo4j_client import Neo4jClient
from vector.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")
console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the GraphRAG knowledge graph"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data-dir", type=Path, help="Directory of .txt files to ingest")
    group.add_argument("--file", type=Path, help="Single .txt file to ingest")
    parser.add_argument(
        "--build-vector",
        action="store_true",
        help="Also build/update the FAISS vector index",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the existing Neo4j graph before ingesting (destructive!)",
    )
    return parser.parse_args()


def collect_files(args: argparse.Namespace) -> list[Path]:
    if args.file:
        if not args.file.exists():
            console.print(f"[red]File not found: {args.file}[/red]")
            sys.exit(1)
        return [args.file]
    else:
        files = sorted(args.data_dir.glob("*.txt"))
        if not files:
            console.print(f"[yellow]No .txt files found in {args.data_dir}[/yellow]")
            sys.exit(0)
        return files


def main() -> None:
    args = parse_args()
    cfg.validate()

    files = collect_files(args)
    console.print(f"\n[bold cyan]GraphRAG Ingestion[/bold cyan] — {len(files)} file(s) to process\n")

    all_texts: list[str] = []
    total_entities = 0
    total_relations = 0

    with Neo4jClient() as client:
        if args.clear:
            console.print("[red]Clearing existing graph…[/red]")
            client.clear_graph()

        builder = GraphBuilder(client)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Ingesting…", total=len(files))

            for fp in files:
                progress.update(task, description=f"[cyan]{fp.name}[/cyan]")
                text = fp.read_text(encoding="utf-8", errors="ignore")
                all_texts.append(text)

                stats = builder.ingest_text(text, source_label=fp.name)
                total_entities += stats["entities"]
                total_relations += stats["relations"]

                progress.advance(task)

    console.print(f"\n[green]✓ Ingestion complete[/green]")
    console.print(f"  Entities written : {total_entities}")
    console.print(f"  Relations written: {total_relations}")

    # Also check graph size
    with Neo4jClient() as client:
        n_nodes = client.get_node_count()
        n_rels  = client.get_relationship_count()
    console.print(f"  Graph now has    : {n_nodes} nodes, {n_rels} relationships\n")

    # Optionally build vector index
    if args.build_vector:
        console.print("[cyan]Building FAISS vector index…[/cyan]")
        store = VectorStore()
        # Split texts into paragraph-level chunks for the vector store
        chunks: list[str] = []
        for text in all_texts:
            chunks.extend([p.strip() for p in text.split("\n\n") if len(p.strip()) > 80])
        store.add_documents(chunks)
        store.save()
        console.print(f"[green]✓ Vector index saved ({len(chunks)} chunks)[/green]\n")


if __name__ == "__main__":
    main()
