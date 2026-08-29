"""
query_rewriter.py  —  Agent B
------------------------------
Takes a raw (possibly vague) user query and rewrites it into a cleaner,
more specific form. Also extracts entity keyword hints used to seed Cypher queries.

Why this is necessary
---------------------
Users ask things like:
  "Tell me about transformer"          → too vague, no Cypher hook
  "how does bert work"                 → lowercase, colloquial
  "what's the difference attention vs mlp" → incomplete phrasing

Agent B normalises these before the graph retriever and answer generator
ever see them, which dramatically improves retrieval precision.
"""

import logging
import re
import textwrap
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from config import cfg

logger = logging.getLogger(__name__)


@dataclass
class RewriteResult:
    rewritten_query: str
    entity_keywords: list[str]
    query_type: str  # "factual" | "relational" | "comparative" | "procedural"


SYSTEM_PROMPT = textwrap.dedent("""
    You are a query understanding and rewriting assistant for a knowledge-graph-based QA system.

    Given a raw user question, you must:
    1. Rewrite it into a clear, specific, grammatically complete question.
    2. Extract 1–4 entity keywords (proper nouns, technical terms) to use as graph search seeds.
    3. Classify the query type as one of: factual | relational | comparative | procedural

    Respond in this exact format (no extra text):
    REWRITTEN: <the improved question>
    KEYWORDS: <comma-separated keywords>
    TYPE: <query type>

    Examples
    --------
    Input:  tell me about bert
    REWRITTEN: What is BERT, how does it work, and what NLP tasks is it used for?
    KEYWORDS: BERT, transformer, pre-training
    TYPE: factual

    Input:  difference between rnn and transformer
    REWRITTEN: What are the key architectural and performance differences between RNNs and Transformer models?
    KEYWORDS: RNN, Transformer, sequence modeling
    TYPE: comparative

    Input:  how to train a language model
    REWRITTEN: What are the steps and techniques involved in training a large language model from scratch?
    KEYWORDS: language model, training, pre-training
    TYPE: procedural
""").strip()


def _get_llm():
    if cfg.llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=cfg.llm_model,
            google_api_key=cfg.google_api_key,
            temperature=0.2,
        )
    elif cfg.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=cfg.llm_model,
            anthropic_api_key=cfg.anthropic_api_key,
            temperature=0.2,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=cfg.llm_model,
            openai_api_key=cfg.openai_api_key,
            temperature=0.2,
        )


def _parse_rewrite_output(raw: str, original_query: str) -> RewriteResult:
    """Parse the structured REWRITTEN/KEYWORDS/TYPE format from the LLM."""
    rewritten = original_query  # safe fallback
    keywords: list[str] = []
    query_type = "factual"

    for line in raw.strip().splitlines():
        line = line.strip()
        if line.startswith("REWRITTEN:"):
            rewritten = line[len("REWRITTEN:"):].strip()
        elif line.startswith("KEYWORDS:"):
            raw_kws = line[len("KEYWORDS:"):].strip()
            keywords = [k.strip() for k in raw_kws.split(",") if k.strip()]
        elif line.startswith("TYPE:"):
            query_type = line[len("TYPE:"):].strip().lower()

    # If keywords were not parsed, fall back to simple noun extraction from the rewrite
    if not keywords:
        keywords = _fallback_keyword_extraction(rewritten)

    return RewriteResult(
        rewritten_query=rewritten,
        entity_keywords=keywords[:4],  # cap at 4
        query_type=query_type,
    )


def _fallback_keyword_extraction(text: str) -> list[str]:
    """
    Very naive keyword extraction — just takes capitalised words / acronyms.
    Used when the LLM doesn't return keywords in the expected format.
    """
    tokens = re.findall(r"\b[A-Z][a-zA-Z]+\b|\b[A-Z]{2,}\b", text)
    # Filter common stopwords that happen to be capitalised
    stopwords = {"What", "How", "Why", "When", "Where", "Which", "The", "A", "An", "Is", "Are", "Does"}
    return [t for t in tokens if t not in stopwords][:4]


class QueryRewriter:
    """
    Agent B — rewrites user queries for better graph retrieval.

    Usage:
        rewriter = QueryRewriter()
        result = rewriter.rewrite("tell me about attention")
        print(result.rewritten_query)
        print(result.entity_keywords)
    """

    def __init__(self) -> None:
        self._llm = _get_llm()

    def rewrite(self, query: str) -> RewriteResult:
        """
        Rewrite the query and return a RewriteResult.
        If the LLM call fails for any reason, returns the original query unchanged
        (fail-open behaviour to keep the pipeline running).
        """
        if not query or not query.strip():
            return RewriteResult(
                rewritten_query=query,
                entity_keywords=[],
                query_type="factual",
            )

        try:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Input: {query.strip()}"),
            ]
            response = self._llm.invoke(messages)
            result = _parse_rewrite_output(response.content, query)
            logger.info(
                "Query rewritten: '%s' -> '%s' | keywords=%s | type=%s",
                query[:60], result.rewritten_query[:60],
                result.entity_keywords, result.query_type,
            )
            return result
        except Exception as e:
            logger.error("QueryRewriter failed: %s — using original query", e)
            return RewriteResult(
                rewritten_query=query,
                entity_keywords=_fallback_keyword_extraction(query),
                query_type="factual",
            )
