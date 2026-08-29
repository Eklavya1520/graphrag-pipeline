"""
answer_gen.py
-------------
LLM-based answer generation node.

Two responsibilities:
  1. generate_answer() — synthesises a final answer from graph + vector context
  2. score_confidence() — asks the LLM to self-assess its answer (0.0 – 1.0)

The confidence score drives the self-correction loop in the pipeline edges.
It is a known limitation that LLM self-scoring is not perfectly calibrated —
a proper solution would use a trained uncertainty estimator.  For now, this
works well enough for the self-correction loop to catch obvious failures.
"""

import logging
import re
import textwrap

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import cfg

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

ANSWER_SYSTEM_PROMPT = textwrap.dedent("""
    You are a knowledgeable assistant that answers questions using the provided context.

    Context comes from two sources:
    - GRAPH CONTEXT: structured entity-relationship data from a knowledge graph
    - VECTOR CONTEXT: relevant text passages retrieved by semantic similarity

    Instructions:
    - Ground your answer strictly in the provided context.
    - If the context is insufficient, say so — do not hallucinate facts.
    - Be concise and specific. Bullet points are fine for multi-part answers.
    - If you reference a relationship from the graph, mention it explicitly.
""").strip()

CRITIC_PROMPT = textwrap.dedent("""
    You are evaluating the quality of an AI-generated answer to a user question.

    Question: {question}
    Answer: {answer}
    Context used: {context_summary}

    Rate the answer on a scale from 0.0 to 1.0 based on:
    - Faithfulness: Is it grounded in the context? (not hallucinated)
    - Relevance: Does it directly address the question?
    - Completeness: Does it cover the key aspects?

    Respond with ONLY a single floating point number between 0.0 and 1.0.
    Example: 0.75
""").strip()


def _get_llm(temperature: float = 0.3):
    if cfg.llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=cfg.llm_model,
            google_api_key=cfg.google_api_key,
            temperature=temperature,
        )
    elif cfg.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=cfg.llm_model,
            anthropic_api_key=cfg.anthropic_api_key,
            temperature=temperature,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=cfg.llm_model,
            openai_api_key=cfg.openai_api_key,
            temperature=temperature,
        )


def _format_context(graph_context: list[str], vector_context: list[str]) -> str:
    """Build the context block fed into the answer prompt."""
    parts = []

    if graph_context:
        parts.append("=== GRAPH CONTEXT ===")
        parts.extend(f"  • {item}" for item in graph_context)

    if vector_context:
        parts.append("\n=== VECTOR CONTEXT ===")
        parts.extend(f"  • {item}" for item in vector_context)

    if not parts:
        return "No relevant context was retrieved."

    return "\n".join(parts)


class AnswerGenerator:
    """
    Handles answer generation and confidence scoring.

    Designed to be stateless so it can be used from any LangGraph node
    without worrying about shared state.
    """

    def __init__(self) -> None:
        self._llm = _get_llm(temperature=0.3)
        self._critic = _get_llm(temperature=0.0)  # deterministic for scoring

    def generate(
        self,
        question: str,
        graph_context: list[str],
        vector_context: list[str],
        conversation_history: list | None = None,
    ) -> str:
        """
        Generate an answer grounded in the provided context.

        Parameters
        ----------
        question            : the (rewritten) question to answer
        graph_context       : list of context strings from Neo4j
        vector_context      : list of context strings from vector fallback
        conversation_history: prior LangChain message objects (for multi-turn)

        Returns
        -------
        answer string
        """
        context_block = _format_context(graph_context, vector_context)

        user_content = (
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        messages = [SystemMessage(content=ANSWER_SYSTEM_PROMPT)]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append(HumanMessage(content=user_content))

        response = self._llm.invoke(messages)
        answer = response.content.strip()
        logger.debug("Generated answer (%d chars)", len(answer))
        return answer

    def score_confidence(
        self,
        question: str,
        answer: str,
        graph_context: list[str],
        vector_context: list[str],
    ) -> float:
        """
        Ask the LLM to score its own answer.  Returns float in [0.0, 1.0].
        Falls back to 0.5 if parsing fails.
        """
        context_summary = f"{len(graph_context)} graph items, {len(vector_context)} vector items"

        prompt = CRITIC_PROMPT.format(
            question=question,
            answer=answer[:500],  # truncate long answers to keep prompt short
            context_summary=context_summary,
        )

        try:
            response = self._critic.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()
            # Extract first float-like value from the response
            match = re.search(r"\d+\.\d+|\d+", raw)
            if match:
                score = float(match.group())
                score = max(0.0, min(1.0, score))  # clamp
                logger.debug("Confidence score: %.2f", score)
                return score
        except Exception as e:
            logger.warning("Confidence scoring failed: %s", e)

        return 0.5  # neutral fallback
