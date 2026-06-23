"""
llm_client.py
-------------
Handles all communication with the Groq LLM API.
Responsibilities:
  - Build a RAG-aware system prompt
  - Inject retrieved context into the user prompt
  - Call Groq API (standard + streaming)
  - Return clean, structured responses

Groq is used for its free tier, low latency, and OpenAI-compatible API.
Model: llama-3.3-70b-versatile  →  GPT-4o level quality, 394 TPS, free tier.
"""

import os
import logging
from typing import List, Dict, Any, Generator

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL    = "llama-3.3-70b-versatile"  # GPT-4o level quality, 394 TPS, free tier
MAX_TOKENS    = 1024
TEMPERATURE   = 0.2   # low = more factual, consistent answers


# ── Prompt engineering ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful AI assistant for Spheretech, a cybersecurity software company based in Istanbul, Turkey.

Your role is to answer questions accurately using ONLY the context provided below. Follow these rules strictly:

1. Base your answer exclusively on the provided context. Do not use outside knowledge.
2. If the context does not contain enough information to answer, say: "I don't have enough information to answer that question. Please contact Spheretech support at support@spheretech.com.tr"
3. Be concise, professional, and friendly.
4. If the question is in Turkish, respond in Turkish. If in English, respond in English.
5. Never make up facts, prices, or product details.
6. When relevant, suggest contacting the Spheretech sales or support team for further assistance.
"""


def _build_user_prompt(query: str, context: str) -> str:
    """
    Inject retrieved context into the user prompt.

    Separating context from the question makes it easier for the LLM
    to distinguish what it knows from what it was told — reducing hallucination.
    """
    return f"""Context from Spheretech knowledge base:
---
{context}
---

User question: {query}

Please answer the question based strictly on the context above."""


# ── API calls ──────────────────────────────────────────────────────────────────

def _get_headers() -> Dict[str, str]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable is not set. "
            "Get a free key at https://console.groq.com"
        )
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }


def generate_answer(
    query: str,
    context: str,
    model: str = GROQ_MODEL,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
) -> Dict[str, Any]:
    """
    Send a RAG prompt to Groq and return the full response.

    Parameters
    ----------
    query       : original user question
    context     : formatted context block from retriever
    model       : Groq model identifier
    max_tokens  : maximum tokens in the response
    temperature : 0.0 = deterministic, 1.0 = creative

    Returns
    -------
    dict with keys: answer (str), model (str), usage (dict), success (bool)
    """
    logger.info(f"Calling Groq API | model={model} | query='{query[:60]}...'")

    payload = {
        "model":       model,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_prompt(query, context)},
        ],
    }

    try:
        response = requests.post(
            GROQ_API_URL,
            headers=_get_headers(),
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        answer = data["choices"][0]["message"]["content"].strip()
        usage  = data.get("usage", {})

        logger.info(
            f"Groq response received | "
            f"tokens: prompt={usage.get('prompt_tokens', '?')} "
            f"completion={usage.get('completion_tokens', '?')}"
        )

        return {
            "success": True,
            "answer":  answer,
            "model":   data.get("model", model),
            "usage":   usage,
        }

    except EnvironmentError as e:
        logger.error(f"Configuration error: {e}")
        return {
            "success": False,
            "answer":  str(e),
            "model":   model,
            "usage":   {},
        }
    except requests.exceptions.Timeout:
        logger.error("Groq API request timed out.")
        return {
            "success": False,
            "answer":  "The request timed out. Please try again.",
            "model":   model,
            "usage":   {},
        }
    except requests.exceptions.HTTPError as e:
        logger.error(f"Groq API HTTP error: {e}")
        return {
            "success": False,
            "answer":  f"API error: {e}. Please check your GROQ_API_KEY.",
            "model":   model,
            "usage":   {},
        }
    except Exception as e:
        logger.error(f"Unexpected error calling Groq API: {e}")
        return {
            "success": False,
            "answer":  "An unexpected error occurred. Please try again.",
            "model":   model,
            "usage":   {},
        }


def generate_answer_stream(
    query: str,
    context: str,
    model: str = GROQ_MODEL,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
) -> Generator[str, None, None]:
    """
    Stream the Groq response token by token.
    Yields text chunks as they arrive — enables word-by-word UI rendering.

    Usage (Streamlit example)
    -------------------------
        for chunk in generate_answer_stream(query, context):
            st.write(chunk)
    """
    logger.info(f"Streaming Groq API | model={model}")

    payload = {
        "model":       model,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "stream":      True,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_prompt(query, context)},
        ],
    }

    try:
        with requests.post(
            GROQ_API_URL,
            headers=_get_headers(),
            json=payload,
            stream=True,
            timeout=60,
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                decoded = line.decode("utf-8")

                # SSE lines start with "data: "
                if not decoded.startswith("data: "):
                    continue

                data_str = decoded[len("data: "):]

                # Stream end signal
                if data_str.strip() == "[DONE]":
                    break

                import json
                data = json.loads(data_str)
                delta = data["choices"][0].get("delta", {})
                token = delta.get("content", "")

                if token:
                    yield token

    except requests.exceptions.HTTPError as e:
        logger.error(f"Groq streaming HTTP error: {e}")
        yield f"\n\n[Error: {e}]"
    except Exception as e:
        logger.error(f"Unexpected streaming error: {e}")
        yield "\n\n[An unexpected error occurred]"


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Mock test — verifies prompt building without a real API key

    test_query = "What is SphereShield and how is it priced?"
    test_context = """[1] Category: Products
    Q: What is SphereShield?
    A: SphereShield is Spheretech's flagship EDR platform with AI-powered threat detection.
    Score: 0.9821

[2] Category: Pricing and Licensing
    Q: How is SphereShield priced?
    A: SphereShield starts at $15 per endpoint per month with volume discounts available.
    Score: 0.9544"""

    # ── Test prompt building (no API key needed) ──
    user_prompt = _build_user_prompt(test_query, test_context)

    assert "Context from Spheretech knowledge base" in user_prompt
    assert test_query in user_prompt
    assert test_context in user_prompt
    print("_build_user_prompt()     → structure correct")

    assert "Turkish" in SYSTEM_PROMPT
    assert "hallucin" not in SYSTEM_PROMPT.lower()   # no hallucination encouragement
    print("SYSTEM_PROMPT            → multilingual + grounded rules present")

    # ── Test error handling when API key is missing ──
    original_key = os.environ.pop("GROQ_API_KEY", None)
    result = generate_answer(test_query, test_context)
    assert result["success"] is False
    assert "GROQ_API_KEY" in result["answer"]
    print("Missing API key          → graceful error, no crash")

    if original_key:
        os.environ["GROQ_API_KEY"] = original_key

    # ── Test generate_answer with real key (if set) ──
    if os.getenv("GROQ_API_KEY"):
        print("\nGROQ_API_KEY found — testing live API call...")
        result = generate_answer(test_query, test_context)
        if result["success"]:
            print(f"✅ Live API call            → success")
            print(f"   Model                   : {result['model']}")
            print(f"   Tokens used             : {result['usage']}")
            print(f"   Answer preview          : {result['answer'][:150]}...")
        else:
            print(f"Live API call failed    : {result['answer']}")
    else:
        print("\n  Set GROQ_API_KEY to test live API call.")

    print("\n llm_client.py smoke test completed.")