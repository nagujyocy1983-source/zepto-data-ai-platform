import json
import os
from typing import Any, TypedDict

import chromadb
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import END, START, StateGraph


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

COLLECTION_NAME = "zepto_policy_documents"

# Default graded mode is fully offline/mock.
MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"

# Optional real-LLM configuration.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.1-8b-instant",
)
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


# ============================================================
# EMBEDDING MODEL
# ============================================================

print("\n" + "=" * 60)
print("INITIALIZING ZEPT0 SUPPORT ASSISTANT")
print("=" * 60)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)

print(
    f"ChromaDB collection loaded: "
    f"{COLLECTION_NAME}"
)

print(
    f"ChromaDB document count: "
    f"{collection.count()}"
)


# ============================================================
# PYDANTIC REQUEST / RESPONSE MODELS
# ============================================================

class AskRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="User question"
    )


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0
    )


# ============================================================
# LANGGRAPH STATE
# ============================================================

class SupportState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_chunks: list[dict[str, Any]]
    answer: str
    sources: list[str]
    confidence: float


# ============================================================
# STRUCTURED PROMPT TEMPLATE
# ============================================================

STRUCTURED_PROMPT_TEMPLATE = """
ROLE:
You are a Zepto customer-support policy assistant.

CONTEXT:
You will receive policy information retrieved from Zepto's approved
policy document corpus.

TASK:
Answer the customer's question using only the supplied context.

FORMAT:
Return valid JSON with exactly these fields:
{
  "answer": "string",
  "sources": ["document_id"],
  "confidence": 0.0
}

LENGTH:
Keep the answer concise and directly relevant to the customer's question.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context.
Do not invent, assume, or add policy details.

FEW-SHOT EXAMPLE:
Question: "What is the delivery fee for an order below INR 149?"
Context: "Orders below INR 149 incur a flat INR 25 delivery fee."
Answer:
{
  "answer": "Orders below INR 149 incur a flat INR 25 delivery fee.",
  "sources": ["doc_01"],
  "confidence": 1.0
}

CUSTOMER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}
"""


# ============================================================
# MOCK MODE
# ============================================================

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


def mock_mode_enabled() -> bool:
    return MOCK_LLM


# ============================================================
# OPTIONAL REAL LLM CALL
# ============================================================

def call_real_llm(prompt: str) -> str:
    """
    Optional MOCK_LLM=0 extension.

    Uses Groq's OpenAI-compatible endpoint.
    This branch is never used in the required default
    MOCK_LLM=1 grading mode.
    """

    if not GROQ_API_KEY:
        raise RuntimeError(
            "MOCK_LLM=0 was requested, but GROQ_API_KEY "
            "is not configured."
        )

    payload = {
        "model": GROQ_MODEL,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return only valid JSON matching the "
                    "requested schema."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }

    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    return data["choices"][0]["message"]["content"]


# ============================================================
# JSON VALIDATION HELPER
# ============================================================

def validate_response_json(raw_text: str) -> AskResponse:
    """
    Validate LLM JSON output using Pydantic.
    Supports both Pydantic v2 and v1-style validation.
    """

    cleaned = raw_text.strip()

    # Remove optional markdown code fences.
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON: {exc}"
        ) from exc

    try:
        return AskResponse.model_validate(payload)
    except AttributeError:
        # Pydantic v1 fallback.
        return AskResponse.parse_obj(payload)


# ============================================================
# REAL LLM JSON WITH RETRIES
# ============================================================

def generate_real_llm_response(
    prompt: str,
) -> AskResponse:
    """
    Optional real-LLM path.

    Retry up to two additional times if validation fails.
    """

    corrective_messages = [
        prompt,
        (
            prompt
            + "\n\nYour previous response was invalid. "
            "Return ONLY valid JSON with exactly these fields: "
            "answer, sources, confidence."
        ),
        (
            prompt
            + "\n\nCorrection required: output ONLY a valid "
            "JSON object. No markdown, no explanation, and no "
            "additional fields."
        ),
    ]

    last_error = None

    for attempt, current_prompt in enumerate(
        corrective_messages,
        start=1
    ):
        try:
            raw = call_real_llm(current_prompt)
            return validate_response_json(raw)

        except Exception as exc:
            last_error = exc

            if attempt == len(corrective_messages):
                break

    return AskResponse(
        answer=(
            "Unable to generate a valid structured response "
            "from the optional real-LLM path."
        ),
        sources=[],
        confidence=0.0,
    )


# ============================================================
# NODE 1 - CLASSIFY INTENT
# ============================================================

def classify_intent(
    state: SupportState,
) -> SupportState:
    """
    Classify the query as:

    policy_question
    or
    general_question

    Required mock mode uses the exact keyword heuristic
    specified in the capstone.
    """

    query = state["query"].strip()

    if mock_mode_enabled():
        lowered = query.lower()

        is_policy_question = any(
            keyword in lowered
            for keyword in POLICY_KEYWORDS
        )

        intent = (
            "policy_question"
            if is_policy_question
            else "general_question"
        )

        return {
            **state,
            "intent": intent,
        }

    # Optional real-LLM classification.
    prompt = f"""
Classify the following customer query as exactly one of:
policy_question
general_question

Return JSON:
{{"intent": "policy_question"}}

Query:
{query}
"""

    try:
        raw = call_real_llm(prompt)

        parsed = json.loads(
            raw.strip()
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        intent = parsed.get("intent")

        if intent not in {
            "policy_question",
            "general_question",
        }:
            raise ValueError(
                "Invalid intent returned by real LLM."
            )

    except Exception:
        # Safe fallback.
        lowered = query.lower()

        intent = (
            "policy_question"
            if any(
                keyword in lowered
                for keyword in POLICY_KEYWORDS
            )
            else "general_question"
        )

    return {
        **state,
        "intent": intent,
    }


# ============================================================
# NODE 2 - RETRIEVE AND ANSWER
# ============================================================

def retrieve_and_answer(
    state: SupportState,
) -> SupportState:
    """
    Retrieve top-3 policy chunks using cosine similarity.

    Retrieval itself always runs locally and does not depend
    on MOCK_LLM.

    Only final answer generation branches on MOCK_LLM.
    """

    query = state["query"]

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    retrieved_chunks: list[dict[str, Any]] = []

    result_ids = results.get("ids", [[]])[0]
    result_documents = results.get(
        "documents",
        [[]],
    )[0]
    result_metadatas = results.get(
        "metadatas",
        [[]],
    )[0]
    result_distances = results.get(
        "distances",
        [[]],
    )[0]

    for index in range(len(result_ids)):
        doc_id = result_ids[index]

        metadata = (
            result_metadatas[index]
            if index < len(result_metadatas)
            else {}
        )

        document_text = (
            result_documents[index]
            if index < len(result_documents)
            else ""
        )

        distance = (
            result_distances[index]
            if index < len(result_distances)
            else None
        )

        retrieved_chunks.append(
            {
                "id": doc_id,
                "source": metadata.get(
                    "source",
                    doc_id,
                ),
                "document": document_text,
                "distance": distance,
            }
        )

    if not retrieved_chunks:
        return {
            **state,
            "retrieved_chunks": [],
            "answer": (
                "No relevant policy document was found."
            ),
            "sources": [],
            "confidence": 0.0,
        }

    top_chunk = retrieved_chunks[0]

    source_ids = [
        chunk["id"]
        for chunk in retrieved_chunks
    ]

    if mock_mode_enabled():
        snippet = (
            top_chunk["document"]
            .replace("\n", " ")
            .strip()[:200]
        )

        answer = (
            "Based on the retrieved context: "
            f"{snippet}"
        )

        return {
            **state,
            "retrieved_chunks": retrieved_chunks,
            "answer": answer,
            "sources": source_ids,
            "confidence": 1.0,
        }

    # Optional real LLM generation.
    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"[{chunk['id']}]\n"
            f"{chunk['document']}"
        )

    context = "\n\n".join(context_parts)

    prompt = STRUCTURED_PROMPT_TEMPLATE.format(
        query=query,
        context=context,
    )

    validated = generate_real_llm_response(prompt)

    # Ensure retrieved sources are represented.
    if not validated.sources:
        validated = AskResponse(
            answer=validated.answer,
            sources=source_ids,
            confidence=validated.confidence,
        )

    return {
        **state,
        "retrieved_chunks": retrieved_chunks,
        "answer": validated.answer,
        "sources": validated.sources,
        "confidence": validated.confidence,
    }


# ============================================================
# NODE 3 - DIRECT ANSWER
# ============================================================

def direct_answer(
    state: SupportState,
) -> SupportState:
    """
    General questions do not use retrieval in mock mode.
    """

    query = state["query"]

    if mock_mode_enabled():
        return {
            **state,
            "retrieved_chunks": [],
            "answer": (
                "I can only answer questions about "
                "Zepto policies right now."
            ),
            "sources": [],
            "confidence": 1.0,
        }

    # Optional real LLM direct-answer path.
    prompt = f"""
ROLE:
You are a Zepto support assistant.

TASK:
Answer the following question directly.

FORMAT:
Return valid JSON:
{{
  "answer": "string",
  "sources": [],
  "confidence": 0.0
}}

NEGATIVE CONSTRAINT:
Do not invent Zepto policy details.

FEW-SHOT EXAMPLE:
Question: "What is your favorite movie?"
Answer:
{{
  "answer": "I can help with Zepto support questions.",
  "sources": [],
  "confidence": 1.0
}}

Question:
{query}
"""

    validated = generate_real_llm_response(prompt)

    return {
        **state,
        "retrieved_chunks": [],
        "answer": validated.answer,
        "sources": [],
        "confidence": validated.confidence,
    }


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def route_after_classification(
    state: SupportState,
) -> str:
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph_builder = StateGraph(SupportState)

graph_builder.add_node(
    "classify_intent",
    classify_intent,
)

graph_builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer,
)

graph_builder.add_node(
    "direct_answer",
    direct_answer,
)

graph_builder.add_edge(
    START,
    "classify_intent",
)

graph_builder.add_conditional_edges(
    "classify_intent",
    route_after_classification,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer",
    },
)

graph_builder.add_edge(
    "retrieve_and_answer",
    END,
)

graph_builder.add_edge(
    "direct_answer",
    END,
)

graph = graph_builder.compile()


# ============================================================
# GRAPH EXECUTION HELPER
# ============================================================

def run_query(query: str) -> AskResponse:
    initial_state: SupportState = {
        "query": query,
        "intent": "",
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "confidence": 0.0,
    }

    final_state = graph.invoke(
        initial_state
    )

    return AskResponse(
        answer=final_state.get(
            "answer",
            "",
        ),
        sources=final_state.get(
            "sources",
            [],
        ),
        confidence=final_state.get(
            "confidence",
            0.0,
        ),
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Zepto Support Assistant",
    description=(
        "Offline RAG support assistant using "
        "LangGraph, ChromaDB and MiniLM embeddings."
    ),
    version="1.0.0",
)


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Zepto Support Assistant",
        "status": "running",
        "mock_llm": str(
            mock_mode_enabled()
        ),
        "collection": COLLECTION_NAME,
    }


# ============================================================
# ASK ENDPOINT
# ============================================================

@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask(request: AskRequest) -> AskResponse:

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    try:
        return run_query(query)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# LOCAL TESTS
# ============================================================

def run_local_examples() -> None:
    print("\n" + "=" * 60)
    print("LOCAL MOCK-MODE EXAMPLES")
    print("=" * 60)

    examples = [
        (
            "POLICY QUESTION",
            "What is the delivery charge for an order below INR 149?",
        ),
        (
            "GENERAL QUESTION",
            "What is your favorite movie?",
        ),
    ]

    for label, query in examples:
        print("\n" + "-" * 60)
        print(label)
        print("-" * 60)

        print(f"Query: {query}")

        result = run_query(query)

        print(
            json.dumps(
                result.model_dump(),
                indent=2,
            )
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ZEPT0 SUPPORT ASSISTANT")
    print("=" * 60)

    print(
        f"MOCK_LLM enabled: "
        f"{mock_mode_enabled()}"
    )

    run_local_examples()

    print("\nStarting FastAPI on port 7860...")

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )