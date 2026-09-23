from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING EMBEDDING MODEL")
print("=" * 60)

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully.")


# ============================================================
# LOAD ALL 8 POLICY DOCUMENTS
# ============================================================

print("\n" + "=" * 60)
print("LOADING POLICY DOCUMENTS")
print("=" * 60)

doc_files = sorted(DOCS_DIR.glob("doc_*.txt"))

if len(doc_files) != 8:
    raise RuntimeError(
        f"Expected exactly 8 policy documents, found {len(doc_files)}."
    )

documents = []
ids = []
metadatas = []

for file_path in doc_files:
    text = file_path.read_text(encoding="utf-8").strip()

    if not text:
        raise RuntimeError(f"{file_path.name} is empty.")

    documents.append(text)
    ids.append(file_path.stem)

    metadatas.append(
        {
            "source": file_path.name,
            "document_id": file_path.stem,
        }
    )

    print(f"Loaded: {file_path.name}")

print(f"\nTotal documents loaded: {len(documents)}")


# ============================================================
# CREATE LOCAL EMBEDDINGS
# ============================================================

print("\n" + "=" * 60)
print("CREATING EMBEDDINGS")
print("=" * 60)

embeddings = model.encode(
    documents,
    normalize_embeddings=True,
    show_progress_bar=True,
).tolist()

print(f"Created {len(embeddings)} embeddings.")


# ============================================================
# CREATE PERSISTENT CHROMADB CLIENT
# ============================================================

print("\n" + "=" * 60)
print("CREATING CHROMADB COLLECTION")
print("=" * 60)

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


# ============================================================
# RECREATE COLLECTION WITH COSINE DISTANCE
# ============================================================

try:
    client.delete_collection(
        name="zepto_policy_documents"
    )
    print("Existing ChromaDB collection removed.")
except Exception:
    print("No previous collection found. Creating a new collection.")

collection = client.create_collection(
    name="zepto_policy_documents",
    metadata={
        "description": "Zepto policy document embeddings",
        "hnsw:space": "cosine",
    },
)

print("Created collection: zepto_policy_documents")
print("Distance metric: cosine")


# ============================================================
# STORE EMBEDDINGS
# ============================================================

print("\n" + "=" * 60)
print("STORING EMBEDDINGS")
print("=" * 60)

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
)

print(f"Stored {len(ids)} documents in ChromaDB.")


# ============================================================
# VERIFY COLLECTION
# ============================================================

print("\n" + "=" * 60)
print("CHROMADB VERIFICATION")
print("=" * 60)

count = collection.count()

print("Collection name: zepto_policy_documents")
print(f"Stored record count: {count}")
print("Distance metric: cosine")

if count != 8:
    raise RuntimeError(
        f"Verification failed. Expected 8 records, found {count}."
    )


# ============================================================
# TEST RETRIEVAL
# ============================================================

print("\n" + "=" * 60)
print("TEST RETRIEVAL")
print("=" * 60)

test_query = (
    "What is the delivery charge for an order below INR 149?"
)

print(f"Test query: {test_query}")

query_embedding = model.encode(
    [test_query],
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


# ============================================================
# DISPLAY RETRIEVED RESULTS
# ============================================================

print("\nTop 3 retrieved documents:")

for index in range(len(results["ids"][0])):
    doc_id = results["ids"][0][index]
    distance = results["distances"][0][index]
    source = results["metadatas"][0][index]["source"]

    print(
        f"{index + 1}. "
        f"{doc_id} | "
        f"{source} | "
        f"cosine_distance={distance:.4f}"
    )


print("\nTop retrieved document:")
print(results["documents"][0][0][:500])


# ============================================================
# RETRIEVAL CORRECTNESS CHECK
# ============================================================

top_document = results["ids"][0][0]

if top_document != "doc_01":
    raise RuntimeError(
        f"Retrieval verification failed. "
        f"Expected doc_01 as the top result, got {top_document}."
    )

print("\nRetrieval verification: PASSED")
print("Correct source document retrieved: doc_01")


# ============================================================
# FINAL COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("MODULE 3 - CHROMADB INDEX COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"ChromaDB path: {CHROMA_DIR}")
print("All 8 policy documents are embedded and queryable.")
print("Cosine similarity is configured for retrieval.")