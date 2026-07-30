import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading embeddings...")
embeddings_df = pd.read_parquet(
    "data/processed/complaint_embeddings.parquet"
)

print("Creating embedding matrix...")
embedding_matrix = np.vstack(
    embeddings_df["embedding"].values
)


def retrieve_chunks(question, k=5):

    query_embedding = model.encode(question)

    similarities = cosine_similarity(
        [query_embedding],
        embedding_matrix
    )[0]

    top_indices = np.argsort(similarities)[-k:][::-1]

    results = []

    for idx in top_indices:

        results.append({

            "text": embeddings_df.iloc[idx]["document"],

            "metadata": embeddings_df.iloc[idx]["metadata"],

            "score": similarities[idx]

        })

    return results


def build_context(results):

    context = ""

    for i, r in enumerate(results, start=1):

        context += f"""

Chunk {i}

{r['text']}

"""

    return context
