from transformers import pipeline

from src.retriever import (
    retrieve_chunks,
    build_context
)


print("Loading language model...")

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1
)


PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust.

Use ONLY the complaint excerpts below.

Identify common themes, customer frustrations,
and recurring issues.

If the information is not available,
say that you do not have enough information.

Context:
{context}

Question:
{question}

Answer:
"""


def ask_rag(question, k=5):

    results = retrieve_chunks(question, k)

    context = build_context(results)

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    answer = generator(
        prompt,
        max_new_tokens=200,
        do_sample=False
    )[0]["generated_text"]

    return answer, results
