import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Local model (small & fast)
MODEL_NAME = "google/flan-t5-base"

# Load once
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

qa_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

def generate_answer(question: str, retrieved_chunks: list[str]) -> str:
    """
    Generate an answer using the retrieved context.
    Works entirely offline using FLAN-T5.
    """
    if not retrieved_chunks:
        return "No relevant context found."

    # Combine top chunks
    context = "\n".join(retrieved_chunks[:3])
    
    prompt = (
        f"Answer the question based on the given movie transcript.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer clearly and concisely:"
    )

    print("\n🤖 Generating Answer (using local model)...\n")
    output = qa_pipeline(prompt, max_new_tokens=128, do_sample=False)
    answer = output[0]["generated_text"].strip()

    print(f"💡 Final Answer:\n{answer}")
    return answer
