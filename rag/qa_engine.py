import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Use a small, local model that supports text2text-generation
MODEL_NAME = "google/flan-t5-base"  # good accuracy + small footprint

# Load model & tokenizer once
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Create pipeline
qa_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)


def generate_answer(question: str, relevant_chunks: list[str], top_k: int = 3) -> str:
    """
    Generate answer using a local FLAN-T5 model.
    """
    # Combine top chunks
    context = "\n".join(relevant_chunks[:top_k])

    # Truncate context if too long for model (max 512 tokens)
    inputs = tokenizer(
        f"Answer the following question based on context:\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:",
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        num_beams=4
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()