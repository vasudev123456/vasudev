from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
def summarize_text(text, max_length=130, min_length=50):
    inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=max_length,
        min_length=min_length,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
long_text = """
Artificial Intelligence (AI) is a rapidly advancing field that aims to create machines capable of human-like thinking.
AI is used in various industries, from healthcare to finance, improving efficiency and accuracy. Machine learning,
a subset of AI, enables computers to learn from data and make predictions without being explicitly programmed.
With deep learning, neural networks can process vast amounts of information and recognize patterns, leading
to advancements in self-driving cars, natural language processing, and medical diagnostics.
"""
summary = summarize_text(long_text)
print("Original Text:")
print(long_text)
print("\nSummarized Text:")
print(summary)
