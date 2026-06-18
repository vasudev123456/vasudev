!pip -q install gensim transformers torch
import gensim.downloader as api
from transformers import pipeline
wv = api.load("glove-wiki-gigaword-100")
gen = pipeline("text-generation", model="gpt2")
def enrich(prompt):
    return " ".join(
        [w for word in prompt.split()
         for w in ([word] + [x[0] for x in wv.most_similar(word, topn=2)]
                   if word in wv else [word])]
    )
generate = lambda p: gen(p, max_length=100)[0]["generated_text"]
prompt = "Explain machine learning in healthcare"
enriched = enrich(prompt)
out1 = generate(prompt)
out2 = generate(enriched)
print(" Original Prompt:\n", prompt)
print("\n Enriched Prompt:\n", enriched)
print("\n Output (Original):\n", out1)
print("\n Output (Enriched):\n", out2)
print("\nComparison:")
print("Original length:", len(out1.split()))
print("Enriched length:", len(out2.split()))
print("Insight: Enriched prompt → more detailed but may add noise.")
