!pip install gensim nltk matplotlib seaborn
import nltk
import gensim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
nltk.download("punkt")
nltk.download("reuters")
nltk.download("punkt_tab")
corpus = reuters.sents()[:5000]
medical_sentences = [[word.lower() for word in sent if word.isalpha()] for sent in corpus]
model = Word2Vec(sentences=medical_sentences, vector_size=100, window=5, min_count=2,
workers=4)
model.save("medical_word2vec.model")
model = Word2Vec.load("medical_word2vec.model")
medical_term = "disease"
if medical_term in model.wv:
    print(f"Top 5 words similar to '{medical_term}':")
    print(model.wv.most_similar(medical_term, topn=5))
else:
    print(f"'{medical_term}' not found in vocabulary.")
initial_selected_words = ["health", "disease", "doctor", "patient", "treatment", "medicine", "virus",
"surgery", "nurse", "hospital"]
selected_words = [word for word in initial_selected_words if word in model.wv]
word_vectors = np.array([model.wv[word] for word in selected_words])
tsne = TSNE(n_components=2, random_state=42, perplexity=3)
word_vectors_2d = tsne.fit_transform(word_vectors)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=word_vectors_2d[:, 0], y=word_vectors_2d[:, 1], s=100, color="blue")
for i, word in enumerate(selected_words):
    plt.annotate(word, (word_vectors_2d[i, 0], word_vectors_2d[i, 1]), fontsize=12,
bbox=dict(facecolor='white', alpha=0.7))
plt.title("t-SNE Visualization of Medical Word Embeddings", fontsize=14)
plt.xlabel("t-SNE Dimension 1", fontsize=12)
plt.ylabel("t-SNE Dimension 2", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
