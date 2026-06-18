import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
import spacy
!python -m spacy download en_core_web_md
nlp = spacy.load("en_core_web_md")
tech_words = ["computer", "internet", "software", "hardware", "network",
              "AI", "cloud", "cybersecurity", "database", "robotics"]
word_vectors = np.array([nlp(word).vector for word in tech_words])
pca_result = PCA(n_components=2).fit_transform(word_vectors)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], s=100, color="blue")
for i, word in enumerate(tech_words):
    plt.annotate(word, (pca_result[i, 0], pca_result[i, 1]),
                 fontsize=12, xytext=(5,5), textcoords="offset points",
                 bbox=dict(facecolor='white', alpha=0.7))
plt.title("PCA Visualization of Word Embeddings", fontsize=14)
plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
