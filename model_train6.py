import fasttext
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

model = fasttext.train_supervised(
    input="C:/Users/andreea/Desktop/VERIDION2/data/test_data.txt",
    lr=0.1,  
    epoch=200,
    wordNgrams=3,
    bucket=200000,
    dim=50,
    loss="ova"
)

df_taxonomy = pd.read_csv("C:/Users/andreea/Desktop/VERIDION2/data/taxonomy.csv")
df_test = pd.read_csv("C:/Users/andreea/Desktop/VERIDION2/data/test_data.csv")

# Function to find word vector
def get_word_vectors(text):
    words = text.split()
    return [model.get_word_vector(word) for word in words]


# Compute vectors for each row
df_test["word_vectors"] = df_test["final_text"].apply(get_word_vectors)
df_taxonomy["word_vectors"] = df_taxonomy["label_norm"].apply(get_word_vectors)

# Function to obtein top 5 most comun words
def get_top_5_vectors(text):
    words = text.split()
    most_common_words = [word for word, _ in Counter(words).most_common(10)]
    word_vectors = [get_word_vectors(word)[0] for word in most_common_words]
    return word_vectors

df_test["comun_vectors"] = df_test["final_text"].apply(get_top_5_vectors)

# Function to caompair vectors ant obtain the similarity score
def compare_vectors(row_vectors, taxonomy_vectors):
    similarities = []
    
    for tax_vectors in taxonomy_vectors:
        similarity_matrix = cosine_similarity(np.array(row_vectors), np.array(tax_vectors))
        binary_comparisons = (similarity_matrix > 0.8).astype(int)  # Prag de similaritate
        similarities.append(binary_comparisons.sum())  # Suma scorurilor

    return similarities


df_test["compararei"] = df_test["comun_vectors"].apply(lambda vecs: compare_vectors(vecs, df_taxonomy["word_vectors"]))

df_test["insurance_label"] = df_test["compararei"].apply(lambda scores: df_taxonomy.loc[np.argmax(scores), "label"])

df_results = df_test[["description", "business_tags", "sector", "category", "niche", "insurance_label"]]

df_results.to_csv("C:/Users/andreea/Desktop/VERIDION2/output8_results.csv", index=False, encoding="utf-8")

print("Procesare completa!")

