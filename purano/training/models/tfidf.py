from sklearn.feature_extraction.text import TfidfVectorizer

from purano.util import tokenize_to_lemmas


def build_idf_vocabulary(texts, max_df=0.2, min_df=20):
    print("Building TfidfVectorizer...")
    vectorizer = TfidfVectorizer(tokenizer=tokenize_to_lemmas, max_df=max_df, min_df=min_df)
    vectorizer.fit(texts)
    idf_vector = vectorizer.idf_.tolist()

    print("{} words in vocabulary".format(len(idf_vector)))
    idfs = list()
    for word, idx in vectorizer.vocabulary_.items():
        idfs.append((word, idf_vector[idx]))

    idfs.sort(key=lambda x: (x[1], x[0]))
    return idfs
