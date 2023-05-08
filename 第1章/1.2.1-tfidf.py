# word level tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
document = ["i come from china.",
            "hello world world world word."]
tfidf_model = TfidfVectorizer().fit(document)
sparse_result = tfidf_model.transform(document)
print("========稀疏矩阵表示法========")
print(sparse_result)
print("========稠密矩阵表示法========")
print(sparse_result.todense())
print("========词汇编号========")
print(tfidf_model.vocabulary_)
