from sklearn.feature_extraction.text import CountVectorizer

text = ['i like dog and i like cat', 'i love coffee', 'i hate milk']
ngram_vectorizer = CountVectorizer(ngram_range=(2, 2), analyzer="word") # ngram_range=(a, b) 表示ngram的取值为 [a, b] 之间
# 直接将未经处理的语料库作为输入，内部会自动进行分词，并去停用词
ngram_vectorizer.fit(text)
print("======= 分词后的值 =======")
print(ngram_vectorizer.get_feature_names())
print("======= 词汇表 =======")
print(ngram_vectorizer.vocabulary_)

print("======= 词袋 =======")
result = ngram_vectorizer.transform(text)
print(result.toarray())


chars = ["I am a good man !"]

ngram_vectorizer = CountVectorizer(ngram_range=(2, 2),analyzer = "char")
# 直接将未经处理的语料库作为输入，内部会自动进行分词，并去停用词
ngram_vectorizer.fit(chars)

print("======= 分词后的值 =======")
print(ngram_vectorizer.get_feature_names())

print("======= 词汇表 =======")
print(ngram_vectorizer.vocabulary_)

print("======= 词袋 =======")
result =  ngram_vectorizer.transform(chars)
print(result.toarray())
