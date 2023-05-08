from gensim.models import Word2Vec

# 1. 模型的训练
# 方法一：直接传入数组文件
sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
model = Word2Vec(sentences, size=100, min_count=1)

# 方法二：文件名的数据加载
# from gensim.models.word2vec import LineSentence
# # max_sentence_length : 单词袋的最大长度,如果超过：取max_sentence_length的作为一个数组，剩下的作为一个数组（如果超过，继续分）；
# # limit：语料库中取limit条句子来训练,None表示训练所有
# sentences = LineSentence('dataset/test.txt', max_sentence_length=5, limit=3)
# print("显示单词袋的最大长度为5，读取语料库中的5条前5条句子：")
# for sentence in sentences: # limit=3，产生了4条，因为语句超过max_sentence_length而发生了截取
#     print(sentence)
# # 训练
# sentences = LineSentence('dataset/test.txt', max_sentence_length=100, limit=200) # 传入的文本格式：一行一个句子。单词必须已经过预处理并用空格分隔。
# model = Word2Vec(sentences, size=100, min_count=1)

# 2. 模型的使用
# 计算两个词向量的相似度
sim1 = model.wv.similarity(u'cat', u'dog')
sim2 = model.wv.similarity(u'cat', u'meow')
print(sim1,sim2)
# 与某个词（cat）最相近的3个字的词
print(model.wv.similar_by_word('cat', topn=3))
# 找出不同类的词
sim3 = model.wv.doesnt_match(u'cat dog say woof'.split())
print(u'cat dog say woof 中不同类的名词', sim3)
# 某个词的词向量
wordvec = model.wv.word_vec('cat')
print(wordvec[:10])
# 求 wmd 距离
sentence1 = 'new jersey sometimes quiet autumn snowy april'
sentence2 = 'lime least liked fruit banana least liked'
print(model.wv.wmdistance(sentence1.split(" "), sentence2.split(" ")))
