#coding=utf8
import gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
# 输入文件
glove_file = "./glove.6B/glove.6B.50d.txt"
# 输出文件
tmp_file = 'glove-output/Wikiglove_word2vec.txt' #get_tmpfile("./glove-output/Wikiglove_word2vec.txt")

# call glove2word2vec script
# default way (through CLI): python -m gensim.scripts.glove2word2vec --input <glove_file> --output <w2v_file>
# 开始转换
glove2word2vec(glove_file, tmp_file)
print('ok')
# 加载转化后的文件
#Linux下训练的词向量，在Windows下使用，不加encoding='utf-8', unicode_errors='ignore'会报错
model = KeyedVectors.load_word2vec_format(tmp_file, encoding='utf-8', unicode_errors='ignore')
model.save("Wikiglove_word2vec.model")
word1 = u'阿鲁举'
if word1 in model:
    print(u"'%s'的词向量为： " % word1)
    print(model[word1])
else:
    print(u'单词不在字典中！')
