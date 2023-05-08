# 使用 sklearn 进行one-hot编码
import numpy as np
from sklearn.preprocessing import OneHotEncoder

ls = ['i', 'like', 'dog'] # 对 ls 中每一个单词进行编码

# 为每个单词进行编号
enNo = {}
deNo = {}
for ind, l in enumerate(ls):
    enNo[l] = ind+1
    deNo[ind+1] = l
print("======== 单词编号 ========")
print(enNo)

No = list(enNo.values())
No = np.array(No).reshape(len(No),-1) # 必须是列向量
enc = OneHotEncoder() # onehot 编码器
enc.fit(No)
target = enc.transform(No).toarray()
print("======== 编码结果 ========")
for i, enc in enumerate(target):
    print("{0}\t-->\t{1}".format(deNo[i+1],enc))
