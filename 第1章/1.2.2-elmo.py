# 准备elmo模型向量
import tensorflow_hub as hub

# 加载模型
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
# 输入的数据集
texts = ["the cat is on the mat", "dogs are in the fog"]
embeddings = elmo(
texts,
signature="default",
as_dict=True)["default"]

# elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
#
# # 另一种方式输入数据
# tokens_input = [["the", "cat", "is", "on", "the", "mat"],
# ["dogs", "are", "in", "the", "fog", ""]]
# # 长度，表示tokens_input第一行6一个有效，第二行5个有效
# tokens_length = [6, 5]
# # 生成elmo embedding
# embeddings = elmo(
# inputs={
# "tokens": tokens_input,
# "sequence_len": tokens_length
# },
# signature="tokens",
# as_dict=True)["default"]

from tensorflow.python.keras import backend as K

sess = K.get_session()
array = sess.run(embeddings)
