from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
bow = CountVectorizer(min_df=5, max_df=.99, ngram_range=(1, 2))
# 删除带有df参数的稀有词和常用词
# 包括单个和2个单词对
train_text = ['John likes to watch movies. Mary likes too.']
test_text = ['John also likes to watch football games.']

X_train_vec = bow.fit_transform(train_text)
X_test_vec = bow.transform(test_text)
print('X_train_vec:', X_train_vec)
#
# cols = bow.get_feature_names() #if you need feature names
# #
# model = RandomForestClassifier(n_estimators=500, n_jobs=8)
# model.fit(X_train_vec, y_train)
# model.score(X_test_vec, y_test)
