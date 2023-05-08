from talon.signature.bruteforce import extract_signature
from langdetect import detect
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


email = "Hi Jane," \
        "" \
        "" \
        "Thank you for keeping me updated on this issue. I'm happy to hear that the issue got resolved after all and you can now use the app in its full functionality again. Also many thanks for your suggestions. We hope to improve this feature in the future. In case you experience any further problems with the app, please don't hesitate to contact me again." \
        "" \
        "" \
        "Best regards, " \
        "" \
        "John Doe " \
        "Customer Support " \
        "" \
        "1600 Amphitheatre Parkway " \
        "Mountain View, CA " \
        "United States"

cleaned_email, _ = extract_signature(email)
print(cleaned_email)

lang = detect(cleaned_email) # lang = 'en' for an English email

sentences = sent_tokenize(email, language = lang)
print('sentences:', sentences)
word_embeddings={}
f = open('./glove.68.100d.txt', encoding='utf-8')
for line in f:
        values=line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
f.close()

sentence_vectors = []
for i in sentences:
        if len(i) !=0:
            v = sum([word_embeddings.get(w, np.zeros((100,)))  for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100, ))
sentence_vectors.append(v)



n_clusters = np.ceil(len(sentence_vectors)**0.5)
kmeans = KMeans(n_clusters=n_clusters)
kmeans = kmeans.fit(sentence_vectors)


avg = []
for j in range(n_clusters):
    idx = np.where(kmeans.labels_ == j)[0]
    avg.append(np.mean(idx))
closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_)
ordering = sorted(range(n_clusters), key=lambda k: avg[k])
summary = ' '.join([email[closest[idx]] for idx in ordering])
