import numpy as np
import pickle
from gensim.models.word2vec import Word2Vec

X_train = pickle.load(open('D:\ML_project\FinalData\X_train_w2v.pkl', 'rb'))
X_test = pickle.load(open('D:\ML_project\FinalData\X_test_w2v.pkl', 'rb'))
y_train = pickle.load(open('FinalData\y_train_w2v.pkl', 'rb'))
y_test = pickle.load(open('FinalData\y_test_w2v.pkl', 'rb'))

list_topic ={
    "thoi-su": 0,
    "kinh-doanh": 1,
    "khoa-hoc": 2,
    "giai-tri": 3,
    "the-thao": 4,
    "phap-luat": 5, 
    "giao-duc": 6,
    "suc-khoe": 7,
    "doi-song": 8,
    "du-lich":  9
}
for i in range (len(y_train)):
    for x, y in list_topic.items():
        if y_train[i] == x:
            y_train[i] = y 
for i in range (len(y_test)):
    for x, y in list_topic.items():
        if y_test[i] == x:
            y_test[i] = y

sentences = []
size_vector = 10000
for text in X_train:
    for sentence in text.split('.'):
        if len(sentence.split()) !=0:
            sentences.append(sentence.split())
for text in X_test:
    for sentence in text.split('.'):
        if len(sentence.split()) !=0:
            sentences.append(sentence.split())
#print(sentences)
# print(sentences[0])
# ['trung_quốc', 'phóng', 'thành_công', 'module', 'trạm', 'vũ_trụ']
# model = Word2Vec(sentences = sentences, vector_size = size_vector, window = 5, min_count = 1, workers = 4 )

w2v_model = Word2Vec(vector_size=size_vector, window=5, min_count= 1, workers=4)
w2v_model.build_vocab(sentences)
w2v_model.train(sentences, total_examples=len(sentences), epochs=10)
w2v_model.wv.save_word2vec_format('w2v_model.bin', binary=True)
