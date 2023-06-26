import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import py_vncorenlp 
from sklearn.feature_extraction.text import CountVectorizer
import pickle 
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='D:/ML_project/vncorenlp')

def BagOfWord(X_train, X_test):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X_train)
    # print("Vocabulary:")
    # print(len(vectorizer.vocabulary_))
    
    # print("Bag-of-Words Matrix:")
    # print(X.toarray())
    return X
        
def tfidf(X_train,X_test):
    for i in range(len(X_train)):
        X_train[i] = X_train[i].replace('.',' ')
        X_train[i] = ' '.join(X_train[i].split())
    tfidf = TfidfVectorizer()
    # text_embedding co dang shape(so bai bao,so luong tu dien)
    text_embedding_iftdf = tfidf.fit_transform(X_train)
    test_embeddding = tfidf.transform(X_test)
    # return text_embedding_iftdf.toarray(), test_embeddding.toarray()
    return text_embedding_iftdf, test_embeddding


def doc2vec(X_train, x_test):
    for i in range (len(X_train)):
        tagged_docs = [TaggedDocument(words=text, tags=[str(y_train[i])+str(j)]) for j, text in enumerate(X_train[i])]

        # Initialize and train the Doc2Vec model
    model = Doc2Vec(vector_size=300, min_count=1, epochs=20)
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)

    return model
    # Save the embedding vectors in an array
    # embedding_vectors = [model.infer_vector(text) for text in X_train[i]]
    # embedding_array = np.array(embedding_vectors)

def Word2Vec_lstm(X_train, X_test):
    sentences = []
    size_vector = 10
    for text in X_train:
        for sentence in text.split('.'):
            if len(sentence.split()) !=0:
                sentences.append(sentence.split())
    for text in X_test:
        for sentence in text.split('.'):
            if len(sentence.split()) !=0:
                sentences.append(sentence.split())
    # a = {'apple'}
    # # for i in range (len(X_train)):
    # #     for j in range(len(X_train[i])):
    # #         a.add(X_train[i][j])
    # for i in range(len(sentences)):
    #     for j in range(len(sentences[i])):
    #         a.add(sentences[i][j])
    # print(len(a))
    # print(sentences[0])
    # ['trung_quốc', 'phóng', 'thành_công', 'module', 'trạm', 'vũ_trụ']
    model = Word2Vec(sentences = sentences, vector_size = size_vector, window = 5, min_count = 0 ,workers = 4 )
    # model.train(sentences, total_examples = 1, epochs = 10)
    words = list(model.wv.index_to_key)
    
    max_length = 0
    for i in range(len(X_train)):
        max_length = max(len(X_train[i].split()),max_length)

    for i in range(len(X_test)):
        max_length = max(len(X_test[i].split()),max_length)
    
    print(max_length)
    # print(words)
    for i in range(len(X_train)):
        X_train[i] = X_train[i].split()
        for j in range(len(X_train[i])):
            if X_train[i][j] in words:
                X_train[i][j] = model.wv[X_train[i][j]]
            else:
                print('1')

    for i in range(len(X_test)):
        X_test[i] = X_test[i].split()
        for j in range(len(X_test[i])):
            X_test[i][j] = model.wv[X_test[i][j]]
            
    zeros_list = [0] * 300

    for i in range(len(X_train)):
        X_train[i] = X_train[i] + [[0]*size_vector]*(max_length-len(X_train[i]))
        # print(X_train[i])
    # print(X_train)
    for i in range(len(X_test)):
        X_test[i] = X_test[i] + [[0]*size_vector]*(max_length-len(X_test[i]))
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    print(X_train.shape)
    # model.save("word2vec.model")
    return X_train, X_test

X_train = pickle.load(open('D:\ML_project\FinalData\X_train_w2v.pkl', 'rb'))
X_test = pickle.load(open('D:\ML_project\FinalData\X_test_w2v.pkl', 'rb'))
y_train = pickle.load(open('D:\ML_project\FinalData\y_train_w2v.pkl', 'rb'))
y_test = pickle.load(open('D:\ML_project\FinalData\y_test_w2v.pkl', 'rb'))
Word2Vec_lstm(X_train, X_test)
# from gensim.test.utils import common_texts
# print(common_texts[1])
# ['survey', 'user', 'computer', 'system', 'response', 'time']