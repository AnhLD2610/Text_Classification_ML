import os
import numpy as np
import py_vncorenlp 
import re
from sklearn.model_selection import train_test_split

rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='D:/ML_project/vncorenlp')
class PreProcessingData:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def get_subfolders(folder_path):
        subfolders = []
        for root, dirs, files in os.walk(folder_path):
            for dir in dirs:
                subfolder_path = os.path.join(root, dir)
                subfolders.append(subfolder_path)
        return subfolders
    
    def get_data(subfolders_data):
        X = []
        y = []
        my_dictionary = {}
        stopwords_path = "D:/ML_project/vietnamese-stopwords-dash.txt"
        with open(stopwords_path, "r", encoding = 'utf-8') as stopwords_file:
            stopwords = stopwords_file.read().splitlines()
        for subfolder in subfolders_data:
            for root, dirs, files in os.walk(subfolder):
                for file in files:
                    with open(subfolder + '\\' + file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        text = ' '.join(lines)
                        # dua het cac ki tu ve lowercase
                        text = text.lower()
                        # xoa cac ki tu dac biet
                        #text = re.sub(r'[^a-zA-Z]', ' ', text)

                        text = re.sub(r'[^\w\s]|[\d]', ' ', text)
                        # tach tu
                        text = rdrsegmenter.word_segment(text)
                        #print(text)
                        text = ' '.join(text)
                        text = text.split()
                        # remove stopwords
                        text1 = []
                        for word in text:
                            if word in stopwords:
                                continue
                            else:
                                text1.append(word)
                        text = ' '.join(text1)
                        
                        X.append(text)
                        y.append(os.path.basename(subfolder))
                        
        
        for i in range(len(X)):
            temp = X[i].split()
            for word in temp:
                if word in my_dictionary:
                    my_dictionary[word] += 1
                else:
                    my_dictionary[word] = 1
        
        for word in list(my_dictionary.keys()):
            if my_dictionary[word] <=20:
               del my_dictionary[word]

        for i in range(len(X)):
            text1 = X[i].split()
            text2 = []
            for word in text1:
                if  word in my_dictionary.keys():
                    text2.append(word)
                else:
                    continue
            X[i] = ' '.join(text2) 
            
        # print(len(my_dictionary))
                        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

        return X_train,X_test,y_train,y_test
    
    def get_data_w2v(subfolders_data):
        X = []
        y = []
        my_dictionary = {}
        stopwords_path = "D:/ML_project/vietnamese-stopwords-dash.txt"
        with open(stopwords_path, "r", encoding = 'utf-8') as stopwords_file:
            stopwords = stopwords_file.read().splitlines()
        for subfolder in subfolders_data:
            for root, dirs, files in os.walk(subfolder):
                for file in files:
                    with open(subfolder + '\\' + file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        text = ' '.join(lines)
                        # dua het cac ki tu ve lowercase
                        text = text.lower()
                        # xoa cac ki tu dac biet
                        #text = re.sub(r'[^a-zA-Z]', ' ', text)

                        text = re.sub(r'[^.\w\s]|[\d]', ' ', text)
                        # tach tu
                        text = rdrsegmenter.word_segment(text)
                        #print(text)
                        text = ' '.join(text)
                        text = text.split()
                        # remove stopwords
                        text1 = []
                        for word in text:
                            if word in stopwords:
                                continue
                            else:
                                text1.append(word)
                        text = ' '.join(text1)
                        
                        X.append(text)
                        y.append(os.path.basename(subfolder))
                        
        # cac cau dang bi dinh kieu 'het cau.'
        # h lsao de loai bo stop word ma van giu lai dc dau cham
        
        for i in range(len(X)):
            text = X[i].split('.')
            for j in range (len(text)):
                # text[j] la cac sentences
                text[j] = text[j].strip()
            X[i] = ' . '.join(text)
                
        for i in range(len(X)):
            temp = X[i].split()
            for word in temp:
                if word in my_dictionary:
                    my_dictionary[word] += 1
                else:
                    my_dictionary[word] = 1
        
        for word in list(my_dictionary.keys()):
            if my_dictionary[word] <=20:
               del my_dictionary[word]

        for i in range(len(X)):
            text1 = X[i].split()
            text2 = []
            for word in text1:
                if  word in my_dictionary.keys():
                    text2.append(word)
                else:
                    continue
            X[i] = ' '.join(text2)
                        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

        return X_train,X_test,y_train,y_test

# luu ca dau '.' khi goi thi xoa dau '.' voi tfidf, doc2vec
# 1 la tfidf, 2 la word2vec tfidf , ko dung doc2vec