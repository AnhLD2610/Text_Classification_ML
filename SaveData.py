
import os
import numpy as np
import pickle
from PreProcessingData import PreProcessingData

folder_path = 'D:\ML_project\FinalData'

subfolders_data = PreProcessingData.get_subfolders(folder_path)
X_train, X_test, y_train, y_test = PreProcessingData.get_data(subfolders_data)
X_train_w2v, X_test_w2v, y_train_w2v, y_test_w2v = PreProcessingData.get_data_w2v(subfolders_data)

# print(len(X_train))
# print(len(X_test))
# print(len(y_train))
# print(len(y_test))

pickle.dump(X_train, open('D:\ML_project\FinalData\X_train.pkl', 'wb'))
pickle.dump(y_train, open('D:\ML_project\FinalData\y_train.pkl', 'wb'))
pickle.dump(X_test, open('D:\ML_project\FinalData\X_test.pkl', 'wb'))
pickle.dump(y_test, open('D:\ML_project\FinalData\y_test.pkl', 'wb'))

pickle.dump(X_train_w2v, open('D:\ML_project\FinalData\X_train_w2v.pkl', 'wb'))
pickle.dump(X_test_w2v, open('D:\ML_project\FinalData\X_test_w2v.pkl', 'wb'))
pickle.dump(y_train_w2v, open('D:\ML_project\FinalData\y_train_w2v.pkl', 'wb'))
pickle.dump(y_test_w2v, open('D:\ML_project\FinalData\y_test_w2v.pkl', 'wb'))



