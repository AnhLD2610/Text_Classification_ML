import pickle
import TextEmbedding 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.utils import to_categorical

# doi ten duoi file thanh ipynb la dc
X_train = pickle.load(open('D:\ML_project\FinalData\X_train.pkl', 'rb'))
X_test = pickle.load(open('D:\ML_project\FinalData\X_test.pkl', 'rb'))
y_train = pickle.load(open('D:\ML_project\FinalData\y_train.pkl', 'rb'))
y_test = pickle.load(open('D:\ML_project\FinalData\y_test.pkl', 'rb'))

#doc2vec_embedding = TextEmbedding.doc2vec(X_train=X_train,y_train=y_train)
doc2vec_embedding, test_embedding = TextEmbedding.tfidf(X_train,X_test)
print(doc2vec_embedding[0].shape)

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

def svm_model(X_train, y_train, X_test, y_test, kernel, C):
    C = 1.0  # SVM regularization parameter
    #svc = LinearSVC(C=C).fit(embedding_array, y_train)
    # rbf_svc = SVC(kernel='rbf', C=10).fit(doc2vec_embedding, y_train)
    rbf_svc = SVC(kernel='poly', degree=3, C=C).fit(doc2vec_embedding, y_train)
    test_predictions = rbf_svc.predict(test_embedding)
    cf = confusion_matrix(y_test, test_predictions)
    
    f1 = f1_score(y_test, test_predictions, average='micro')
    recall = recall_score(y_test, test_predictions, average='micro')
    accuracy = accuracy_score(y_test, test_predictions)
    return accuracy, recall, f1

def random_forest(X_train, y_train, X_test, y_test):
    rdf = RandomForestClassifier(n_estimators=1000, max_depth=30, random_state=0)
    rdf.fit(doc2vec_embedding, y_train)
    test_predictions = rdf.predict(test_embedding)
    f1 = f1_score(y_test, test_predictions, average='micro')
    recall = recall_score(y_test, test_predictions, average='micro')
    accuracy = accuracy_score(y_test, test_predictions)
    return accuracy, recall, f1

def KNN(X_train, y_train, X_test, y_test):
    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(doc2vec_embedding, y_train)
    test_predictions = neigh.predict(test_embedding)
    f1 = f1_score(y_test, test_predictions, average='micro')
    recall = recall_score(y_test, test_predictions, average='micro')
    accuracy = accuracy_score(y_test, test_predictions)
    return accuracy, recall, f1

def LSTM_model(X_train, y_train, X_test, y_test):
    
    model = Sequential()
    model.add(LSTM(10, activation = 'relu', input_shape = X_train[0]))
    model.add(Dense(10, activation = 'sigmoid'))

    # Compile the model
    # Cross-Entropy Loss (Log Loss)
    # Categorical Hinge Loss
    # metrics = accuracy 
    model.compile(optimizer='adam',learning_rate = 10, loss= "categorical_crossentropy", metrics="accuracy")
    model.summary()
    history = model.fit(X_train, y_train, batch_size = 32, epochs = 20, verbose = 1)

    # Test the model after training
    test_results = model.evaluate(test_embedding, y_test, verbose=1)
    f1 = f1_score(y_test, test_results, average='micro')
    recall = recall_score(y_test, test_results, average='micro')
    accuracy = accuracy_score(y_test, test_results)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {100*test_results[1]}%')
    
    return accuracy, recall, f1

# hyper parameter are kernel, C, 
# print choose of C for each kernel C= 0.0001,0.001, 0.01, 0.1,1,10,100,1000
def GridSearchCV_SVM():
    params = dict()
    param_grid = {
        # 'C': [0.1, 1, 10, 100],
        # # 'gamma': [0.1, 0.01, 0.001],
        # 'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
        'C': [1, 10],
        # # 'gamma': [0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    }
    f1_scorer = make_scorer(f1_score, average='micro')
    recall_scorer = make_scorer(recall_score, average='micro')
    svm_model = SVC()
    grid_search = GridSearchCV(svm_model, param_grid, cv=5)
    grid_search.fit(doc2vec_embedding, y_train)

    print(grid_search.cv_results_['mean_test_score'])
    print(grid_search.cv_results_['params'])
    # grid_search = GridSearchCV(svm_model, param_grid, scoring= f1_scorer, cv=5)
    # grid_search = GridSearchCV(svm_model, param_grid, scoring= recall_scorer, cv=5)

    
    # grid_search.fit(doc2vec_embedding, y_train)
    # print("Best Parameters: ", grid_search.best_params_)
    # print("Best Score: ", grid_search.best_score_)

    # accuracy = grid_search.score(X_test, y_test)
    # print("Test Accuracy: ", accuracy)
def GridSearch_KNN():
    grid_params = { 'n_neighbors' : [5,7,9,11,13,15],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}
    #Since we have provided the class validation score as 3( cv= 3),
    # Grid Search will evaluate the model 6 x 2 x 3 x 3 = 108 times with different hyperparameters.
    gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=3, n_jobs = -1)
    g_res = gs.fit(X_train, y_train)
    
def GridSearch_RandomForest():
    param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                            cv = 3, n_jobs = -1, verbose = 2)
    
# def GridSearch_LSTM(X_train, y_train, units = 50):
#     # number of epochs 
#     # number of neurons 
#     # number of batch_size 
#     model = Sequential()
#     model.add(LSTM(units, activation = 'relu', input_shape = X_train[0]))
#     model.add(Dense(10, activation = 'sigmoid'))
#     model.compile(optimizer='adam', loss= "categorical_crossentropy", metrics="accuracy")
    
#     lstm_model = KerasClassifier(build_fn=model, verbose=1)

#     parameters = {'units': [50, 100, 150], 'batch_size': [32, 64, 128], 'epochs': [10, 20]}
#     grid_search = GridSearchCV(estimator=lstm_model, param_grid=parameters, scoring=make_scorer(accuracy_score), cv=3)
#     grid_search.fit(X_train, y_train)

#     # Print the best parameters and score
#     print("Best parameters: ", grid_search.best_params_)
#     print("Best score: ", grid_search.best_score_)
#     return 1 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def create_lstm_model(units=50):
    model = Sequential()
    model.add(LSTM(units, activation='relu', input_shape=(X_train.shape[1],1)))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def GridSearch_LSTM(X_train, y_train, units=50):
    lstm_model = KerasClassifier(build_fn=create_lstm_model, units=units, verbose=1)

    parameters = {'units': [200, 100, 150], 'batch_size': [32, 64, 128], 'epochs': [10, 20]}
    grid_search = GridSearchCV(estimator=lstm_model, param_grid=parameters, scoring=make_scorer(accuracy_score), cv=3)
    grid_search.fit(X_train, y_train)

    # Print the best parameters and score
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    return 1
y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)
GridSearch_LSTM(X_train,y_train)
# print(y_train)
    
'''
def plot_result(parameters, mean_validation_score):
    param_values = parameters
    scores = mean_validation_score

    sorted_indices = np.argsort(scores)
    sorted_scores = np.array(scores)[sorted_indices]
    sorted_params = np.array(param_values)[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_scores, 'o-')
    plt.ylabel('Mean Validation Score')
    plt.title('Grid Search Results')

    x_labels = ['\n'.join([f'{param}: {value}' for param, value in param_set.items()]) for param_set in sorted_params]
    plt.xticks(range(len(sorted_params)), x_labels, rotation=0, ha='center')

    plt.tight_layout()
    plt.show()

result = [0.3,0.3,0.4,0.9, 0.8, 0.75, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
para = [{'C': 1, 'kernel': 'rbf'}, {'C': 1, 'kernel': 'linear'}, {'C': 10, 'kernel': 'rbf'}, {'C': 10, 'kernel': 'linear'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'},{'C': 1, 'kernel': 'rbf'}]
# plot_result(para, result)
result1 = []
para1 = []
for i in range (11):
    result1.append(result[i])
    para1.append(para[i])
plot_result(para1,result1)
result1 = []
para1 = []
for i in range (11,len(result)):
    result1.append(result[i])
    para1.append(para[i])
plot_result(para1,result1)
'''

# lstm thi output de dang onehot , con may cai khac de binh thuong duoc 



    

