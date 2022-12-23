import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
import time


data = pd.read_csv('cleaned_imputed.csv')

x_regressor = data[['HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']]
y_regressor = data['FTHG']

scaler = MinMaxScaler()
x_regressor = scaler.fit_transform(x_regressor)
x_train_regressor, y_train_regressor, x_test_regressor, y_test_regressor = train_test_split(x_regressor, y_regressor, train_size=0.2)

x_classifier = data[['HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']]
y_classifier = data['Result']

scaler = MinMaxScaler()
x_classifier = scaler.fit_transform(x_classifier)
x_train_classifier, y_train_classifier, x_test_classifier, y_test_classifier = train_test_split(x_classifier, y_classifier, train_size=0.2)


# params = {'hidden_layer_sizes':[(10, 22, 1)],'early_stopping':[True],'validation_fraction':[0.2], 'activation':[ 'tanh'], 'solver':['adam'],'learning_rate':['invscaling'], 'max_iter':[10000], 'verbose':[True]}
# # params = {'hidden_layer_sizes':[(10, 100, 100, 1), (10, 150, 150, 1), (10, 100, 100, 100, 1), (10, 150, 150, 150, 1)], 'activation':['logistic','relu', 'identity', 'tanh'], 'solver':['sgd', 'lbfgs', 'adam'], 'alpha':[0.0001, 0.001, 0.00001], 'learning_rate':['constant', 'invscaling', 'adaptive'], 'learning_rate_init':[0.001, 0.0001, 0.01], 'power_t':[0.1, 0.5, 0.9], 'max_iter':[200], 'shuffle':[True, False], 'warm_start':[True, False], 'momentum':[0.1, 0.5, 0.9],  'beta_1':[0.1, 0.5, 0.9], 'beta_2':[0.1, 0.5, 0.9], 'verbose': [True]}

# nn = MLPRegressor()
# grid = GridSearchCV(estimator=nn, param_grid=params, n_jobs=-1)
# start = time.time()
# grid.fit(x, y)
# end = time.time()
# print(grid.best_estimator_, grid.best_index_, grid.best_score_)
# print(end - start / 60)

# with open('estimator.txt', 'w') as f:
#     f.write(f'Best estimator {grid.best_estimator_} \n, Best index {grid.best_index_} \n, Best parameters {grid.best_params_} \n, Best Score {grid.best_score_} \n' , )

params = {'hidden_layer_sizes':[(10, 22, 1),(10, 33, 1),(10, 44, 1),(10, 55, 1), (10, 66, 1),(10, 77, 1),(10, 88, 1), (10, 99, 1),],'early_stopping':[True],'validation_fraction':[0.2], 'activation':[ 'tanh'], 'solver':['adam'],'learning_rate':['invscaling'], 'max_iter':[10000], 'verbose':[True]}


nn = MLPClassifier()
grid = GridSearchCV(estimator=nn, param_grid=params, n_jobs=-1)
start = time.time()
grid.fit(x_classifier, y_classifier)
end = time.time()
print(grid.best_estimator_, grid.best_index_, grid.best_score_)
print(end - start / 60)

with open('all_matches_classifier_estimator.txt', 'w') as f:
    f.write(f'Best estimator {grid.best_estimator_} \n, Best index {grid.best_index_} \n, Best parameters {grid.best_params_} \n, Best Score {grid.best_score_} \n' , )
