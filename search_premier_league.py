import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time


data = pd.read_csv('cleaned_imputed_premier_league.csv')

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
x_classifier = x_classifier[:100]
y_classifier = y_classifier[:100]


# params = {'hidden_layer_sizes':[(10, 22, 1),(10, 33, 1),(10, 44, 1),(10, 55, 1), (10, 66, 1),(10, 77, 1),(10, 88, 1), (10, 99, 1),],'early_stopping':[True],'validation_fraction':[0.2], 'activation':[ 'tanh'], 'solver':['adam'],'learning_rate':['invscaling'], 'max_iter':[10000], 'verbose':[True]}

# nn = MLPRegressor()
# grid = GridSearchCV(estimator=nn, param_grid=params, n_jobs=-1)
# start = time.time()
# grid.fit(x_regressor, y_regressor)
# end = time.time()
# print(grid.best_estimator_, grid.best_index_, grid.best_score_)
# print(end - start / 60)

# with open('premier_league_regressor_estimator.txt', 'w') as f:
#     f.write(f'Best estimator {grid.best_estimator_} \n, Best index {grid.best_index_} \n, Best parameters {grid.best_params_} \n, Best Score {grid.best_score_} \n' , )

# Trying to estimate hidden layer size with the formula Nh = (a*(Ni + No)) where Ni=number of inputs, No=number of outputs, Ns=number of samples in training data set, a=arbiterary scalling factor 2-10
# params = {'hidden_layer_sizes':[(10, 26, 3),(10, 39, 3),(10, 52, 3),(10, 65, 3), (10, 78, 3),(10, 91, 3),(10, 104, 3), (10, 117, 3), (10, 130, 3),],'early_stopping':[True],'validation_fraction':[0.2], 'activation':['logistic','relu', 'identity', 'tanh'], 'solver':['sgd', 'lbfgs', 'adam'],'learning_rate':['invscaling'], 'max_iter':[10000], 'verbose':[True]}


# nn = MLPClassifier()
# grid = GridSearchCV(estimator=nn, param_grid=params, n_jobs=-1)
# start = time.time()
# grid.fit(x_classifier, y_classifier)
# end = time.time()
# print(grid.best_estimator_, grid.best_index_, grid.best_score_)
# print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')

# with open('premier_league_classifier_estimator.txt', 'w') as f:
#     f.write(f'Best estimator {grid.best_estimator_} \n, Best index {grid.best_index_} \n, Best parameters {grid.best_params_} \n, Best Score {grid.best_score_} \n' , )

# SHOULD TAKE ABOUT 10 MINUTES


params = {'n_neighbors':[2,3,4,5,6], 'weights':['distance', 'uniform'], 'leaf_size':[6, 7, 8, 9, 10,11,12,13,14 ], 'p':[1,2, 3, 4]}

knn = KNeighborsClassifier()
grid = GridSearchCV(estimator=knn, param_grid=params, n_jobs=-1)
start = time.time()
grid.fit(x_classifier, y_classifier)
end = time.time()
print(grid.best_estimator_, grid.best_index_, grid.best_score_)
print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')

with open('premier_league_knn.txt', 'w') as f:
    f.write(f'Best estimator {grid.best_estimator_} \n, Best index {grid.best_index_} \n, Best parameters {grid.best_params_} \n, Best Score {grid.best_score_} \n' , )



# SHOULD TAKE ABOUT A MINUTE

params = {'loss':['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insentive', 'squared_epsilon_insentive'], 'penalty':['l2', 'l1', 'elasticnet', 'none'], 'alpha':[0.001,0.002, 0.003, 0.004, 0.005], 'l1_ratio':[1.0, 1.5, 1.75, 2.0], 'fit_intercept':[True, False], 'max_iter':[10000], 'epsilon':[0.0, 0.1, 0.2, 0.3, 0.4],'shuffle':[True, False], 'random_state':[42], 'learning_rate':['optimal', 'constant', 'invscaling', 'adaptive'], 'eta0':[0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4], 'power_t':[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,], 'early_stopping':[True], 'warm_start':[True], 'average':[True, False], 'n_iter_no_change':[10]}

sgd = SGDClassifier()
grid = GridSearchCV(estimator=sgd, param_grid=params, n_jobs=-1)
start = time.time()
grid.fit(x_classifier, y_classifier)
end = time.time()
print(grid.best_estimator_, grid.best_index_, grid.best_score_)
print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')

with open('premier_league_sgd.txt', 'w') as f:
    f.write(f'Best estimator {grid.best_estimator_} \n, Best index {grid.best_index_} \n, Best parameters {grid.best_params_} \n, Best Score {grid.best_score_} \n' , )


params = {'n_estimators':[30, 40, 50, 60, 70], 'criterion':['gini', 'entropy', 'log_loss'], 'max_depth':[None], 'min_samples_split':[ 3, 4, 5, 6], 'min_samples_leaf':[4, 5, 7, 8, 9], 'min_weight_fraction_leaf':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 'max_features':['sqrt', 'log2'],  'min_impurity_decrease':[0.0, 0.1, 0.2, 0.3, 0.4], 'bootstrap':[True, False], 'oob_score':[False, True], 'random_state':[42], 'warm_start':[False, True], 'class_weight':['balanced', 'balanced_subsample'], 'ccp_alpha':[0.0, 0.1, 0.2, 0.3, 0.4], 'max_samples':[None]}

rfc = RandomForestClassifier()
grid = GridSearchCV(estimator=rfc, param_grid=params, n_jobs=-1)
start = time.time()
grid.fit(x_classifier, y_classifier)
end = time.time()
print(grid.best_estimator_, grid.best_index_, grid.best_score_)
print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')

with open('premier_league_rfc.txt', 'w') as f:
    f.write(f'Best estimator {grid.best_estimator_} \n, Best index {grid.best_index_} \n, Best parameters {grid.best_params_} \n, Best Score {grid.best_score_} \n' , )

params = {'penalty':['l2', 'l2', 'elasticnet'], 'dual':[True,False], 'C':[1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0], 'fit_intercept':[True, False], 'intercept_scaling':[2, 3, 4, 5, 6], 'class_weight':['balanced'], 'random_state':[42], 'solver':['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'], 'max_iter':[100], 'multi_class':['auto', 'ovr', 'multinomial'], 'warm_start':[True, False]}

log = LogisticRegression()
grid = GridSearchCV(estimator=log, param_grid=params, n_jobs=-1)
start = time.time()
grid.fit(x_classifier, y_classifier)
end = time.time()
print(grid.best_estimator_, grid.best_index_, grid.best_score_)
print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')

with open('premier_league_log.txt', 'w') as f:
    f.write(f'Best estimator {grid.best_estimator_} \n, Best index {grid.best_index_} \n, Best parameters {grid.best_params_} \n, Best Score {grid.best_score_} \n' , )

params = {'var_smoothing':[ 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]}

gb = GaussianNB()
grid = GridSearchCV(estimator=gb, param_grid=params, n_jobs=-1)
start = time.time()
grid.fit(x_classifier, y_classifier)
end = time.time()
print(grid.best_estimator_, grid.best_index_, grid.best_score_)
print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')

with open('premier_league_gb.txt', 'w') as f:
    f.write(f'Best estimator {grid.best_estimator_} \n, Best index {grid.best_index_} \n, Best parameters {grid.best_params_} \n, Best Score {grid.best_score_} \n' , )

params = {'solver':['svd', 'lsqr', 'eigen'], 'shrinkage':[0.7, 0.8, 0.9, 1, 1.2, 1.3, 1.4, 1.5], 'store_covariance':[True, False], 'tol':[0.0001], 'covariance_estimator':[None]}

lda = LinearDiscriminantAnalysis()
grid = GridSearchCV(estimator=lda, param_grid=params, n_jobs=-1)
start = time.time()
grid.fit(x_classifier, y_classifier)
end = time.time()
print(grid.best_estimator_, grid.best_index_, grid.best_score_)
print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')

with open('premier_league_lda.txt', 'w') as f:
    f.write(f'Best estimator {grid.best_estimator_} \n, Best index {grid.best_index_} \n, Best parameters {grid.best_params_} \n, Best Score {grid.best_score_} \n' , )



params = {'C':[1.3, 1.4, 1.5, 1.6, 1.7], 'kernel':['rbf', 'poly', 'linear', 'sigmoid'], "degree":[1, 2], 'gamma':['scale', 'auto'], 'coef0':[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'shrinking':[True, False], 'probability':[True, False], 'verbose':[True], 'max_iter':[-1], 'decision_function_shape':['ovr', 'ovo'], 'break_ties':[True, False], 'random_state':[42]}


svc = SVC()
grid = GridSearchCV(estimator=svc, param_distributions=params, n_jobs=-1)
start = time.time()
grid.fit(x_classifier, y_classifier)
end = time.time()
print(grid.best_estimator_, grid.best_index_, grid.best_score_)
print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')

with open('premier_league_svc.txt', 'w') as f:
    f.write(f'Best estimator {grid.best_estimator_} \n, Best index {grid.best_index_} \n, Best parameters {grid.best_params_} \n, Best Score {grid.best_score_} \n' , )
# Averaging regression models with weighted averages to stack them

# Building a voting/meta model for the result classifications

