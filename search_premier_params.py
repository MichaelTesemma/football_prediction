import pandas as pd
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import  MLPClassifier
from sklearn.metrics import accuracy_score
import time
from testing import grid, random, x_classifier_50_scaled, x_classifier_50, y_classifier, y_classifier_50
from data import x_classifier_premier_onehot_encoded_50, x_classifier_premier_onehot_encoded_scaled_50, x_classifier_premier, y_classifier_premier, x_test_classifier_premier, y_test_classifier_premier
from sklearn.ensemble import StackingClassifier

# Random searching the model

# knn = KNeighborsClassifier()
# params = {'n_neighbors':[2,3,4,5,6], 'weights':['distance', 'uniform'], 'leaf_size':[6, 7, 8, 9, 10,11,12,13,14 ], 'p':[1,2, 3, 4]}
# knn_grid = random(model=knn, params=params, file='premier_league_knn.txt', x_classifier=x_classifier_50_scaled, y_classifier=y_classifier_50)
# knn_grid.run()

# sgd = SGDClassifier()
# params = {'loss':['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insentive', 'squared_epsilon_insentive'], 'penalty':['l2', 'l1', 'elasticnet', 'none'], 'alpha':[0.001,0.002, 0.003, 0.004, 0.005], 'l1_ratio':[1.0, 1.5], 'fit_intercept':[True, False], 'max_iter':[10000], 'epsilon':[0.0, 0.1, 0.2, 0.3, 0.4],'shuffle':[True, False], 'random_state':[42], 'learning_rate':['optimal', 'constant', 'invscaling', 'adaptive'], 'eta0':[0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4], 'power_t':[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,], 'early_stopping':[True], 'warm_start':[True], 'average':[True, False], 'n_iter_no_change':[10]}
# sgd_grid = random(model=sgd, params=params, file='premier_league_sgd.txt', x_classifier=x_classifier_50_scaled, y_classifier=y_classifier_50)
# sgd_grid.run()

# rfc = RandomForestClassifier()
# params = {'n_estimators':[30, 40, 50, 60, 70], 'criterion':['gini', 'entropy', 'log_loss'], 'max_depth':[None], 'min_samples_split':[ 3, 4, 5, 6], 'min_samples_leaf':[4, 5, 7, 8, 9], 'min_weight_fraction_leaf':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 'max_features':['sqrt', 'log2'],  'min_impurity_decrease':[0.0, 0.1, 0.2, 0.3, 0.4], 'bootstrap':[True, False], 'oob_score':[False, True], 'random_state':[42], 'warm_start':[False, True], 'class_weight':['balanced', 'balanced_subsample'], 'ccp_alpha':[0.0, 0.1, 0.2, 0.3, 0.4], 'max_samples':[None]}
# rfc_grid = random(model=rfc, params=params, file='premier_league_rfc.txt', x_classifier=x_classifier_50_scaled, y_classifier=y_classifier_50)
# rfc_grid.run()

# log = LogisticRegression()
# params = {'penalty':['l2', 'l2', 'elasticnet'], 'dual':[True,False], 'C':[1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0], 'fit_intercept':[True, False], 'intercept_scaling':[2, 3, 4, 5, 6], 'class_weight':['balanced'], 'random_state':[42], 'solver':['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'], 'max_iter':[100], 'multi_class':['auto', 'ovr', 'multinomial'], 'warm_start':[True, False]}
# log_grid = random(model=log, params=params, file='premier_league_log.txt', x_classifier=x_classifier_50_scaled, y_classifier=y_classifier_50)
# log_grid.run()

# gb = GaussianNB()
# params = {'var_smoothing':[ 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]}
# gb_grid = random(model=gb, params=params, file='premier_league_gb.txt', x_classifier=x_classifier_50_scaled, y_classifier=y_classifier_50)
# gb_grid.run()

# lda = LinearDiscriminantAnalysis()
# params = {'solver':['svd', 'lsqr', 'eigen'], 'shrinkage':[0.7, 0.8, 0.9, 1, 1.2, 1.3, 1.4, 1.5], 'store_covariance':[True, False], 'tol':[0.0001], 'covariance_estimator':[None]}
# lda_grid = random(model=lda, params=params, file='premier_league_lda.txt', x_classifier=x_classifier_50_scaled, y_classifier=y_classifier_50)
# lda_grid.run()

# svc = SVC()
# params = {'C':[1.3, 1.4, 1.5, 1.6, 1.7], 'kernel':['rbf', 'poly', 'linear', 'sigmoid'], "degree":[1, 2], 'gamma':['scale', 'auto'], 'coef0':[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'shrinking':[True, False], 'probability':[True, False], 'verbose':[True], 'max_iter':[-1], 'decision_function_shape':['ovr', 'ovo'], 'break_ties':[True, False], 'random_state':[42]}
# svc_grid = random(model=svc, params=params, file='premier_league_svc.txt', x_classifier=x_classifier_50_scaled, y_classifier=y_classifier_50)
# svc_grid.run()

# Grid searching the models

# knn = KNeighborsClassifier()
# params = {'leaf_size':[10, 9, 8, 7, 6, 5], 'n_neighbors':[2, 3, 4], 'p':[6, 7, 5], 'weights':['distance', 'uniform']}
# knn_grid = grid(model=knn, params=params, file='premier_league_knn.txt', x_classifier=x_classifier_premier_onehot_encoded_scaled_50, y_classifier=y_classifier_50)
# knn_grid.run()

# sgd = SGDClassifier()
# params = {'alpha':[0.003, 0.004, 0.005], 'average':[True, False], 'early_stopping':[True], 'eta0':[0.6, 0.7, 0.8], 'l1_ratio':[1.0, 1.1, 1.2], 'learning_rate':['optimal', 'constant', 'invscaling', 'adaptive'], 'loss':['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insentive', 'squared_epsilon_insentive'], 'max_iter':[10000], 'n_iter_no_change':[10], 'penalty':['l2', 'l1', 'elasticnet', 'none'], 'power_t':[1.2, 1.3, 1.4], 'random_state':[42], 'shuffle':[False, True], 'warm_start':[False, True]}
# sgd_grid = grid(model=sgd, params=params, file='premier_league_sgd.txt', x_classifier=x_classifier_premier_onehot_encoded_scaled_50, y_classifier=y_classifier_50)
# sgd_grid.run()

# rfc = RandomForestClassifier()
# params = {'ccp_alpha':[0.1, 0.2, 0.3], 'class_weight':['balanced', 'balanced_subsample'], 'criterion':['gini', 'entropy', 'log_loss'], 'min_samples_leaf':[8, 9, 10], 'min_samples_split':[2, 3, 4], 'min_weight_fraction_leaf':[0.2, 0.3, 0.4], 'n_estimators':[40, 50, 60], 'random_state':[42], 'warm_start':[True, False]}
# rfc_grid = grid(model=rfc, params=params, file='premier_league_rfc.txt', x_classifier=x_classifier_premier_onehot_encoded_scaled_50, y_classifier=y_classifier_50)
# rfc_grid.run()

# log = LogisticRegression()
# params = {'penalty':['l2', 'l2', 'elasticnet'], 'C':[1.5, 1.6, 1.4], 'class_weight':['balanced'], 'intercept_scaling':[1, 2, 3], 'random_state':[42], 'solver':['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'], 'warm_start':[True, False]}
# log_grid = grid(model=log, params=params, file='premier_league_log.txt', x_classifier=x_classifier_premier_onehot_encoded_scaled_50, y_classifier=y_classifier_50)
# log_grid.run()

# gb = GaussianNB()
# params = {'var_smoothing':[0.055, 0.06, 0.065 ]}
# gb_random = random(model=gb, params=params, file='premier_league_gb.txt', x_classifier=x_classifier_premier_onehot_encoded_scaled_50, y_classifier=y_classifier_50)
# gb_random.run()

# lda = LinearDiscriminantAnalysis()
# params = {'shrinkage': [1, 2, 3], 'solver':['svd', 'lsqr', 'eigen'], 'store_covariance':[True, False]}
# lda_grid = grid(model=lda, params=params, file='premier_league_lda.txt', x_classifier=x_classifier_premier_onehot_encoded_50, y_classifier=y_classifier_50)
# lda_grid.run()

# svc = SVC()
# params = {'C':[1.4, 1.5, 1.6], 'coef0':[0.4, 0.5, 0.6], 'decision_function_shape':['ovo', 'ovr'], 'degree':[1, 2, 3], 'kernel':['rbf', 'poly', 'linear', 'sigmoid'], 'random_state':[42]}
# svc_grid = grid(model=svc, params=params, file='premier_league_svc.txt', x_classifier=x_classifier_premier_onehot_encoded_50, y_classifier=y_classifier_50)
# svc_grid.run()

# Final model

knn = KNeighborsClassifier(leaf_size=10, n_neighbors=3, p=6, weights='distance') #0.6
rfc = RandomForestClassifier(ccp_alpha=0.2, class_weight='balanced_subsample', criterion='entropy', min_samples_leaf=8, min_weight_fraction_leaf=0.4, n_estimators=60, random_state=42, warm_start=True) #0.5
log = LogisticRegression(C=1.5, class_weight='balanced', random_state=42, solver='liblinear', warm_start=True) #0.44
gb = GaussianNB(var_smoothing=0.055) #4.8
sgd = SGDClassifier(alpha=0.003, average=True, early_stopping=True, eta0=0.7, l1_ratio=1.0, learning_rate='adaptive', loss='squared_hinge', max_iter=10000, n_iter_no_change=10, penalty='l1', power_t=1.2, random_state=42, shuffle=False) #6.4
lda = LinearDiscriminantAnalysis(shrinkage=1, solver='lsqr', store_covariance=True) #0.45
svc =SVC(C=1.4, coef0=0.4, decision_function_shape='ovo', degree=1, kernel='poly', random_state=42) #0.5


# Hidden layer size is determined by Nh = Ns*(a * (Ni + No)) where Ni = number of neurons No = number of output Ns = number of samples in dataset
params = {'activation':['tanh'], 'early_stopping':[True], "hidden_layer_sizes":[(17,102, 3)], 'learning_rate':['invscaling'], 'max_iter':[10000], 'validation_fraction':[0.2]}

mlp = MLPClassifier(params)

stack = StackingClassifier(estimators=[knn, rfc, log, gb, sgd, lda, svc], final_estimator=mlp)
stack.fit(x_classifier_premier, y_classifier_premier)

y_prediction = stack.predict(x_test_classifier_premier)
score = accuracy_score(y_test_classifier_premier)
print(score)
