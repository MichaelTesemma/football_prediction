import pandas as pd
from data import x_classifier_all_full_scaled, y_classifier_all_full, y_classifier_premier, x_classifier_premier_scaled
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import  MLPClassifier
from testing import grid

all_data = pd.read_csv('cleaned_imputed.csv')
all_data = all_data[['HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']]

knn = KNeighborsClassifier(leaf_size=10, n_neighbors=3, p=6, weights='distance') #0.6
rfc = RandomForestClassifier(ccp_alpha=0.2, class_weight='balanced_subsample', criterion='entropy', min_samples_leaf=8, min_weight_fraction_leaf=0.4, n_estimators=60, random_state=42, warm_start=True) #0.5
log = LogisticRegression(C=1.5, class_weight='balanced', random_state=42, solver='liblinear', warm_start=True) #0.44
gb = GaussianNB(var_smoothing=0.055) #4.8
sgd = SGDClassifier(alpha=0.003, average=True, early_stopping=True, eta0=0.7, l1_ratio=1.0, learning_rate='adaptive', loss='squared_hinge', max_iter=10000, n_iter_no_change=10, penalty='l1', power_t=1.2, random_state=42, shuffle=False) #6.4
lda = LinearDiscriminantAnalysis(shrinkage=1, solver='lsqr', store_covariance=True) #0.45
svc =SVC(C=1.4, coef0=0.4, decision_function_shape='ovo', degree=1, kernel='poly', random_state=42) #0.5
mlp = MLPClassifier(activation='logistic', early_stopping=True,hidden_layer_sizes=(17, 105, 3), learning_rate='invscaling', max_iter=1000000, validation_fraction=0.2, verbose=True)

knn.fit(x_classifier_all_full_scaled, y_classifier_all_full)
all_data['knn'] = knn.predict(x_classifier_all_full_scaled)

rfc.fit(x_classifier_all_full_scaled, y_classifier_all_full)
all_data['rfc'] = rfc.predict(x_classifier_all_full_scaled)

log.fit(x_classifier_all_full_scaled, y_classifier_all_full)
all_data['log'] = log.predict(x_classifier_all_full_scaled)

gb.fit(x_classifier_all_full_scaled, y_classifier_all_full)
all_data['gb'] = gb.predict(x_classifier_all_full_scaled)

sgd.fit(x_classifier_all_full_scaled, y_classifier_all_full)
all_data['sgd'] = sgd.predict(x_classifier_all_full_scaled)

lda.fit(x_classifier_all_full_scaled, y_classifier_all_full)
all_data['lda'] = lda.predict(x_classifier_all_full_scaled)

svc.fit(x_classifier_all_full_scaled, y_classifier_all_full)
all_data['svc'] = svc.predict(x_classifier_all_full_scaled)

mlp.fit(x_classifier_all_full_scaled, y_classifier_all_full)
all_data['mlp'] = mlp.predict(x_classifier_all_full_scaled)

all_data['result'] = y_classifier_all_full

all_data.to_csv('stack_data_with_mlp.csv')

# Hidden layer size is determined by Nh = Ns * (a * (Ni + No)) where Ni = number of neurons No = number of output Ns = number of samples in dataset
# mlp = MLPClassifier()
# params = {'activation':['logistic'], 'early_stopping':[True], "hidden_layer_sizes":[(17,90, 3),(17,95, 3), (17, 90, 3),(17,100, 3), (17,105, 3)], 'learning_rate':['invscaling'], 'max_iter':[1000000], 'validation_fraction':[0.2], 'verbose':[True]}
# mlp_grid = grid(model=mlp, params=params, file='premier_league_mlp.txt', x_classifier=all_data, y_classifier=y_classifier_all_full)
# mlp_grid.run()
