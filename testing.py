import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv('cleaned_imputed_premier_league.csv')

x_regressor = data[['HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']]
y_regressor = data['FTHG']

scaler = MinMaxScaler()
x_regressor = scaler.fit_transform(x_regressor)
x_train_regressor, y_train_regressor, x_test_regressor, y_test_regressor = train_test_split(x_regressor, y_regressor, train_size=0.2)

x_classifier = data[['HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']]
y_classifier = data['Result']

scaler = MinMaxScaler()

x_classifier_50 = x_classifier[:50]
y_classifier_50 = y_classifier[:50]

x_classifier_scaled = scaler.fit_transform(x_classifier)

x_train_classifier, y_train_classifier, x_test_classifier, y_test_classifier = train_test_split(x_classifier, y_classifier, train_size=0.2)
x_classifier_50_scaled = x_classifier_scaled[:50]



class grid:
    def __init__(self, model, params,file, x_classifier, y_classifier):
        self.model = model
        self.params = params
        self.grid = GridSearchCV(estimator=self.model, param_grid=self.params, n_jobs=-1)
        self.x_classifier = x_classifier
        self.y_classifier = y_classifier
        self.file = file


    def run(self):
        self.grid.fit(self.x_classifier, self.y_classifier)
        with open(self.file, 'w') as f:
            f.write(f'Best estimator {self.grid.best_estimator_} \n, Best index {self.grid.best_index_} \n, Best parameters {self.grid.best_params_} \n, Best Score {self.grid.best_score_} \n' , )
            print(f'{self.model} is done')


class random:
    def __init__(self, model, params, file, x_classifier, y_classifier):
        self.model = model
        self.params = params
        self.grid = RandomizedSearchCV(estimator=self.model, param_distributions=self.params, n_iter=50 ,n_jobs=-1)
        self.x_classifier = x_classifier
        self.y_classifier = y_classifier
        self.file = file


    def run(self):
        self.grid.fit(self.x_classifier, self.y_classifier)
        with open(self.file, 'w') as f:
            f.write(f'Best estimator {self.grid.best_estimator_} \n, Best index {self.grid.best_index_} \n, Best parameters {self.grid.best_params_} \n, Best Score {self.grid.best_score_} \n' , )
            print(f'{self.model} is done')
