from testing import grid, random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from check import accuracy


data = pd.read_csv('stack_data_with_mlp.csv')
results = data['result']
data = data.drop(['knn', 'log', 'rfc', 'gb', 'sgd', 'lda', 'svc', 'result', 'mlp'], axis=1)

# rfc = RandomForestClassifier()
# params = {'ccp_alpha':[ 0.8,0.9, 1.0], 'class_weight':['balanced', 'balanced_subsample'], 'criterion':['gini', 'entropy', 'log_loss'], 'min_samples_leaf':[ 16, 17, 18], 'min_samples_split':[2, 3, 4], 'min_weight_fraction_leaf':[0.4, 0.5, 0.6], 'n_estimators':[ 45, 46, 47], 'random_state':[42], 'warm_start':[True, False]}
# rfc_grid = grid(model=rfc, params=params, file='premier_league_rfc.txt', x_classifier=data, y_classifier=results)
# rfc_grid.run()

# print('donezo')

rfc = RandomForestClassifier(ccp_alpha=0.8, class_weight='balanced', criterion='entropy', min_samples_leaf=20, min_samples_split=5, min_weight_fraction_leaf=0.5, n_estimators=46, random_state=42)
rfc.fit(data, results)
predictions = rfc.predict(data)

right = 0
wrong = 0

for i in results:
    if predictions[i] == results[i]:
        right += 1
    else:
        wrong += 1

print(f'Right results {right}, wrong results {wrong}')

percentage = right * 100 / len(results)

print(f'{percentage}%')
