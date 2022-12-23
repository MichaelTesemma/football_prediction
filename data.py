import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

all_data = pd.read_csv('cleaned_imputed.csv')

# full dataset without any scaling

x_regressor_all_full = all_data[['HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']]
y_regressor_all_full = all_data['FTHG']

# scaled regression dataset

scaler = MinMaxScaler()
x_regressor_all_full_scaled = scaler.fit_transform(x_regressor_all_full)
x_train_regressor, y_train_regressor, x_test_regressor, y_test_regressor = train_test_split(x_regressor_all_full, y_regressor_all_full, train_size=0.2)

x_classifier_all_full = all_data[['HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']]
y_classifier_all_full = all_data['Result']
label = LabelEncoder()
y_classifier_all_full = label.fit_transform(y_classifier_all_full)

# scaled classification dataset

scaler = MinMaxScaler()
x_classifier_all_full_scaled = scaler.fit_transform(x_classifier_all_full)
x_train_classifier, y_train_classifier, x_test_classifier, y_test_classifier = train_test_split(x_classifier_all_full, y_classifier_all_full, train_size=0.2)

# scaled small dataset

x_classifier_50_scaled = x_classifier_all_full_scaled[:50]
y_classifier_50 = y_classifier_all_full[:50]

# unscaled small dataset

x_classifier_50 = x_classifier_all_full[:50]
y_classifier_50 = y_classifier_all_full[:50]

# premier league set

premier_league_data = pd.read_csv('cleaned_premier_league.csv')

# full dataset without any scaling

x_regressor_premier = premier_league_data[['HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']]
y_regressor_premier = premier_league_data['FTHG']

# scaled regression dataset

scaler = MinMaxScaler()
x_regressor_premier_scaled = scaler.fit_transform(x_regressor_premier)
x_train_regressor, y_train_regressor, x_test_regressor, y_test_regressor = train_test_split(x_regressor_premier, y_regressor_premier, train_size=0.2)

x_classifier_premier = premier_league_data[['HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']]
y_classifier_premier = premier_league_data['Result']


label = LabelEncoder()
y_classifier_premier = label.fit_transform(y_classifier_premier)


# scaled classification dataset

scaler = MinMaxScaler()
x_classifier_premier_scaled = scaler.fit_transform(x_classifier_premier)
x_train_classifier_premier, y_train_classifier_premier, x_test_classifier_premier, y_test_classifier_premier = train_test_split(x_classifier_premier, y_classifier_premier, train_size=0.2)

# scaled small dataset

x_classifier_50_scaled = x_classifier_premier_scaled[:50]
y_classifier_50 = y_classifier_premier[:50]

# unscaled small dataset

x_classifier_50 = x_classifier_premier[:50]
y_classifier_50 = y_classifier_premier[:50]

# onehot encoded classifier

x_classifier_premier_onehot_encoded = pd.get_dummies(x_classifier_premier, columns=['HST', 'AST', 'HC', 'AC'])
x_classifier_premier_onehot_encoded_scaled = scaler.fit_transform(x_classifier_premier_onehot_encoded)
x_classifier_premier_onehot_encoded_50 = x_classifier_premier_onehot_encoded[:50]
x_classifier_premier_onehot_encoded_scaled_50 = x_classifier_premier_onehot_encoded_scaled[:50]



