import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('19_08/esercizio3/dataset/AirQualityUCI.csv', sep=';', na_values=['NaN', '-200'])

df['CO(GT)'] = df['CO(GT)'].str.replace(',', '.', regex=False)
df['CO(GT)'] = pd.to_numeric(df['CO(GT)'], errors='coerce')
df['Time'] = df['Time'].str.replace('.', ':', regex=False)
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')

df.dropna()

daily_avg = df.groupby(df['DateTime'].dt.date)['CO(GT)'].transform('mean')
weekly_avg = df.groupby(df['DateTime'].dt.to_period('W'))['CO(GT)'].transform('mean')
total_avg = df['CO(GT)'].mean()

df['Qualita Aria daily (CO(GT))'] = df['CO(GT)'] > daily_avg
df['Qualita Aria daily (CO(GT))'] = df['Qualita Aria daily (CO(GT))'].map({True: 'Buona', False: 'Bassa'})

df['Qualita Aria weekly (CO(GT))'] = df['CO(GT)'] > weekly_avg
df['Qualita Aria weekly (CO(GT))'] = df['Qualita Aria weekly (CO(GT))'].map({True: 'Buona', False: 'Bassa'})

df['Qualita Aria total (CO(GT))'] = df['CO(GT)'] > total_avg
df['Qualita Aria total (CO(GT))'] = df['Qualita Aria total (CO(GT))'].map({True: 'Buona', False: 'Bassa'})

df_model = df.dropna(subset=['CO(GT)', 'Qualita Aria daily (CO(GT))', 'Qualita Aria weekly (CO(GT))', 'Qualita Aria total (CO(GT))'])

le = LabelEncoder()
y = le.fit_transform(df_model['Qualita Aria daily (CO(GT))'])

X = df_model[['CO(GT)']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

daily_log = LogisticRegression()
daily_log.fit(X_train, y_train)

y_pred = daily_log.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Weekly

y = le.fit_transform(df_model['Qualita Aria weekly (CO(GT))'])

X = df_model[['CO(GT)']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

weekly_log = LogisticRegression()
weekly_log.fit(X_train, y_train)

y_pred = weekly_log.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))
