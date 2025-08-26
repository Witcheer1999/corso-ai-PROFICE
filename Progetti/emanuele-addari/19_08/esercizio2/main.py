import pandas as pd

df = pd.read_csv('19_08/esercizio3/dataset/AirQualityUCI.csv', sep=';', na_values=['NaN', '-200'])

df['CO(GT)'] = df['CO(GT)'].str.replace(',', '.', regex=False)
df['CO(GT)'] = pd.to_numeric(df['CO(GT)'], errors='coerce')
df['Time'] = df['Time'].str.replace('.', ':', regex=False)
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')

df.dropna()

top_3 = df.nlargest(3, 'CO(GT)')
print(top_3[['DateTime', 'CO(GT)']])

daily_avg = df.groupby(df['DateTime'].dt.date)['CO(GT)'].transform('mean')

df['Qualita Aria daily (CO(GT))'] = df['CO(GT)'] > daily_avg
df['Qualita Aria daily (CO(GT))'] = df['Qualita Aria daily (CO(GT))'].map({True: 'Buona', False: 'Bassa'})

weekly_avg = df.groupby(df['DateTime'].dt.to_period('W'))['CO(GT)'].transform('mean')

df['Qualita Aria weekly (CO(GT))'] = df['CO(GT)'] > weekly_avg
df['Qualita Aria weekly (CO(GT))'] = df['Qualita Aria weekly (CO(GT))'].map({True: 'Buona', False: 'Bassa'})

total_avg = df['CO(GT)'].mean()
df['Qualita Aria total (CO(GT))'] = df['CO(GT)'] > total_avg
df['Qualita Aria total (CO(GT))'] = df['Qualita Aria total (CO(GT))'].map({True: 'Buona', False: 'Bassa'})

print(pd.crosstab(df['Qualita Aria daily (CO(GT))'], df['Qualita Aria weekly (CO(GT))'], normalize='all')*100)
print(pd.crosstab(df['Qualita Aria daily (CO(GT))'], df['Qualita Aria total (CO(GT))'], normalize='all')*100)
print(pd.crosstab(df['Qualita Aria weekly (CO(GT))'], df['Qualita Aria total (CO(GT))'], normalize='all')*100)

perc_bassa_global = (df['Qualita Aria weekly (CO(GT))'] == 'Bassa').mean() * 100
print(f'Percentuale di settimane con Qualita Aria Bassa: {perc_bassa_global}%')