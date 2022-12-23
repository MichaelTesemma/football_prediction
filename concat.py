from os import listdir
import pandas as pd

folder = listdir('data1')
work = []


# print(len(folder))
# df_list = []
# for i in folder:
#     df_list.append(pd.read_csv(f'data/{i}'))

# df = pd.concat(df_list)
# df = df.reset_index(drop=True)
# df.to_csv('compiled.csv')

print(len(folder))
df_list = []
for i in folder:
    df_list.append(pd.read_csv(f'data/{i}'))

df = pd.concat(df_list)
df = df.reset_index(drop=True)
df.to_csv('compiled_premier_league.csv')
