import pandas as pd
from os import listdir

folder = listdir('data1')

df_list = []
for file in folder:
    check = pd.read_csv(f'data/{file}')
    if 'HomeTeam' and 'AwayTeam' and 'FTHG' 'FTAG' and 'HY' and 'HST' and 'AST' and 'HC' and 'HF' and 'AF' and 'HY' and 'AY' and 'HR' and 'AR' not in check.columns:

        print('no dice')
    else:
        df_list.append(pd.read_csv(f'data/{file}'))
        print('done')

df = pd.concat(df_list)
df = df.reset_index(drop=True)

df = pd.DataFrame().assign(Result=df['FTR'],HomeTeam=df['HomeTeam'], AwayTeam=df['AwayTeam'], FTHG=df['FTHG'], FTAG=df['FTAG'], HST=df['HST'], AST=df['AST'], HC=df['HC'], AC=df['AC'], HF=df['HF'], AF=df['AF'], HY=df['HY'], AY=df['AY'], HR=df['HR'], AR=df['AR'])
df.to_csv('cleaned_premier_league.csv')




# for file in folder:
#     df = pd.read_csv(f'data/{file}')
#     df = pd.DataFrame().assign(HomeTeam=df['HomeTeam'], AwayTeam=df['AwayTeam'], FTHG=df['FTHG'], FTAG=df['FTAG'],   HY=df['HY'], AY=df['AY'], HR=df['HR'], AR=df['AR'])
#     df.to_csv('cleaned.csv')
# li = []
# for file in folder:
#     df = pd.read_csv(f'data/{file}')
#     if 'HST' and 'AST' not in df.columns:
#         li.append(file)
#         print(len(li), file)