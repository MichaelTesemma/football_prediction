import os
import pandas as pd

def get_fixtures(rawDataPath, yearFrom, yearTo):
    def format_api_url(year):
        base_url = 'https://www.football-data.co.uk/mmz4281/{}{}/EC.csv'
        return base_url.format(str(year)[2:], str(year + 1)[2:])

    def format_file_path(year):
        return os.path.join(rawDataPath, f'{year}-{year + 1}.csv')
    
    for year in range(yearFrom, yearTo + 1):
        url = format_api_url(year)
        file_path = format_file_path(year)
        try:
            if not os.path.exists(file_path) or year > yearTo - 2:
                df = pd.read_csv(url)
                df.to_csv(f'{file_path}', index=False)
                print(f'Saved fixtures for {year}...')
        except:
            print(f'Failed to save fixtures for {year}!')
            continue

get_fixtures('data5', 2000, 2022)