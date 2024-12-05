import csv
from pathlib import Path
import numpy as np
import numpy.core.defchararray as np_f
import re
import pygsheets
import pandas as pd
import os 
import re


def convert_google_sheet_url(url):
    # Regular expression to match and capture the necessary part of the URL
    pattern = r'https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)(/edit#gid=(\d+)|/edit.*)?'

    # Replace function to construct the new URL for CSV export
    # If gid is present in the URL, it includes it in the export URL, otherwise, it's omitted
    replacement = lambda m: f'https://docs.google.com/spreadsheets/d/{m.group(1)}/export?' + (f'gid={m.group(3)}&' if m.group(3) else '') + 'format=csv'

    # Replace using regex
    new_url = re.sub(pattern, replacement, url)

    return new_url


url = 'https://docs.google.com/spreadsheets/d/1Gjkm2lpoUVVOPRdGAfub_oTlnbpMpZyBUa0ZxiP7Esc/edit?gid=0#gid=0'
new_url = convert_google_sheet_url(url)

print(new_url)

df_new = pd.read_csv(new_url)
df = pd.DataFrame(df_new)

print(df)

#os.makedirs('Spreadsheet_to_CSV_Test_Data', exist_ok=True)  
df.to_csv('/Users/mdorancy/spread-csv-test/spreadsheet_to_csv.csv')  

    