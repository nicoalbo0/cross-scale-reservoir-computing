import cdsapi
import os
import sys
import zipfile
import time
from calendar import monthrange
import numpy as np

dir = os.getcwd()
FAIL_LOG = f'{dir}/data/sst/failed_days.log'

c = cdsapi.Client()

for year_np in np.arange(1982, 2016):
    for month_np in np.arange(1, 13):

        days = monthrange(year_np, month_np)[1]
        day_list = ['%02d' % d for d in np.arange(1, days + 1)]

        for day in day_list:

            year  = '%02d' % year_np
            month = '%02d' % month_np
            dst   = f'{dir}/data/sst/{year}/{month}/{day}'

            # skip if already downloaded (zip extracted, no leftover zip)
            if os.path.isdir(dst):
                contents = [f for f in os.listdir(dst) if not f.endswith('.zip')]
                if contents:
                    continue

            try:
                os.makedirs(dst, exist_ok=True)
            except OSError:
                sys.exit(f'Cant create: {dst}')

            time.sleep(1)  # avoid excessive traffic

            zip_path = f'{dst}/download.zip'
            try:
                c.retrieve(
                    'satellite-sea-surface-temperature-ensemble-product',
                    {'variable': 'all', 'format': 'zip',
                     'day': day, 'month': month, 'year': year},
                    zip_path)
            except Exception as e:
                with open(FAIL_LOG, 'a') as f:
                    f.write(f'{year}-{month}-{day}\t{type(e).__name__}: {e}\n')
                print(f'[skip] {year}-{month}-{day}: {type(e).__name__}', flush=True)
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                continue

            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(dst)
                os.remove(zip_path)
            except zipfile.BadZipFile as e:
                with open(FAIL_LOG, 'a') as f:
                    f.write(f'{year}-{month}-{day}\tBadZipFile: {e}\n')
                if os.path.exists(zip_path):
                    os.remove(zip_path)
