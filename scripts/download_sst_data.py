import cdsapi
import glob
import os
import zipfile
import time
from calendar import monthrange
import numpy as np

dir = os.getcwd()

c = cdsapi.Client()

for year_np in np.arange(1982,2016):
    month_list_np = np.arange(1,13)

    for month_np in month_list_np:

        days = monthrange(year_np,month_np)[1]
        daysarray = np.arange(1,days+1)

        nparraytostr = lambda daysarray:'%02d'%daysarray
        day_list = list(map(nparraytostr,daysarray))

        for day in day_list:

            year = '%02d'%year_np
            month = '%02d'%month_np
            
            # to avoid excessive traffic    
            time.sleep(1)

            day_dir = f'{dir}/data/sst/{year}/{month}/{day}'

            try:
                os.makedirs(day_dir, exist_ok=True)
            except OSError as e:
                sys.exit(f'Cant create: {day_dir}')

            # skip if this day was already downloaded (any .nc file present)
            if glob.glob(f'{day_dir}/*.nc'):
                print(f'skip {year}-{month}-{day} (already downloaded)')
                continue

            # download file
            c.retrieve(
                'satellite-sea-surface-temperature-ensemble-product',
                {
                    'variable': 'all',
                    'format': 'zip',
                    'day': day,
                    'month': month,
                    'year': year,
                },
                f'{dir}/data/sst/{year}/{month}/{day}/download.zip')

            # unzip it
            with zipfile.ZipFile(f'{dir}/data/sst/{year}/{month}/{day}/download.zip', 'r') as zip_ref:
                zip_ref.extractall(f'{dir}/data/sst/{year}/{month}/{day}')

            # delete the zip file
            os.remove(f'{dir}/data/sst/{year}/{month}/{day}/download.zip')
