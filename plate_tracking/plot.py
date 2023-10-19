#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(style="darkgrid")

# TODO: Constants from args

TRACKING_ID = 0
PICKLE_FILE = "samples/cut/024_dl_4_reps.pkl"
# PICKLE_FILE = "plate_tracking/samples/cut/024_dl_4_reps.pkl"

# if __name__ == "__main__":
#%%
df = pd.read_pickle(PICKLE_FILE)

for i in df['id'].unique():
    if i == 0:
        value_vars = ['x_raw', 'x_filtered', 'y_raw', 'y_filtered']
        dfi = df.query(f'id == {i}').drop(columns=['id'])
        
        # Reshape the dataframe into a long format using pd.melt
        dfi = pd.melt(dfi, id_vars=['time'], var_name='variable', value_name='value')
        dfi['filtered'] = dfi['variable'].str.contains('_filtered')
        dfi['axis'] = dfi['variable'].str.extract(r'([xy])')
        dfi = dfi.drop(columns=['variable'])
        dfi = dfi[['time', 'filtered', 'axis', 'value']]

        sns.lineplot(dfi, x='time', y='value', hue='axis', style='filtered', alpha=0.7)

# df_wide = df.pivot(index
# sns.lineplot(df, x='time', y='y')
# %%
