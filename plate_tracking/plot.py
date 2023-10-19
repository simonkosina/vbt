#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(style="darkgrid")

# TODO: Constants from args

TRACKING_ID = 0
PICKLE_FILE = "samples/cut/016_squat_8_reps.pkl"
# PICKLE_FILE = "plate_tracking/samples/cut/016_squat_8_reps.pkl"

# if __name__ == "__main__":
#%%
df = pd.read_pickle(PICKLE_FILE)

for i in df['id'].unique():
    dfi = df.query(f'id == {i}').drop(columns=['id'])
    dfi = dfi.melt(id_vars=['time'], value_vars=['x', 'y'], var_name='position', value_name='value')
    dfi = dfi.pivot(index='time', columns='position', values='value')

    sns.lineplot(dfi)
# df_wide = df.pivot(index
# sns.lineplot(df, x='time', y='y')
# %%
