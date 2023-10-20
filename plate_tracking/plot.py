#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(style='darkgrid')
sns.set_palette(palette='Set1')

# TODO: Constants from args

TRACKING_ID = 0
PICKLE_FILE = "samples/cut/024_dl_4_reps.pkl"
# PICKLE_FILE = "plate_tracking/samples/cut/024_dl_4_reps.pkl"

# if __name__ == "__main__":
df = pd.read_pickle(PICKLE_FILE)

for i in df['id'].unique():
    if i == 0:
        value_vars = ['x_raw', 'x_filtered', 'y_raw', 'y_filtered']
        dfi = df.query(f'id == {i}').drop(columns=['id'])

        # Reshape the dataframe into a long format using pd.melt
        dfi = pd.melt(dfi, id_vars=['time'], var_name='variable', value_name='value')
        dfi['filtered'] = dfi['variable'].str.contains('_filtered')
        dfi['Position'] = dfi['variable'].str.extract(r'([xy])')
        dfi = dfi.drop(columns=['variable'])
        dfi = dfi[['time', 'filtered', 'Position', 'value']]

        filtered = dfi.query('filtered == True').drop(columns=['filtered'])
        filtered['Position'] = filtered['Position'].map(lambda x: f'{x} filtered')
        raw = dfi.query('filtered == False').drop(columns=['filtered'])
        raw['Position'] = raw['Position'].map(lambda x: f'{x} raw')

        # TODO: Title, speeds
        sns.lineplot(
            filtered,
            x='time',
            y='value',
            hue='Position',
            alpha=1,
            linewidth=1
        )
        sns.lineplot(
            raw,
            x='time',
            y='value',
            hue='Position',
            alpha=0.4,
            linewidth=1
        )

        plt.ylim((0,1))
        plt.xlabel('Time [s]')
        plt.ylabel('Normalized image coordinates')

# df_wide = df.pivot(index
# sns.lineplot(df, x='time', y='y')
# %%
