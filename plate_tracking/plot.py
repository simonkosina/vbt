#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(style='darkgrid')

# TODO: Args from constants

TRACKING_ID = 0
# PICKLE_FILE = "samples/raw/021_bp_8_reps.pkl"
PICKLE_FILE = "samples/raw/019_squat_4_reps.pkl"

df = pd.read_pickle(PICKLE_FILE)
df = df.query('id == 0').drop(columns=['id'])

df_pos = df.drop(columns=['dx', 'dy'])
df_vel = df.drop(columns=['x_raw', 'x_filtered', 'y_raw', 'y_filtered']).rename(columns={'dx': 'x', 'dy': 'y'})

# Reshape the dataframe into a long format
df_pos = pd.melt(df_pos, id_vars=['time'], var_name='variable', value_name='value')
df_pos['filtered'] = df_pos['variable'].str.contains('_filtered')
df_pos['Position'] = df_pos['variable'].str.extract(r'([xy])')
df_pos = df_pos.drop(columns=['variable'])
df_pos = df_pos[['time', 'filtered', 'Position', 'value']]

filtered = df_pos.query('filtered == True').drop(columns=['filtered'])
filtered['Position'] = filtered['Position'].map(lambda x: f'{x} filtered')
raw = df_pos.query('filtered == False').drop(columns=['filtered'])
raw['Position'] = raw['Position'].map(lambda x: f'{x} raw')

df_vel = pd.melt(df_vel, id_vars=['time'], var_name='Velocity', value_name='value')

fig, (pos_ax, vel_ax) = plt.subplots(2, sharex=True, figsize=(8,8))
fig.suptitle(PICKLE_FILE)

# TODO: Title, speeds, reps?
sns.lineplot(
    filtered,
    x='time',
    y='value',
    hue='Position',
    ax=pos_ax,
    palette='rocket'
)
sns.lineplot(
    raw,
    x='time',
    y='value',
    hue='Position',
    alpha=0.4,
    ax=pos_ax,
    palette='rocket'
)
sns.lineplot(
    df_vel,
    x='time',
    y='value',
    hue='Velocity',
    ax=vel_ax,
    palette='rocket'
)

pos_ax.set(
    ylabel='[Normalized image coordinates]',
    xlabel=None,
    title='Bar position over time',
)
pos_ax.legend(ncol=4)
vel_ax.set(
    ylabel=r'[(Normalized image coordinates)$\cdot$s$^{-1}$]',
    xlabel=None,
    title='Bar speed over time'
)

plt.xlabel('Time [s]')
plt.tight_layout()
# %%
