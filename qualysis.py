"""
Plots comparisons between matching qualysis exports
and the created dfs.
"""

import click
import os
import glob
import re
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from math import ceil, floor
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

filename_regexp = re.compile(r"""(\S*)  # Match the original video filename
                             _id        # Skip the '_id' part
                             (\d+)      # Match the object tracking id
                             _          # Skip the underscore
                             (\S*)      # Match the model name
                             \.pkl.gz   # Ignore the file extension
                             """, re.VERBOSE)


@click.command()
@click.option('--qualysis_dir', default='qualysis_exports', help='Directory containing the qualysis exports.', show_default=True)
@click.option('--df_dir', default='qualysis_dfs', help='Directory containing the dfs.', show_default=True)
@click.option('--show_fig', is_flag=True, help='Show the figure.', show_default=True)
@click.option('--fig_dir', default=None, help='Directory for saving the figures. If not set the figures won\'t be saved.', show_default=True)
@click.option('--plate_diameter', default=0.45, help='Diameter of the weight plate used in meters.', type=float, show_default=True)
def main(qualysis_dir, df_dir, show_fig, fig_dir, plate_diameter):
    """
    Plot comparisons between qualysis exports and the created dfs.
    """

    sns.set_theme(context='paper', style='ticks')
    sns.set_palette('rocket', 2)

    qualysis_files = glob.glob(os.path.join(qualysis_dir, '*.tsv'))
    df_files = glob.glob(os.path.join(df_dir, '*.pkl.gz'))

    if fig_dir is not None:
        os.makedirs(fig_dir, exist_ok=True)

    videos = []
    rx = []
    px = []
    ry = []
    py = []
    mse_x = []
    mse_y = []

    for qualysis_file in qualysis_files:
        # Find the matching df file
        matching_df_file = next((x for x in df_files if os.path.basename(
            x).startswith(os.path.basename(qualysis_file).split('.')[0])), None)

        if matching_df_file is None:
            print(f'No matching df file found for: {qualysis_file}')
            continue

        result = filename_regexp.match(os.path.basename(matching_df_file))

        try:
            video, tracking_id, model = result.groups()
        except:
            continue

        videos.append(video)

        # Load the qualysis export file into a df
        qualysis_df = pd.read_csv(
            qualysis_file,
            delimiter='\t',
            skiprows=11,
            dtype={
                'Frame': int,
                'Time': float,
                'Osa L X': float,
                'Osa L Y': float,
                'Osa L Z': float,
                'Osa R X': float,
                'Osa R Y': float,
                'Osa R Z': float,
            },
            usecols=['Time', 'Osa L X', 'Osa L Z'],
            # converters={'x': lambda x: float(
            #     x.replace(',', '.')), 'y': lambda x: float(x.replace(',', '.'))},
            index_col=False
        )

        # rename columns from 3d to 2d
        qualysis_df = qualysis_df.rename(columns={
            'Time': 'time',
            'Osa L X': 'x',
            'Osa L Z': 'y'
        })

        # mm to m conversion
        qualysis_df['x'] = -qualysis_df['x'] / 1000.0
        qualysis_df['y'] = qualysis_df['y'] / 1000.0

        # Load the matching df
        matching_df = pd.read_pickle(matching_df_file).drop(
            columns=['dx', 'dy'])
        matching_df = matching_df.query(
            f'id == {tracking_id}').drop(columns=['id'])

        # Calculate as running average
        matching_df['norm_plate_width'] = matching_df['norm_plate_width'].rolling(
            window=30, center=False, min_periods=1).mean()
        matching_df['norm_plate_height'] = matching_df['norm_plate_height'].rolling(
            window=30, center=False, min_periods=1).mean()

        matching_df['x'] = matching_df['x'] * \
            plate_diameter / matching_df['norm_plate_width']
        matching_df['y'] = - matching_df['y'] * plate_diameter / \
            matching_df['norm_plate_height']  # inverted in image coordinates
        matching_df = matching_df.drop(
            columns=['norm_plate_width', 'norm_plate_height'])

        # Align the coordinate systems
        y_shift = qualysis_df['y'].mean() - matching_df['y'].mean()
        matching_df['y'] += y_shift

        x_shift = qualysis_df['x'].mean() - matching_df['x'].mean()
        matching_df['x'] += x_shift

        fig, axs = plt.subplots(2, sharex=True, figsize=(8, 4))

        sns.lineplot(ax=axs[0], x='time', y='x',
                     data=qualysis_df, label='Qualysis')
        sns.lineplot(ax=axs[0], x='time', y='x',
                     data=matching_df, label='Vlastné')

        sns.lineplot(ax=axs[1], x='time', y='y',
                     data=qualysis_df, label='Qualysis')
        sns.lineplot(ax=axs[1], x='time', y='y',
                     data=matching_df, label='Vlastné')

        # Format x axis
        x_max = ceil(axs[1].get_xlim()[1])
        major_ticks = range(0, x_max, 5)
        minor_ticks = range(0, x_max, 1)
        plt.xticks(major_ticks, major_ticks, minor=False)
        plt.xticks(minor_ticks, [], minor=True)
        plt.xlim(0, max(qualysis_df['time'].max(), matching_df['time'].max()))
        plt.xlabel('Čas [s]')

        x_lim_diff = axs[0].get_ylim()[1] - axs[0].get_ylim()[0]
        y_lim_diff = axs[1].get_ylim()[1] - axs[1].get_ylim()[0]

        if x_lim_diff > y_lim_diff:
            y_min, y_max = axs[1].get_ylim()
            axs[1].set_ylim = axs[1].set_ylim(y_min - x_lim_diff / 2, y_max + x_lim_diff / 2)
        else:
            x_min, x_max = axs[0].get_ylim()
            axs[0].set_ylim = axs[0].set_ylim(x_min - y_lim_diff / 2, x_max + y_lim_diff / 2)

        # Format y axis
        axs[0].set_ylabel('X [m]')
        axs[1].set_ylabel('Y [m]')

        # Calculate the correlation coefficient
        t_max = min(qualysis_df['time'].max(), matching_df['time'].max())
        t_min = max(qualysis_df['time'].min(), matching_df['time'].min())
        ts = np.linspace(t_min, t_max, int(t_max * 30))  # 30 fps
        x_qualysis = interp1d(
            qualysis_df['time'], qualysis_df['x'], kind='linear')(ts)
        x_meassured = interp1d(
            matching_df['time'], matching_df['x'], kind='linear')(ts)
        y_qualysis = interp1d(
            qualysis_df['time'], qualysis_df['y'], kind='linear')(ts)
        y_meassured = interp1d(
            matching_df['time'], matching_df['y'], kind='linear')(ts)

        result_x = pearsonr(x_qualysis, x_meassured)
        result_y = pearsonr(y_qualysis, y_meassured)

        rx.append(result_x.statistic)
        px.append(result_x.pvalue)
        ry.append(result_y.statistic)
        py.append(result_y.pvalue)
        mse_x.append(mean_squared_error(x_qualysis, x_meassured))
        mse_y.append(mean_squared_error(y_qualysis, y_meassured))

        # axs[0].legend(title=f'$r = {result_x.statistic:.4f}$')
        # axs[1].legend(title=f'$r = {result_y.statistic:.4f}$')

        # Display fig level legend and turn off individual legends
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', ncols=2, framealpha=1.0)

        axs[0].legend().set_visible(False)
        axs[1].legend().set_visible(False)

        # plt.suptitle(f'{video}')
        plt.tight_layout()

        if show_fig:
            plt.show()

        if fig_dir is not None:
            fig.savefig(os.path.join(
                fig_dir, f'{video}_id{tracking_id}_{model}.pdf'))

        plt.close(fig)

    df = pd.DataFrame({
        'video': videos,
        'mse_x': mse_x,
        'mse_y': mse_y,
        'result_x': rx,
        'p_x': px,
        'result_y': ry,
        'p_y': py
    })
    df = df.sort_values(by='video')

    replacament = '\_'
    df['video'] = df[['video']].map(lambda x: f'\\texttt{{{x.replace("_", replacament)}}}')

    df[['mse_x', 'mse_y', 'result_x', 'result_y']] = df[[
        'mse_x', 'mse_y', 'result_x', 'result_y']].applymap('${:.4f}$'.format)

    df[['p_x', 'p_y']] = df[['p_x', 'p_y']].applymap('${:e}$'.format)

    df = df.drop(columns=['p_x', 'p_y'])

    df = df.rename(columns={
        'video': 'Video',
        'mse_x': '$\\text{MSE}_x$',
        'mse_y': '$\\text{MSE}_y$',
        'result_x': '$r_x$',
        'result_y': '$r_y$',
        'p_x': '$p_x$',
        'p_y': '$p_y$'
    })

    latex_table = df.to_latex(index=False)
    print(latex_table)

if __name__ == "__main__":
    main()
