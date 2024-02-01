import click
import os
import re
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from math import ceil, floor
from RepCounter import find_concentrics_in_df
from Phase import Phase

sns.set_theme(style='ticks')
sns.set_palette('rocket')
filename_regexp = re.compile(r"""(\S*)  # Match the original video filename
                             _id        # Skip the '_id' part
                             (\d+)      # Match the object tracking id
                             _          # Skip the underscore
                             (\S*)      # Match the model name
                             \.pkl      # Ignore the file extension
                             """, re.VERBOSE)

phase_cmap = {
    Phase.CONCENTRIC: 'C3',
    Phase.ECCENTRIC: 'C1'
}


@click.command()
@click.argument('src', type=str, nargs=-1)
@click.option('--show_fig', is_flag=True, help='Show the figure.')
@click.option('--save_fig', is_flag=True, help='Save the plots for each input file as a PDF to the same directory as the input pickle file.')
@click.option('--plate_diameter', default=0.45, help='Diameter of the weight plate used in meters.', type=float)
@click.option('--fig_dir', default=None, help='Directory for saving the figures.')
def main(src, show_fig, save_fig, plate_diameter, fig_dir):
    """
    Visualize the bar position and speeds over time based
    on the passed in dataframe in the pickle format.
    """
    if fig_dir is not None:
        os.makedirs(fig_dir, exist_ok=True)

    for s in src:
        if not os.path.isfile(s):
            raise FileNotFoundError()

        visualize(s, show_fig, save_fig, plate_diameter, fig_dir)


def visualize(src, show_fig, save_fig, plate_diameter, fig_dir):
    filename = os.path.basename(src)
    result = filename_regexp.match(filename)
    video, tracking_id, model = result.groups()

    df = pd.read_pickle(src)
    df = df.query(f'id == {tracking_id}').drop(columns=['id'])

    df_pos = df.drop(columns=['dx', 'dy', 'norm_plate_height', 'norm_plate_width'])
    df_vel = df.drop(columns=['x', 'y', 'norm_plate_height', 'norm_plate_width']).rename(
        columns={'dx': 'x', 'dy': 'y'})

    # Reshape the dataframe into a long format
    df_pos = pd.melt(df_pos, id_vars=['time'],
                     var_name='variable', value_name='value')
    df_pos['Position'] = df_pos['variable'].str.extract(r'([xy])')
    df_pos = df_pos.drop(columns=['variable'])
    df_pos = df_pos[['time', 'Position', 'value']]

    df_vel = pd.melt(df_vel, id_vars=['time'],
                     var_name='Velocity', value_name='value')

    fig, (pos_ax, vel_ax) = plt.subplots(2, sharex=True, figsize=(12, 8))

    # title = f'{video}, id: {tracking_id}, model: {model}'
    title = 'Bar center position and speed over time in x and y image coordinates.'
    fig.suptitle(title)

    sns.lineplot(
        df_pos,
        x='time',
        y='value',
        hue='Position',
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

    start = df['time'].min()
    end = df['time'].max()

    pos_ylim = pos_ax.get_ylim()
    pos_ax.set(
        ylabel='[Normalized image coordinates]',
        xlabel=None,
        title='Bar position over time, ROM for each concentric phase displayed in [m]',
        ylim=[max(pos_ylim[0] - 0.2, 0), min(pos_ylim[1] + 0.2, 1)],
        xlim=[start, end]
    )
    pos_ax.legend(ncol=4, loc='lower left')

    vel_ylim = vel_ax.get_ylim()
    vel_ax.set(
        ylabel=r'[(Normalized image coordinates)$\cdot$s$^{-1}$]',
        xlabel=None,
        title='Bar speed over time, ACV for each concentric phase displayed in [m/s]',
        xlim=[start, end],
        # ylim=[vel_ylim[0], vel_ylim[1] + 0.2],
    )
    vel_ax.legend(ncol=1, loc='upper left')

    # Display repetition phases as background colors
    phases = find_concentrics_in_df(df, plate_diameter)

    for phase in phases:
        pos_ax.axvspan(xmin=phase.time_start, xmax=phase.time_end,
                       facecolor=phase_cmap[phase.type], alpha=0.2)
        vel_ax.axvspan(xmin=phase.time_start, xmax=phase.time_end,
                       facecolor=phase_cmap[phase.type], alpha=0.2)

        if phase.type == Phase.CONCENTRIC:
            # average concentric velocity [m/s]
            acv = phase.rom / phase.duration
            pos_ax.text(
                x=(phase.time_start + phase.time_end) / 2 + 0.02,
                y=pos_ylim[1] if pos_ax.get_ylim()[1] < 1 else pos_ax.get_ylim()[0] + 0.02,
                s=f'{phase.rom:0.2f}',
                horizontalalignment='center',
                verticalalignment='bottom',
                rotation='vertical',
            )
            vel_ax.text(
                x=(phase.time_start + phase.time_end) / 2 + 0.02,
                y=vel_ylim[1]*0.8,
                s=f'{acv:0.2f}',
                horizontalalignment='center',
                verticalalignment='center',
                rotation='vertical',
            )

    # Create custom legend
    legend_patches = [
        mpatches.Patch(
            color=phase_cmap[Phase.CONCENTRIC], alpha=0.2, label='Concentric'),
        mpatches.Patch(
            color=phase_cmap[Phase.ECCENTRIC], alpha=0.2, label='Eccentric')
    ]

    # Add legend to the plot
    fig.legend(handles=legend_patches)
    plt.xlabel('Time [s]')

    x_max = ceil(vel_ax.get_xlim()[1])
    x_min = floor(vel_ax.get_xlim()[0])
    x_min = x_min - x_min % 5 # round down to nearest multiple of 5
    major_ticks = range(x_min, x_max, 5)
    minor_ticks = range(x_min, x_max, 1)

    plt.xticks(major_ticks, major_ticks, minor=False) 
    plt.xticks(minor_ticks, [], minor=True)

    plt.tight_layout()

    if save_fig:
        filename = f'{os.path.basename(src).split(".")[0]}.pdf'

        if fig_dir is None:
            path = filename
        else:
            path = os.path.join(fig_dir, filename)

        plt.savefig(path)

    if show_fig:
        plt.show()

    plt.close()


if __name__ == "__main__":
    main()
