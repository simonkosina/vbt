import click
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re


sns.set_theme(style='darkgrid')
filename_regexp = re.compile(r"""(\S*)  # Match the original video filename
                             _id        # Skip the '_id' part
                             (\d+)      # Match the object tracking id
                             _          # Skip the underscore
                             (\S*)      # Match the model name
                             \.pkl      # Ignore the file extension
                             """, re.VERBOSE)


@click.command()
@click.argument('src', type=str, nargs=-1)
@click.option('--show_fig', is_flag=True, help='Show the figure.')
@click.option('--save_fig', is_flag=True, help='Save the plots for each input file as a PDF to the same directory as the input pickle file.')
def main(src, show_fig, save_fig):
    """
    Visualize the bar position and speeds over time based
    on the passed in dataframe in the pickle format.
    """
    for s in src:
        if not os.path.isfile(s):
            raise FileNotFoundError()

        visualize(s, show_fig, save_fig)


def visualize(src, show_fig, save_fig):
    filename = os.path.basename(src)
    result = filename_regexp.match(filename)
    video, tracking_id, model = result.groups()

    df = pd.read_pickle(src)
    df = df.query(f'id == {tracking_id}').drop(columns=['id'])

    df_pos = df.drop(columns=['dx', 'dy'])
    df_vel = df.drop(columns=['x_raw', 'x_filtered', 'y_raw', 'y_filtered']).rename(
        columns={'dx': 'x', 'dy': 'y'})

    # Reshape the dataframe into a long format
    df_pos = pd.melt(df_pos, id_vars=['time'],
                     var_name='variable', value_name='value')
    df_pos['filtered'] = df_pos['variable'].str.contains('_filtered')
    df_pos['Position'] = df_pos['variable'].str.extract(r'([xy])')
    df_pos = df_pos.drop(columns=['variable'])
    df_pos = df_pos[['time', 'filtered', 'Position', 'value']]

    filtered = df_pos.query('filtered == True').drop(columns=['filtered'])
    filtered['Position'] = filtered['Position'].map(lambda x: f'{x} filtered')
    raw = df_pos.query('filtered == False').drop(columns=['filtered'])
    raw['Position'] = raw['Position'].map(lambda x: f'{x} raw')

    df_vel = pd.melt(df_vel, id_vars=['time'],
                     var_name='Velocity', value_name='value')

    fig, (pos_ax, vel_ax) = plt.subplots(2, sharex=True, figsize=(8, 8))

    title = f'{video}, id: {tracking_id}, model: {model}'
    fig.suptitle(title)

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
    pos_ax.legend(ncol=4, loc='upper left')
    vel_ax.set(
        ylabel=r'[(Normalized image coordinates)$\cdot$s$^{-1}$]',
        xlabel=None,
        title='Bar speed over time'
    )

    plt.xlabel('Time [s]')
    plt.tight_layout()

    if save_fig:
        path = f'{src.split(".")[0]}.pdf'
        plt.savefig(path)

    if show_fig:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    main()
