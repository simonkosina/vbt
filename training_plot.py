"""
Plot training curves for all models in the models directory.
"""

import glob
import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import defaultdict

LOG_DIR = 'models'
FIG_DIR = 'figs'


if __name__ == "__main__":
    sns.set_theme(context='paper', style='ticks')
    regexp = re.compile(r'val_loss: (\d+\.\d+)')
    log_files = glob.glob(os.path.join(LOG_DIR, '*.log'))

    losses = defaultdict(lambda: [])

    for file in log_files:
        with open(file, 'r') as f:
            for line in f.readlines():
                match = re.findall(regexp, line)

                if len(match) == 0:
                    continue

                loss = float(match[0])
                losses[os.path.basename(file).split('.')[0]].append(loss)

    df = pd.DataFrame(losses)
    df = df.reindex(sorted(df.columns), axis=1)
    df['epoch'] = range(1, len(df) + 1)
    df = pd.melt(df, id_vars=['epoch'], var_name='Model', value_name='loss')

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(ax=ax, data=df, x='epoch', y='loss', hue='Model')

    ax.set(xlabel='Epoch', ylabel='Strata na validačnej sade')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'training_plot.pdf'))
    # plt.show()
