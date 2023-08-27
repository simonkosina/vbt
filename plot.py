import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

DATA_FILE = 'heights.csv'


def animate(i):
    data = pd.read_csv(DATA_FILE)

    xs = data['time']
    ys = data['height']

    plt.cla()

    plt.plot(xs, ys, label='Height')
    plt.legend(loc='upper left')
    plt.tight_layout()


if __name__ == "__main__":
    anim = animation.FuncAnimation(plt.gcf(), animate, interval=100)

    plt.tight_layout()
    plt.show()
