import os
import matplotlib.pyplot as plt


def print_graph(stats, xlabel, ylabel, title, save_dir, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    plt.plot(stats)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_dir)
