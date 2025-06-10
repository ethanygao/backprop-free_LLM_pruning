import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import matplotlib.pyplot as plt

def min_max_scaler(data):
    min_val = torch.min(data)
    max_val = torch.max(data)

    data = (data - min_val) / (max_val - min_val)
    return data

def normalize_data(data, target_mean):
    min_val = torch.min(data)
    max_val = torch.max(data)

    data = (data - min_val) / (max_val - min_val+1e-8)

    data_mean_adjusted = torch.mean(data)
    data = target_mean + (data - data_mean_adjusted)

    return data

def sigmap(data, target_mean=None):
    # Normalize the data
    norm_data = (data - data.mean()) / (data.std()+1e-8)

    sigmap_data = torch.sigmoid(norm_data)
    # adjust mean value
    delta_mean_adjusted = target_mean - sigmap_data.mean() if target_mean is not None else 0
    return sigmap_data + delta_mean_adjusted

import numpy as np

def plot_histogram(datasets, labels, title, save_path):
    # Create a new figure with 4 subplots and a specified size
    fig, axs = plt.subplots(2, 2, figsize=(25, 15))

    # Flatten the axs array
    axs = axs.flatten()

    # Plot each dataset
    for ax, data, label in zip(axs, datasets, labels):
        # Convert the tensor to a NumPy array
        data = data.cpu().numpy()

        # Generate 10 equally spaced bins between the minimum and maximum data values
        bins = np.linspace(data.min(), data.max(), 10)

        # Create a histogram with the specified bins
        counts, edges, patches = ax.hist(data, bins=bins)

        # Set the title and labels with a specified font size
        ax.set_title(label, fontsize=14)
        ax.set_xlabel('Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)

        # Set the xticks to be the bin edges
        ax.set_xticks(edges)

         # Label the raw counts and the percentages below the x-axis...
        bin_centers = 0.5 * np.diff(edges) + edges[:-1]
        for count, x in zip(counts, bin_centers):
            # Label the raw counts
            ax.annotate(str(count), xy=(x, count), xycoords=('data', 'data'),
                        xytext=(0, 5), textcoords='offset points', va='bottom', ha='center')

    # Set the main title for the figure with a specified font size
    fig.suptitle(title, fontsize=16)

    # Save the plot to a file
    plt.savefig(save_path)

    # Clear the current figure
    plt.clf()