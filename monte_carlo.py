
"""
Significance testing of epoch analysis using random numbers.
Apparently not real Monte Carlo.
Kimberlee Dube
September, 2017
"""

import numpy as np
import matplotlib.pyplot as plt
from ForbushDecrease import epochanalysis_IZMIRAN as epi
import seaborn as sns


def significance_test(time_series, epoch_length, num_events, num_iterations):

    store_iters = np.zeros((epoch_length, num_iterations))

    for i in range(num_iterations):
        # Choose 45 random event start dates from neutron counts
        # limit on max value so event lengths are consistent
        rand_starts = np.random.randint(0, high=len(time_series) - epoch_length, size=num_events)

        # Create event array
        rand_event_arr = np.zeros((epoch_length, num_events))
        for j in range(num_events):
            rand_event_arr[:, j] = time_series[rand_starts[j]:rand_starts[j] + epoch_length]

        # Average over each day
        rand_event_average = np.zeros(epoch_length)  # average events together for each day
        for j in range(epoch_length):
            rand_event_average[j] = np.nanmean(rand_event_arr[j, :])

        store_iters[:, i] = rand_event_average

    means = np.zeros(epoch_length)
    standevs = np.zeros(epoch_length)

    # For each column in store_iters (corresponding to each day) plot histogram
    for i in range(epoch_length):
        c = store_iters[i, :]
        means[i] = np.nanmean(c)
        standevs[i] = np.nanstd(c)

        """
        sns.set(context="talk", style="darkgrid")
        sns.set_color_codes("dark")
        # plots histogram. kde=True overlays kernel density estimate (estimate of pdf)
        sns.distplot(c, bins=100, kde=False, norm_hist=False, hist_kws={"range": [-5, 5]}, color='g')
        plt.plot([np.nanmean(c) + 1.96 * np.nanstd(c), np.nanmean(c) + 1.96 * np.nanstd(c)], [0, 10], '-b')
        plt.plot([np.nanmean(c) - 1.96 * np.nanstd(c), np.nanmean(c) - 1.96 * np.nanstd(c)], [0, 10], '-b')
        plt.show()
        """

    return means, standevs

# -----------------------------------------------------------------------------
if __name__ == "__main__":

    epl = 50
    ne = 45
    nits = 10000

    open_file = open('/home/kimberlee/Masters/ForbushDecrease/MOSC_2002_2017.txt', "r")
    counts = []  # Initialize arrays
    dates = []
    lines = open_file.readlines()
    for j in range(len(lines)):
        spl = lines[j].split()
        dates.append(str(spl[0]))
        counts.append(float(spl[2]))
    counts = np.array(counts)
    counts[counts < 1] = np.nan

    counts = epi.anomalize(counts)  # find anomaly

    mean, std = significance_test(counts, epl, ne, nits)

    sns.set(context="talk", style="darkgrid")
    plt.plot(mean + 1.96*std)
    plt.plot(mean - 1.96*std)
    plt.show()
