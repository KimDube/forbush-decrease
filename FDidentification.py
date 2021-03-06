
"""
Locate Forbush decreases in neutron counts
Kimberlee Dube
August/ September 2017
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def loadneutrondata(station):
    """
    Load available daily neutron count values for input station.
    :param station: string, 4 letter neutron monitor station code: OULU, MOSC, NEWK, CLIM
    :return: dates - array of floats containing decimal dates
             counts - array of corrected neutron counts corresponding to dates
    """
    station_list = ['OULU', 'MOSC', 'NEWK', 'CLIM']
    file_locs = ['OULU_2002_2017.txt', 'MOSC_2002_2017.txt', 'NEWK_2002_2017.txt',
                 'CLIM_2002_2017.txt']
    data_file = station_list.index(station)
    open_file = open('/home/kimberlee/Masters/ForbushDecrease/'+file_locs[data_file], "r")

    counts = []  # Initialize arrays
    dates = []

    lines = open_file.readlines()
    for i in range(len(lines)):
        s = lines[i].split()
        dates.append(str(s[0]))
        counts.append(float(s[2]))

    counts = np.array(counts)
    counts[counts < 1] = np.nan

    return np.array(dates), counts


# -----------------------------------------------------------------------------
def findevents(dates, counts):
    """
    Classify minimums in count as times when count decreases by >=3% from 90 day running mean
    :param dates:
    :param counts:
    :return:
    """
    y = pd.Series(counts)
    mean90days = y.rolling(center=True, window=90).mean()
    diff = 100 * (counts - mean90days) / mean90days
    below3 = np.where(diff <= -3)
    below3_dates = np.array(dates[below3])

    # find reference level (average of 14 days before event)
    below3 = below3[0]
    count_change = np.zeros(len(below3))
    for i in range(len(below3)):
        background = np.nanmean(counts[below3[i]-14:below3[i]-1])
        # find percent change in count relative to background
        count_change[i] = 100 * (counts[below3[i]] - background) / background

    return below3_dates, count_change


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    dates, counts = loadneutrondata("CLIM")
    bel3days, bel3counts = findevents(dates, counts)

    a = np.where(bel3counts <= -5)
    bel3days = bel3days[a]
    bel3counts = bel3counts[a]
    for i in range(len(bel3days)):
        print(bel3days[i], bel3counts[i])

    plt.plot(counts)
    plt.title("Oulu - Daily Neutron Counts")
    plt.show()
