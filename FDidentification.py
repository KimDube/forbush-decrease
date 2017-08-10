# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Locate forbush decreases in neutron counts
# Kimberlee Dube
# August 2017
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta, datetime


# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Parameter: number - float, Decimal date, eg. 2016.35
# Return: Input number as datetime, eg. 2016-05-08
def convert_partial_year(number):
    year = int(number)
    d = timedelta(days=(number - year)*365)
    day_one = datetime(year, 1, 1)
    date = d + day_one
    return date


# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Load available daily neutron count values for input station.
# Parameter: station - string, 4 letter neutron monitor station code
# options: Oulu, Moscow, Newark
# Return: dates - array of floats containing decimal dates
#         counts - array of corrected neutron counts corresponding to dates
def loadneutrondata(station):
    station_list = ['OULU', 'MOSC', 'NEWK', 'CLIM']
    file_locs = ['OULU_2002_2017.txt', 'MOSC_2002_2017.txt', 'NEWK_2002_2017.txt',
                 'CLIM_2002_2017.txt']
    data_file = station_list.index(station)
    open_file = open(file_locs[data_file], "r")

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


# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def findevents(dates, counts):
    counts_n = 100 * (counts - np.nanmean(counts)) / np.nanmean(counts)  # normalize

    # Classify minimums in count as times when count decreases by >=3% from 90 day running mean
    y = pd.Series(counts_n)
    mean90days = y.rolling(center=True, window=90).mean()
    diff = counts_n - mean90days
    below3 = np.where(np.abs(diff) >= 3)
    below3_dates = np.array(dates[below3])

    # find reference level (average of 14 days before event)
    below3 = below3[0]
    count_change = np.zeros(len(below3))
    for i in range(len(below3)):
        background = np.nanmean(counts[below3[i]-14:below3[i]-1])
        # find relative percent change in count
        count_change[i] = 100 * (counts[below3[i]] - background) / background

    return below3_dates, count_change


# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
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