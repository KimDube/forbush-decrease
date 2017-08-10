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
# Return: dates - array of floats containing decimal dates
#         counts - array of corrected neutron counts corresponding to dates
def loadneutrondata(station):
    station_list = ['OULU']
    file_locs = ['OULU_2002_01_01']
    data_file = station_list.index(station)
    open_file = open(file_locs[data_file], "r")

    counts = []  # Initialize arrays
    dates = []

    lines = open_file.readlines()
    for i in range(len(lines)):
        s = lines[i].split()
        dates.append(float(s[2]))
        counts.append(float(s[4]))

    return np.array(dates), np.array(counts)


# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
dates, counts = loadneutrondata("OULU")
counts = 100 * (counts - np.nanmean(counts)) / np.nanmean(counts)

# 90 day running mean
y = pd.Series(counts)
mean90days = y.rolling(center=True, window=90).mean()

diff = counts - mean90days
below5 = np.where(np.abs(diff) >= 3)
below5days = np.array(dates[below5])
for i in range(len(below5days)):
    print(convert_partial_year(below5days[i]))

plt.plot(dates, counts)
plt.plot(dates, mean90days, 'r')
plt.plot(dates, diff, 'c')
plt.title("Oulu - Daily Neutron Counts")
plt.show()