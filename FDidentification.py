# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Locate forbush decreases in neutron counts
# Kimberlee Dube
# August 2017
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta, datetime

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def convert_partial_year(number):
    year = int(number)
    d = timedelta(days=(number - year)*(365))
    day_one = datetime(year,1,1)
    date = d + day_one
    return date


counts = np.zeros(5695)
dates = np.zeros(5695)
text_file = open("OULU_2002_01_01", "r")
lines = text_file.readlines()
for i in range(len(lines)):
    s = lines[i].split()
    dates[i] = s[2]
    counts[i] = s[4]

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