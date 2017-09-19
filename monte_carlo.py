
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

# Choose 45 random start dates from neutron counts
rand_starts = np.random.randint(0, high=len(counts), size=45)

"""
sns.set(context="talk", style="darkgrid")
sns.set_color_codes("dark")
# plots histogram. kde=True overlays kernel density estimate (estimate of pdf)
sns.distplot(counts, bins=100, kde=False, norm_hist=False, hist_kws={"range": [-15, 15]}, color='g')
plt.plot([np.nanmean(counts)+1.96*np.nanstd(counts), np.nanmean(counts)+1.96*np.nanstd(counts)], [0, 900], '-b')
plt.plot([np.nanmean(counts)-1.96*np.nanstd(counts), np.nanmean(counts)-1.96*np.nanstd(counts)], [0, 900], '-b')
plt.show()
"""