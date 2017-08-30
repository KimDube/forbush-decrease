
"""
Superposed epoch analysis of Forbush decreases events from IZMIRAN database
Author: Kimberlee Dube
August/ September 2017
"""

from arglevel3.data_sources import osiris_nc
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns
import datetime
from ForbushDecrease import FDidentification as fdi


# -----------------------------------------------------------------------------
def findmeangcr(station, start_days):
    dates, counts = fdi.loadneutrondata(station)
    for i in range(len(dates)):  # convert strings to match start/end_days
        dates[i] = dates[i].replace('.', '-')
    # search for start_days and store count values
    gcrarr = np.zeros((41, len(start_days)))
    for i in range(len(start_days)):
        x = np.where(dates == start_days[i])
        x = int(x[0][0])
        for j in range(41):
            gcrarr[j, i] = counts[x+j]

        background = np.nanmean(gcrarr[0:14, i])
        gcrarr[:, i] = 100 * (gcrarr[:, i] - background) / background
    """
    fig, ax = plt.subplots()
    sns.set_context("talk")
    cmap = ListedColormap(sns.color_palette("RdBu_r", 50))
    fax = ax.contourf(np.arange(-20, 21, 1), np.arange(1, len(start_days) + 1), gcrarr.transpose(), np.arange(-3, 3, 0.1),
                      cmap=cmap, extend='both')
    plt.ylabel("Event Number")
    plt.xlabel("Day from Event")
    plt.title("Forbush Decreases: 2006-11")
    plt.gca().invert_yaxis()
    cb = plt.colorbar(fax, orientation='horizontal', fraction=0.2, aspect=30)
    cb.set_label("Neutron Count Variation")
    plt.show()
    """
    gcr_mean = np.zeros(41)  # for each day average events together
    for i in range(41):
        gcr_mean[i] = np.nanmean(gcrarr[i, :])

    return gcr_mean

# -----------------------------------------------------------------------------
# events are those from IZMIRAN db with MagM >= 2.0

'''
# Events with Mag >=2%
event_dates = ['2006-04-03', '2006-07-09', '2006-07-27', '2006-08-07', '2006-08-19', '2006-09-23',
               '2006-10-19', '2006-11-09', '2006-12-08',
               '2007-01-29', '2007-02-12', '2007-03-22', '2007-09-20', '2007-10-18', '2007-10-25', '2007-11-19',
               '2007-12-17', '2008-01-04', '2008-01-31', '2008-03-08', '2008-03-26', '2008-05-20', '2008-05-28',
               '2008-06-14', '2008-06-24', '2008-08-08', '2008-09-03', '2008-11-24', '2008-12-16', '2008-12-30',
               '2009-02-14', '2009-07-22', '2009-08-05', '2009-10-22', '2010-01-19', '2010-02-11', '2010-04-11',
               '2010-05-02', '2010-05-28', '2010-07-14', '2010-08-03', '2010-08-23', '2010-10-04', '2010-10-11',
               '2010-10-22', '2011-02-04', '2011-03-01', '2011-03-10', '2011-04-05', '2011-04-11', '2011-04-29',
               '2011-05-27', '2011-06-04', '2011-06-09', '2011-06-17', '2011-06-22', '2011-07-11', '2011-07-18',
               '2011-07-30', '2011-08-05', '2011-09-26', '2011-10-05', '2011-10-08', '2011-10-24', '2011-11-01',
               '2011-11-28', '2012-01-22', '2012-03-08', '2012-04-12', '2012-05-03', '2012-05-30', '2012-06-16']
'''
# Events with Mag >=3%, 20 days before, 20 days after
event_dates = ['2006-04-13', '2006-07-09', '2006-11-09', '2006-12-08', '2006-12-14', '2007-01-29', '2008-01-04',
               '2008-06-14', '2010-04-05', '2010-05-28', '2010-08-03', '2010-12-12', '2011-02-18', '2011-03-10',
               '2011-03-29', '2011-04-05', '2011-04-11', '2011-06-04', '2011-06-22', '2011-07-11', '2011-08-05']


start_dates = []
end_dates = []
fake_end_dates = []
for i in event_dates:
    s = datetime.datetime.strptime(i, "%Y-%m-%d") + datetime.timedelta(-20)
    e = datetime.datetime.strptime(i, "%Y-%m-%d") + datetime.timedelta(20)
    f = datetime.datetime.strptime(i, "%Y-%m-%d") + datetime.timedelta(41)
    s = s.strftime('%Y-%m-%d')
    start_dates.append(s)
    e = e.strftime('%Y-%m-%d')
    end_dates.append(e)
    f = f.strftime('%Y-%m-%d')
    fake_end_dates.append(f)

print(start_dates)
print(end_dates)
"""
gcrmean = findmeangcr('MOSC', start_dates)
times = np.arange(-20, 21, 1)
sns.set(context="talk", style="ticks", palette='cubehelix')
fig2, ax1 = plt.subplots(figsize=(10, 8))
ax1.plot(times, gcrmean)
ax1.set_xlabel("Day from event")
ax1.set_ylabel("Neutron Count Variation")
plt.show()
"""
numevents = len(event_dates)
eventarr = np.zeros((41, numevents))  # days, num. events

altofinterest = 30.5  # altitude to look at

for i in range(len(start_dates)):
    print(i)
    # x has dimensions [altitude [km], time]
    # contains variables: aerosol, latitude, longitude, angstrom, median_radius
    x = osiris_nc.aer_level2_from_nc(nc_folder="/home/kimberlee/OsirisData/Level2/daily_version600/", version='6.00',
                                     start_date=start_dates[i], end_date=fake_end_dates[i])
    x = x.sel(time=slice(start_dates[i], end_dates[i]))
    print(x.time)

    # select location
    a = x.where(x.latitude < 20)
    b = a.where(a.latitude > -20)
    # Choose either aerosol or angstrom here
    km20 = b.aerosol.sel(altitude=altofinterest)
    # take daily average
    km20_daily = km20.resample('D', dim='time', how='mean')
    km20_daily = np.array(km20_daily)

    background = np.nanmean(km20_daily[0:14])
    km20_daily = 100 * (km20_daily - background) / background  # Variation from background

    # Account for missing data at endpoints that makes series the wrong length
    if i == 5:
        km20_daily = np.append(km20_daily, np.nan)
        km20_daily = np.append(km20_daily, np.nan)
    if i == 1 or i == 11 or i == 16:
        km20_daily = np.append(km20_daily, np.nan)
    if i == 3:
        km20_daily = np.append(km20_daily, np.nan)
        km20_daily = np.append(km20_daily, np.nan)
        km20_daily = np.roll(km20_daily, 2)
    if i == 0 or i == 4:
        km20_daily = np.append(km20_daily, np.nan)
        km20_daily = np.append(km20_daily, np.nan)
        km20_daily = np.append(km20_daily, np.nan)
        km20_daily = np.append(km20_daily, np.nan)
        km20_daily = np.append(km20_daily, np.nan)
    if i == 7:
        km20_daily = np.append(km20_daily, np.nan)
        km20_daily = np.append(km20_daily, np.nan)
        km20_daily = np.append(km20_daily, np.nan)
        km20_daily = np.append(km20_daily, np.nan)
    if i == 20:
        km20_daily = np.append(km20_daily, np.nan)
        km20_daily = np.append(km20_daily, np.nan)
        km20_daily = np.append(km20_daily, np.nan)

    eventarr[:, i] = km20_daily

compm = np.zeros(41)  # average events together for each day
for i in range(41):
    compm[i] = (np.nanmean(eventarr[i, :]))


fig, ax = plt.subplots()
sns.set_context("talk")
cmap = ListedColormap(sns.color_palette("RdBu_r", 50))
fax = ax.contourf(np.arange(-20, 21, 1), np.arange(1, numevents+1), eventarr.transpose(), np.arange(-50, 50, 1),
                  cmap=cmap, extend='both')
plt.ylabel("Event Number")
plt.xlabel("Day from Event")
plt.title("Forbush Decreases: 2006-11")
plt.gca().invert_yaxis()
cb = plt.colorbar(fax, orientation='horizontal', fraction=0.2, aspect=30)
cb.set_label("Aerosol Extinction")
plt.show()

times = np.arange(-20, 21, 1)
sns.set(context="talk", style="ticks", palette='cubehelix')
fig2, ax1 = plt.subplots(figsize=(10, 8))
ax1.plot(times, compm)
ax1.set_title("tropics , %1.1f km" % altofinterest)
ax1.set_xlabel("Day from event")
ax1.set_ylabel("Aerosol Extinction")
plt.show()

