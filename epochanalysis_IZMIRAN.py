
"""
Superposed epoch analysis of Forbush decreases events from IZMIRAN database:
http://spaceweather.izmiran.ru/eng/dbs.html

Event selection: decrease in MagM of at least 3.0%. Event start times are
shifted from those in database so that the minimum count corresponds with time
zero. Events are excluded if there was another event of comparable magnitude
within the epoch (-14 to +35 days).

Author: Kimberlee Dube
September 2017
"""

from arglevel3.data_sources import osiris_nc
from arglevel3.data_sources import omps_2d
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import xarray


# Number of days in each epoch. 14 days (2 weeks) before event to 35 days (5 weeks) after event.
EPOCH_LENGTH = 50

# MagM >= 3%
EVENTS = ['2002-02-04', '2002-03-18', '2002-04-17', '2002-05-23', '2002-08-18', '2003-01-26', '2003-05-29',
          '2003-08-17', '2003-10-30', '2003-11-30', '2004-07-26', '2004-09-13', '2004-11-09', '2005-01-18',
          '2005-05-15', '2005-07-17', '2005-09-11', '2006-07-09', '2006-12-14', '2007-01-29', '2010-08-03',
          '2011-04-11', '2011-06-22', '2011-08-05', '2011-10-30', '2012-01-22', '2012-03-08', '2012-04-04',
          '2012-07-14', '2012-09-04', '2012-10-11', '2012-11-23', '2013-01-16', '2013-03-17', '2013-05-31',
          '2013-07-12', '2014-04-18', '2014-06-17', '2014-09-12', '2014-12-21', '2015-03-17', '2015-05-06',
          '2015-06-22', '2015-09-07', '2015-11-06']

NUM_EVENTS = len(EVENTS)


# -----------------------------------------------------------------------------
def findmeangcr(start_days, contour_plot=0):
    """
    Epoch analysis of GCR from Moscow neutron monitor.
    Epoch start dates are shifted so maximum decrease is at day zero.
    :param start_days:
    :param contour_plot:
    :return:
    """
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

    # First centre data on zero: fig 2a of Laken and Calogovic 2013
    counts = 100 * (counts - np.nanmean(counts)) / np.nanmean(counts)

    # Take anomaly in counts to be difference from 35 day running mean -> high-pass filter.
    y = pd.Series(counts)
    f2 = y.rolling(center=True, window=35).mean()
    u = y - f2
    counts = np.array(u)

    # convert strings to match start_days
    for j in range(len(dates)):
        dates[j] = dates[j].replace('.', '-')
    dates = np.array(dates)
    start_days = np.array(start_days)

    # search for epoch start_days and store corresponding count values for complete epoch
    gcrarr = np.zeros((EPOCH_LENGTH, NUM_EVENTS))
    new_start_days = np.zeros(NUM_EVENTS)
    for j in range(NUM_EVENTS):
        temp = np.where(dates == start_days[j])
        temp = int(temp[0][0])
        for k in range(EPOCH_LENGTH):
            gcrarr[k, j] = counts[temp+k]

        # shift so minimum is at zero and find new epoch.
        d = np.argmin(gcrarr[:, j])
        start_days[j] = datetime.datetime.strptime(start_days[j], "%Y-%m-%d") + datetime.timedelta(int(d)-14)
        new_start_days = start_days[j]
        temp = np.where(dates == start_days[j])
        temp = int(temp[0][0])
        for k in range(EPOCH_LENGTH):
            gcrarr[k, j] = counts[temp+k]

    if contour_plot == 1:
        fig1, ax1 = plt.subplots()
        sns.set_context("talk")
        cmap = ListedColormap(sns.color_palette("RdBu_r", 50))
        fax = ax1.contourf(np.arange(-14, 36, 1), np.arange(1, len(start_days) + 1), gcrarr.transpose(),
                           np.arange(-5, 5, 0.5), cmap=cmap, extend='both')
        plt.ylabel("Event Number")
        plt.xlabel("Day from Event")
        plt.title("Forbush Decreases: 2002-15")
        plt.gca().invert_yaxis()
        cb = plt.colorbar(fax, orientation='horizontal', fraction=0.2, aspect=30)
        cb.set_label("Moscow Neutron Count Anomaly")
        plt.show()

    sns.set(context="talk", style="darkgrid")
    sns.set_palette('hls', 12)
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.plot(np.arange(-14, 36, 1), gcrarr)
    plt.ylabel("Neutron Count Anomaly")
    plt.xlabel("Day from Event")
    plt.title("Forbush Decreases: 2002-15")
    plt.show()

    gcr_mean = np.zeros(EPOCH_LENGTH)  # for each day average events together
    gcr_std = np.zeros(EPOCH_LENGTH)
    for j in range(EPOCH_LENGTH):
        gcr_mean[j] = np.nanmean(gcrarr[j, :])
        gcr_std[j] = np.nanstd(gcrarr[j, :])

    return gcr_mean, gcr_std, new_start_days


# -----------------------------------------------------------------------------
def standardize(xarr):
    return (xarr - xarr.mean()) / xarr.std()


# -----------------------------------------------------------------------------
def longtermanomaly(xarr):
    return (xarr - xarr.mean()) / xarr.mean()


# -----------------------------------------------------------------------------
def monthlyanomaly(xarr):
    """
    Anomaly as difference from monthly mean. Behaves strangely... value is zero
    at end of every year...
    :param xarr: xarray (reduced to 1 variable in time dimension)
    :return: monthly anomaly of xarr
    """
    # TODO: figure out why this function has weird results
    month_arr = np.array(xarr.time.dt.month)  # months corresponding to each daily value
    monthlymeans = xarr.groupby('time.month').mean()  # take mean of each month
    # For each day find the variation from the mean for that month
    for j in range(12):
        c = np.where(month_arr == j)
        xarr[c] = (xarr[c] - monthlymeans[j]) / monthlymeans[j]
    return xarr


# -----------------------------------------------------------------------------
if __name__ == "__main__":

    start_dates = []
    end_dates = []
    for i in EVENTS:
        # look 2 weeks before and 6 weeks after (effects could take up to a month...)
        s = datetime.datetime.strptime(i, "%Y-%m-%d") + datetime.timedelta(-14)
        e = datetime.datetime.strptime(i, "%Y-%m-%d") + datetime.timedelta(42)
        s = s.strftime('%Y-%m-%d')
        start_dates.append(s)
        e = e.strftime('%Y-%m-%d')
        end_dates.append(e)

    # Standard error = std / sqrt(n)
    meangcr, stdgcr, new_start_dates = findmeangcr(start_dates, contour_plot=1)
    sns.set(context="talk", style="darkgrid")
    sns.set_palette('hls', 12)
    plt.plot(np.arange(-14, 36, 1), meangcr)
    plt.plot(np.arange(-14, 36, 1), meangcr + (stdgcr / np.sqrt(NUM_EVENTS)))
    plt.plot(np.arange(-14, 36, 1), meangcr - (stdgcr / np.sqrt(NUM_EVENTS)))
    plt.ylabel("Mean Neutron Count Anomaly")
    plt.xlabel("Day from Event")
    plt.title("Forbush Decreases: 2002-15")
    plt.show()

    """
    altofinterest = 20.5
    # Data only exists up to 2012 for version 6
    # x = osiris_nc.aer_level2_from_nc(nc_folder="/home/kimberlee/OsirisData/Level2/daily_version600/",
    #                                 version='6.00', start_date='2002-01-01', end_date='2013-12-31')
    x = omps_2d.aer_level2_from_nc('/home/kimberlee/ValhallaData/OMPS/OMPS_Tomography_v1.0.2/L2')

    x.to_netcdf(path='/home/kimberlee/Masters/OMPS_all', mode='w')
    # x = xarray.open_dataset('/home/kimberlee/Masters/test_o3658')
    print(x)
    exit()

    a = x.where(x.latitude < 20)  # select location
    b = a.where(a.latitude > -20)
    km20 = b.aerosol.sel(altitude=altofinterest)  # Choose either aerosol or angstrom here
    km20_daily = km20.resample('D', dim='time', how='mean')  # take daily averages
    # km20_daily = longtermanomaly(km20_daily)  # find anomaly

    numevents = len(events)
    eventarr = np.zeros((50, numevents))  # days, num. events

    means = np.zeros(numevents)
    for i in range(numevents):
        print(start_dates[i])
        eventarr[:, i] = km20_daily.sel(time=slice(start_dates[i], end_dates[i]))
        means[i] = np.nanmean(eventarr[:, i])

    print(means)

    grand_mean = np.nanmean(means)
    print(grand_mean)
    eventarr = eventarr - grand_mean

    compm = np.zeros(50)  # average events together for each day
    for i in range(50):
        compm[i] = np.nanmean(eventarr[i, :])

    sns.set(context="talk", style="darkgrid")
    sns.set_palette('hls', 12)
    fig, ax = plt.subplots(figsize=(10, 12))
    plt.plot(np.arange(-14, 36, 1), eventarr)
    plt.plot(np.arange(-14, 36, 1), compm, 'k')
    plt.ylabel("Aerosol Extinction Anomaly")
    plt.xlabel("Day from Event")
    plt.title("Forbush Decreases: 2002-11")
    plt.show()

    """

