
"""
Superposed epoch analysis of Forbush decreases events from IZMIRAN database
Author: Kimberlee Dube
August/ September 2017
"""

from arglevel3.data_sources import osiris_nc
from arglevel3.data_sources import omps_2d
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
    altofinterest = 20.5
    # Data only exists up to 2012 for version 6
    x = osiris_nc.aer_level2_from_nc(nc_folder="/home/kimberlee/OsirisData/Level2/daily_version600/",
                                     version='6.00', start_date='2002-01-01', end_date='2013-12-31')
    # x = omps_2d.aer_level2_from_nc('smb://datastore/valhalla/data/OMPS/OMPS_Tomography_v1.0.2/L2/')
    a = x.where(x.latitude < 20)  # select location
    b = a.where(a.latitude > -20)
    km20 = b.aerosol.sel(altitude=altofinterest)  # Choose either aerosol or angstrom here
    km20_daily = km20.resample('D', dim='time', how='mean')  # take daily averages
    km20_daily = longtermanomaly(km20_daily)  # find anomaly

    events = ['2002-03-18', '2002-04-17', '2002-05-23', '2002-08-18', '2002-09-07', '2002-09-30', '2002-11-17',
              '2003-05-29', '2003-08-17', '2003-10-21', '2003-10-24', '2003-10-30', '2003-11-20', '2004-01-06',
              '2004-01-22', '2004-07-26', '2004-09-13', '2004-11-07', '2004-11-09', '2004-12-05', '2005-01-18',
              '2005-01-21', '2005-05-08', '2005-05-15', '2005-05-29', '2005-08-24', '2005-09-11', '2006-12-14',
              '2010-08-03', '2011-02-18', '2011-08-05', '2011-09-29', '2011-10-24']

    start_dates = []
    end_dates = []
    for i in events:
        # look 2 weeks before and 5 weeks after (effects could take up to a month...)
        s = datetime.datetime.strptime(i, "%Y-%m-%d") + datetime.timedelta(-14)
        e = datetime.datetime.strptime(i, "%Y-%m-%d") + datetime.timedelta(35)
        s = s.strftime('%Y-%m-%d')
        start_dates.append(s)
        e = e.strftime('%Y-%m-%d')
        end_dates.append(e)

    numevents = len(events)
    eventarr = np.zeros((50, numevents))  # days, num. events

    for i in range(numevents):
        print(start_dates[i])
        e = km20_daily.sel(time=slice(start_dates[i], end_dates[i]))
        eventarr[:, i] = e / np.max(abs(e))

    compm = np.zeros(50)  # average events together for each day
    for i in range(50):
        compm[i] = np.nanmean(eventarr[i, :])

    sns.set(context="poster", style="darkgrid")
    sns.set_palette('hls', 12)
    fig, ax = plt.subplots(figsize=(10, 12))
    plt.plot(np.arange(-14, 36, 1), eventarr)
    plt.plot(np.arange(-14, 36, 1), compm, 'k')
    plt.ylabel("Aerosol Extinction Anomaly")
    plt.xlabel("Day from Event")
    plt.title("Forbush Decreases: 2002-11")
    plt.show()

