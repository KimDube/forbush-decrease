
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
from ForbushDecrease import monte_carlo as mc


# -----------------------------------------------------------------------------
def anomalize(timeseries):
    """
    Centre time series on zero (percent difference from mean). Then take anomaly
    as difference from 35 day running mean.
    :param timeseries: 1d array in steps of 1 day
    :return: timeseries as percent difference from mean, filtered with 35 day
    high-pass.
    """
    # First centre data on zero: fig 2a of Laken and Calogovic 2013
    timeseries = 100 * (timeseries - np.nanmean(timeseries)) / np.nanmean(timeseries)

    # Take anomaly in counts to be difference from 35 day running mean
    #  -> high-pass filter.
    y = pd.Series(timeseries)
    f2 = y.rolling(center=True, window=35).mean()
    u = y - f2
    timeseries = np.array(u)
    return timeseries


# -----------------------------------------------------------------------------
def shift_epoch(event_array, initial_epoch_start):
    """
    Shifts first day in epoch so that the minima will be aligned with day zero.
    (Epoch start should be 2 weeks before minima, hence the -14 shift factor).
    :param event_array:
    :param initial_epoch_start:
    :return: list of new epoch starts as strings.
    """
    new_onset = []
    for j in range(NUM_EVENTS):
        d = np.argmin(event_array[:, j])
        ne = datetime.datetime.strptime(initial_epoch_start[j], "%Y-%m-%d") + datetime.timedelta(int(d) - 14)
        new_onset.append(ne.strftime('%Y-%m-%d'))

    return new_onset


# -----------------------------------------------------------------------------
def find_mean_gcr(epoch_starts, plot=0, shifted=0):
    """
    Epoch analysis of GCR from Moscow neutron monitor.
    Epoch start dates are shifted so maximum decrease is at day zero.
    :param epoch_starts:
    :param plot:
    :param shifted:
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

    counts = anomalize(counts)  # find anomaly

    # convert strings to match start_days
    for j in range(len(dates)):
        dates[j] = dates[j].replace('.', '-')
    dates = np.array(dates)
    epoch_starts = np.array(epoch_starts)

    # search for epoch start_days and store corresponding count values for complete epoch
    gcrarr = np.zeros((EPOCH_LENGTH, NUM_EVENTS))

    # If shifted is 0 then epoch starts need to be shifted so minima is aligned
    #  with day zero (default).
    if shifted == 0:
        new_start_days = []
        for j in range(NUM_EVENTS):
            temp = np.where(dates == epoch_starts[j])
            temp = int(temp[0][0])
            for k in range(EPOCH_LENGTH):
                gcrarr[k, j] = counts[temp+k]
            # shift the event onset times so the minima is aligned with day zero.
            new_start_days = shift_epoch(gcrarr, epoch_starts)
        # Find new epochs
        for j in range(NUM_EVENTS):
            temp = np.where(dates == new_start_days[j])
            temp = int(temp[0][0])
            for k in range(EPOCH_LENGTH):
                gcrarr[k, j] = counts[temp+k]
        # Save so this doesn't always have to be done
        np.save('./shifted_epoch_onsets', new_start_days)

    # If shifted is 1 then the epochs already have the minima aligned with day 0
    elif shifted == 1:
        for j in range(NUM_EVENTS):
            temp = np.where(dates == epoch_starts[j])
            temp = int(temp[0][0])
            for k in range(EPOCH_LENGTH):
                gcrarr[k, j] = counts[temp + k]

    gcr_mean = np.zeros(EPOCH_LENGTH)  # for each day average events together
    gcr_std = np.zeros(EPOCH_LENGTH)
    for j in range(EPOCH_LENGTH):
        gcr_mean[j] = np.nanmean(gcrarr[j, :])
        gcr_std[j] = np.nanstd(gcrarr[j, :])

    gcr_mean_err, gcr_std_err = mc.significance_test(counts, EPOCH_LENGTH, NUM_EVENTS, 10000)

    # Makes a surface plot and line plot for all events. Also the mean plot
    # with standard error
    if plot == 1:
        """
        fig1, ax1 = plt.subplots()
        sns.set_context("talk")
        cmap = ListedColormap(sns.color_palette("RdBu_r", 50))
        fax = ax1.contourf(PLOT_RANGE, np.arange(1, len(epoch_starts) + 1), gcrarr.transpose(),
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
        plt.plot(PLOT_RANGE, gcrarr)
        plt.ylabel("Neutron Count Anomaly")
        plt.xlabel("Day from Event")
        plt.title("Forbush Decreases: 2002-15")
        plt.show()
        """

        sns.set(context="talk", style="darkgrid")
        sns.set_palette('hls', 12)
        plt.plot(PLOT_RANGE, gcr_mean, 'b')
        plt.plot(PLOT_RANGE, gcr_mean + (gcr_std / np.sqrt(NUM_EVENTS)), 'm')
        plt.plot(PLOT_RANGE, gcr_mean - (gcr_std / np.sqrt(NUM_EVENTS)), 'm')

        plt.plot(PLOT_RANGE, gcr_mean_err + 1.96 * gcr_std_err, 'k-.')
        plt.plot(PLOT_RANGE, gcr_mean_err - 1.96 * gcr_std_err, 'k-.')

        plt.ylabel("Mean Neutron Count Anomaly")
        plt.xlabel("Day from Event")
        plt.title("Forbush Decreases: 2002-15")
        plt.show()

    return gcr_mean


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Number of days in each epoch. 14 days (2 weeks) before event to 35 days (5 weeks) after event.
    EPOCH_LENGTH = 50
    PLOT_RANGE = np.arange(-14, 36, 1)
    EVENTS = ['2002-02-04', '2002-03-18', '2002-04-17', '2002-05-23', '2002-08-18', '2003-01-26', '2003-05-29',
              '2003-08-17', '2003-10-30', '2003-11-30', '2004-07-26', '2004-09-13', '2004-11-09', '2005-01-18',
              '2005-05-15', '2005-07-17', '2005-09-11', '2006-07-09', '2006-12-14', '2007-01-29', '2010-08-03',
              '2011-04-11', '2011-06-22', '2011-08-05', '2011-10-30', '2012-01-22', '2012-03-08', '2012-04-04',
              '2012-07-14', '2012-09-04', '2012-10-11', '2012-11-23', '2013-01-16', '2013-03-17', '2013-05-31',
              '2013-07-12', '2014-04-18', '2014-06-17', '2014-09-12', '2014-12-21', '2015-03-17', '2015-05-06',
              '2015-06-22', '2015-09-07', '2015-11-06']
    NUM_EVENTS = len(EVENTS)

    try:  # check if a file containing shifted epoch start dates exists. If so use that.
        start_dates = np.load('shifted_epoch_onsets.npy')
        end_dates = []
        for i in start_dates:
            # look 2 weeks before and 5 weeks after (epoch of 50 days).
            e = datetime.datetime.strptime(i, "%Y-%m-%d") + datetime.timedelta(49)
            e = e.strftime('%Y-%m-%d')
            end_dates.append(e)

        mean_gcr = find_mean_gcr(start_dates, plot=1, shifted=1)

    except:  # if epoch start file doesn't exist use initial IZMIRAN events to create it.
        print("Using non-shifted event onsets")
        start_dates = []
        end_dates = []
        for i in EVENTS:
            # look 2 weeks before and 5 weeks after (effects could take up to a month...)
            s = datetime.datetime.strptime(i, "%Y-%m-%d") + datetime.timedelta(-14)
            e = datetime.datetime.strptime(i, "%Y-%m-%d") + datetime.timedelta(35)
            s = s.strftime('%Y-%m-%d')
            start_dates.append(s)
            e = e.strftime('%Y-%m-%d')
            end_dates.append(e)

        mean_gcr = find_mean_gcr(start_dates, plot=0, shifted=0)

    alt_of_interest = 20.5
    # Data only exists up to 2012 for version 6
    x = osiris_nc.aer_level2_from_nc(nc_folder="/home/kimberlee/OsirisData/Level2/daily/",
                                     version='5.07', start_date='2002-01-01', end_date='2016-12-31')
    # x = omps_2d.aer_level2_from_nc('/home/kimberlee/ValhallaData/OMPS/OMPS_Tomography_v1.0.2/L2')

    # x.to_netcdf(path='/home/kimberlee/Masters/OMPS_all', mode='w')
    # x = xarray.open_dataset('/home/kimberlee/Masters/test_o3658')
    # print(x)
    # exit()

    a = x.where(x.latitude < 20)  # select location
    b = a.where(a.latitude > -20)
    km20 = b.aerosol.sel(altitude=alt_of_interest)  # Choose either aerosol or angstrom here
    km20_daily = km20.resample('D', dim='time', how='mean')  # take daily averages

    # km20_daily = anomalize(km20_daily)  # find normalized anomaly
    km20_daily = 100 * ((km20_daily - km20_daily.mean()) / km20_daily.mean())
    km20_rolling = km20_daily.rolling(time=35, min_periods=5).mean()
    km20_daily = km20_daily - km20_rolling

    eventarr = np.zeros((EPOCH_LENGTH, NUM_EVENTS))  # days, num. events
    for i in range(NUM_EVENTS):
        try:
            eventarr[:, i] = km20_daily.sel(time=slice(start_dates[i], end_dates[i]))
        except:
            eventarr[:, i] = np.nan
            print("No data to load: %s" % start_dates[i])

    compm = np.zeros(EPOCH_LENGTH)  # average events together for each day
    for i in range(EPOCH_LENGTH):
        compm[i] = np.nanmean(eventarr[i, :])

    aer_mean_err, aer_std_err = mc.significance_test(km20_daily, EPOCH_LENGTH, NUM_EVENTS, 10000)

    sns.set(context="talk", style="darkgrid")
    sns.set_palette('hls', 12)
    fig, ax = plt.subplots(figsize=(20, 8))
    # plt.plot(PLOT_RANGE, eventarr)
    plt.plot(PLOT_RANGE, compm)
    plt.plot(PLOT_RANGE, aer_mean_err + 1.96 * aer_std_err, 'k-.')
    plt.plot(PLOT_RANGE, aer_mean_err - 1.96 * aer_std_err, 'k-.')

    plt.ylabel("OSIRIS Aerosol Extinction Anomaly - 5.07")
    plt.xlabel("Day from Event")
    plt.title("2002-15, %i km, tropics" % alt_of_interest)
    plt.savefig('/home/kimberlee/Masters/ForbushDecrease/OSIRIS_line_507_%ikm' % alt_of_interest, dpi=300)
    plt.show()

