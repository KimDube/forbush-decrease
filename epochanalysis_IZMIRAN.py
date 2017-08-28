
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

# events are those from IZMIRAN db with MagM >= 2.0
event_dates = ['2007-01-29', '2007-02-12', '2007-03-22', '2007-09-20', '2007-10-18', '2007-10-25', '2007-11-19',
               '2007-12-17', '2008-01-04', '2008-01-31', '2008-03-08', '2008-03-26', '2008-05-20', '2008-05-28',
               '2008-06-14', '2008-06-24', '2008-08-08', '2008-09-03', '2008-11-24', '2008-12-16', '2008-12-30',
               '2009-02-14', '2009-07-22', '2009-08-05', '2009-10-22', '2010-01-19', '2010-02-11', '2010-04-11',
               '2010-05-02', '2010-05-28', '2010-07-14', '2010-08-03', '2010-08-23', '2010-10-04', '2010-10-11',
               '2010-10-22', '2011-02-04', '2011-03-01', '2011-03-10', '2011-04-05', '2011-04-11', '2011-04-29',
               '2011-05-27', '2011-06-04', '2011-06-09', '2011-06-17', '2011-06-22', '2011-07-11', '2011-07-18',
               '2011-07-30', '2011-08-05', '2011-09-26', '2011-10-05', '2011-10-08', '2011-10-24', '2011-11-01',
               '2011-11-28']

start_dates = []
end_dates = []
fake_end_dates = []
for i in event_dates:
    s = datetime.datetime.strptime(i, "%Y-%m-%d") + datetime.timedelta(-10)
    e = datetime.datetime.strptime(i, "%Y-%m-%d") + datetime.timedelta(11)
    f = datetime.datetime.strptime(i, "%Y-%m-%d") + datetime.timedelta(41)
    s = s.strftime('%Y-%m-%d')
    start_dates.append(s)
    e = e.strftime('%Y-%m-%d')
    end_dates.append(e)
    f = f.strftime('%Y-%m-%d')
    fake_end_dates.append(f)

print(start_dates)
print(fake_end_dates)

numevents = len(event_dates)
eventarr = np.zeros((22, numevents))  # days, num. events

altofinterest = 30.5  # altitude to look at

for i in range(len(start_dates)):
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

    # background = np.nanmean(km20_daily[0:10])
    # km20_daily = 100 * (km20_daily - background) / background  # Variation from background

    eventarr[:, i] = km20_daily

compm = np.zeros(22)  # average events together for each day
for i in range(22):
    compm[i] = (np.nanmean(eventarr[i, :]))


fig, ax = plt.subplots()
sns.set_context("talk")
cmap = ListedColormap(sns.color_palette("PuOr_r", 50))
fax = ax.contourf(np.arange(-10, 12, 1), np.arange(1, numevents+1), eventarr.transpose(),
                  cmap=cmap, extend='both')
plt.ylabel("Event Number")
plt.xlabel("Day from Event")
plt.title("Forbush Decreases: 2002-05")
plt.gca().invert_yaxis()
cb = plt.colorbar(fax, orientation='horizontal', fraction=0.2, aspect=30)
cb.set_label("Aerosol Extinction")
plt.show()

times = np.arange(-10, 12, 1)
sns.set(context="talk", style="ticks", palette='cubehelix')
fig2, ax1 = plt.subplots(figsize=(10, 8))
ax1.plot(times, compm)
ax1.set_title("tropics , %1.1f km" % altofinterest)
ax1.set_xlabel("Day from event")
ax1.set_ylabel("Extinction Variation")
plt.show()

