import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
from scipy.ndimage import gaussian_filter

# data_directory = '/mnt/DataGuillaume/CA1/A5602/A5602-201003'
data_directory = '/mnt/LocalHDD/CA1/A5602/A5602-201003'


episodes = ['sleep', 'wake', 'sleep', 'wake', 'sleep']
events = ['1', '3']





spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
position 							= loadPosition(data_directory, events, episodes)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					
acceleration						= loadAuxiliary(data_directory, n_probe = 2)
acceleration 						= acceleration[[0,1,2]]
acceleration.columns 				= pd.Index(np.arange(3))
sleep_ep 							= refineSleepFromAccel(acceleration, sleep_ep)
rip_ep, rip_tsd						= loadRipples(data_directory)
spatial_curves, extent				= computePlaceFields(spikes, position[['x', 'z']], wake_ep.loc[[1]], 40)

ahv = computeAngularHeadVelocity(position['ry'], wake_ep)


pc = [0, 3, 5, 7, 8, 10, 17, 18, 19, 21]

zx = position[['z', 'x']].as_dataframe()
index = (zx['z'] > 0) & (zx['z'] < 0.22) & (zx['x'] > -0.23) & (zx['x'] < 0.0)
start = np.where(np.diff(index*1) == 1)[0]
end = np.where(np.diff(index*1) == -1)[0]

ladder_ep = nts.IntervalSet(start = zx.index.values[start], end = zx.index.values[end])
ladder1_ep = ladder_ep.intersect(wake_ep.loc[[0]])
ladder2_ep = ladder_ep.intersect(wake_ep.loc[[1]])

# isolating head scan
sahv = np.abs(ahv.restrict(ladder_ep).as_series())
index = (sahv>3)*1
start = np.where(np.diff(index) == 1)[0]
end = np.where(np.diff(index) == -1)[0]
headscan_ep = nts.IntervalSet(start = sahv.index.values[start], end = sahv.index.values[end])
headscan1_ep = headscan_ep.intersect(ladder1_ep.loc[[0]])
headscan2_ep = headscan_ep.intersect(ladder2_ep.loc[[1]])


# ripples on the ladder
rippos1_tsd = position.index.values[np.argmin(np.abs(np.vstack(rip_tsd.restrict(wake_ep.loc[[0]]).index.values) - position.restrict(wake_ep.loc[[0]]).index.values), 1)]
rippos2_tsd = position.index.values[np.argmin(np.abs(np.vstack(rip_tsd.restrict(wake_ep.loc[[1]]).index.values) - position.restrict(wake_ep.loc[[1]]).index.values), 1)]

rippos1_tsd = position.loc[rippos1_tsd].as_dataframe()
rippos2_tsd = position.loc[rippos2_tsd].as_dataframe()

index = (rippos1_tsd['z'] > 0) & (rippos1_tsd['z'] < 0.22) & (rippos1_tsd['x'] > -0.23) & (rippos1_tsd['x'] < 0.0)
riplad1_tsd = rippos1_tsd[index]
index = (rippos2_tsd['z'] > 0) & (rippos2_tsd['z'] < 0.22) & (rippos2_tsd['x'] > -0.23) & (rippos2_tsd['x'] < 0.0)
riplad2_tsd = rippos2_tsd[index]


# load lfp of ripples on the ladder
lfp = loadLFP(os.path.join(data_directory,data_directory.split('/')[-1]+'.eeg'), n_channels=32, channel=14, frequency=1250.0, precision='int16')

riplad1_ep = nts.IntervalSet(start = riplad1_tsd.index.values-1e5, end = riplad1_tsd.index.values+1e5)
riplad2_ep = nts.IntervalSet(start = riplad2_tsd.index.values-1e5, end = riplad2_tsd.index.values+1e5)
riplad1_ep = riplad1_ep.merge_close_intervals(0)
riplad2_ep = riplad2_ep.merge_close_intervals(0)


figure()
# ax = subplot(311)
# plot(lfp)
# plot(lfp.restrict(ladder1_ep))
# plot(lfp.restrict(ladder2_ep))
subplot(211, sharex = ax)
plot(ahv)
plot(ahv.restrict(ladder1_ep))
plot(ahv.restrict(ladder2_ep))
subplot(212, sharex = ax)
for i, n in enumerate(pc):
	plot(spikes[n].restrict(wake_ep).fillna(i), '|')



figure()
subplot(121)
plot(position['z'].restrict(wake_ep.loc[[0]]), position['x'].restrict(wake_ep.loc[[0]]))
plot(rippos1_tsd['z'], rippos1_tsd['x'], 'o')
plot(riplad1_tsd['z'], riplad1_tsd['x'], 'o')
axvline(0.0)
axvline(0.22)
axhline(-0.23)
axhline(0.0)
subplot(122)
xlabel('z')
ylabel('x')
plot(position['z'].restrict(wake_ep.loc[[1]]), position['x'].restrict(wake_ep.loc[[1]]))
plot(rippos2_tsd['z'], rippos2_tsd['x'], 'o')
plot(riplad2_tsd['z'], riplad2_tsd['x'], 'o')
axvline(0.0)
axvline(0.22)
axhline(-0.23)
axhline(0.0)
xlabel('z')
ylabel('x')


figure()
ax = subplot(311)
plot(position['x'].restrict(ladder1_ep))
plot(position['x'].restrict(ladder2_ep))
subplot(312, sharex = ax)
plot(position['z'].restrict(ladder1_ep))
plot(position['z'].restrict(ladder2_ep))
subplot(313, sharex = ax)
plot(ahv)
plot(ahv.restrict(ladder1_ep))
plot(ahv.restrict(ladder2_ep))

