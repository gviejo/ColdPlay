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

path = os.path.join(data_directory,data_directory.split('/')[-1]+'.eeg')

lfp = loadLFP(path, n_channels=32, channel=14, frequency=1250.0, precision='int16')
lfp = downsample(lfp, 1, 5)
lfp = lfp.restrict(wake_ep)



lfp_filt		= nts.Tsd(lfp.index.values, butter_bandpass_filter(lfp, 5, 15, 250, 2))
power	 		= nts.Tsd(lfp_filt.index.values, np.abs(lfp_filt.values))
enveloppe,dummy	= getPeaksandTroughs(power, 5)	
enveloppe 		= enveloppe.as_series()
index 			= (enveloppe > np.percentile(enveloppe, 20)).values*1.0
start_cand 		= np.where((index[1:] - index[0:-1]) == 1)[0]+1
end_cand 		= np.where((index[1:] - index[0:-1]) == -1)[0]
if end_cand[0] < start_cand[0]:	end_cand = end_cand[1:]
if end_cand[-1] < start_cand[-1]: start_cand = start_cand[0:-1]
tmp 			= np.where(end_cand != start_cand)
start_cand 		= enveloppe.index.values[start_cand[tmp]]
end_cand	 	= enveloppe.index.values[end_cand[tmp]]
good_ep			= nts.IntervalSet(start_cand, end_cand)
good_ep			= good_ep.drop_short_intervals(300000)

theta_wake_ep 	= wake_ep.intersect(good_ep).merge_close_intervals(30000).drop_short_intervals(1000000)


plot(lfp.restrict(wake_ep))
plot(lfp.restrict(theta_wake_ep))
