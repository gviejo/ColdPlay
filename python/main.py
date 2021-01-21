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



figure()
subplot(121)
plot(position['z'].restrict(wake_ep.loc[[0]]), position['x'].restrict(wake_ep.loc[[0]]))

subplot(122)
plot(position['z'].restrict(wake_ep.loc[[1]]), position['x'].restrict(wake_ep.loc[[1]]))


spatial_curves, extent				= computePlaceFields(spikes, position[['x', 'z']], wake_ep.loc[[1]], 40)



figure()
count = 1
for j in np.unique(shank):
	neurons = np.where(shank == j)[0]
	for k,i in enumerate(neurons):
		subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count)
		tmp = spatial_curves[i]
		imshow(gaussian_filter(tmp, 2), interpolation = 'bilinear')
		colorbar()
		count += 1
		title(str(j)+' '+str(i))
		xticks([])


pc = [0, 3, 5, 7, 8, 10, 17, 18, 19, 21]


sys.exit()




