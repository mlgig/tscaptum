import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from scipy.interpolate import interp1d


def plot_saliency_map_multi(sample, attribution, channel_names = [], title = 'Saliency map', colorbar = True):

	if len(sample.shape) == 1: # if the univariate input is a 1D array
		sample = np.expand_dims(sample, axis=0)
		attribution = np.expand_dims(attribution, axis=0)

	n_channels = sample.shape[0]

	x = np.array([ii for ii in range(sample.shape[-1])])

	cap = max(abs(attribution.min()), abs(attribution.max()))
	cvals = [-cap, 0, cap]
	# if saliency.min() < 0:
	#     cvals  = [saliency.min(), 0, saliency.max()]
	# else:
	#     cvals  = [0,0, saliency.max()]
	colors = ["blue","gray","red"]
	norm=plt.Normalize(min(cvals),max(cvals))
	tuples = list(zip(map(norm,cvals), colors))
	cmap = LinearSegmentedColormap.from_list("", tuples)


	fig, axs = plt.subplots(n_channels, 1, sharex=True, figsize=(8, 1*n_channels),constrained_layout=True)

	for p in range(sample.shape[0]):
		y = sample[p,:]
		sy = attribution[p,:]
		points = np.array([x, y]).T.reshape(-1, 1, 2)
		segments = np.concatenate([points[:-1], points[1:]], axis=1)

		lc = LineCollection(segments, cmap=cmap, norm=norm)
		lc.set_array(sy)
		lc.set_linewidth(2)


		current_ax = axs if sample.shape[0] == 1 else axs[p]

		line = current_ax.add_collection(lc)
		current_ax.set_xlim(x.min(), x.max())
		current_ax.set_ylim(y.min() - 1, y.max()+1)
		if len(channel_names) >= n_channels:
			current_ax.set_ylabel(channel_names[p])



	if colorbar:
		fig.colorbar(line, ax=axs, aspect= 50)

	fig.align_ylabels(axs)



	plt.show()

def plot_saliency_map_uni(sample, attribution, title = 'Saliency map'):

	def transform(X):
		ma,mi = np.max(X), np.min(X)
		X = (X - mi)/(ma-mi)
		return X*100

	weight = transform(abs(attribution))
	ts = np.squeeze(sample)

	max_length1, max_length2 = ts.shape[0],10000 #
	x1 = np.linspace(0,max_length1,num = max_length1)
	x2 = np.linspace(0,max_length1,num = max_length2)
	y1 = ts

	f = interp1d(x1, y1) # interpolate time series
	fcas = interp1d(x1, weight) # interpolate weight color
	weight = fcas(x2) # convert vector of original weight vector to new weight vector

	plt.figure(figsize=(6, 2))
	plt.scatter(x2,f(x2), c = weight, cmap = 'jet', marker='.', s= 1,vmin=0,vmax = 100)
	plt.title(title)
	plt.show()