import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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