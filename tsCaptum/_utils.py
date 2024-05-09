import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

from ._tsCaptum_loader import _tsCaptum_loader


def equal_length_segmentation(n_chunks: int, n_channels: int, series_length: int):
	"""
	function returning how to group time points into time Series accordingly to the given arguments
	To be noted that it operates channel-wise i.e. each channel is divided into "n_chunks" chunks

	:param n_chunks:        number of chunks to be used
	:param n_channels:      number of channel of each instance in the dataset
	:param series_length:   length of each channel of each instance in the dataset
	:return:                a numpy array representing how to group the time points
	"""
	quotient, reminder = np.floor(series_length / n_chunks).astype(int) , series_length % n_chunks

	first_group = np.array([[i + j * n_chunks for i in range(reminder)] for j in range(n_channels)])
	first_group = np.expand_dims(np.repeat(first_group,(quotient+1), axis=1), 0)

	second_group = np.array([[i + j * n_chunks for i in range( reminder,n_chunks )] for j in range(n_channels)])
	second_group = np.expand_dims(np.repeat(second_group,quotient, axis=1), 0)
	final_group = np.concatenate((first_group,second_group),axis=-1)
	return torch.tensor(final_group)


def _check_labels(labels, predictor_type):
	if predictor_type == "classifier":
		# transform to numeric labels
		le = LabelEncoder()
		labels_idx = torch.tensor(le.fit_transform(labels)).type(torch.int64)

	elif predictor_type == "regressor":
		assert labels is None
		le = None
		labels_idx = None

	else:
		raise (
			" provided predictor type not recognized. Please specify whether is a classifier or regressor ")

	return le, labels_idx


def _check_convert_data_format(X, labels, batch_size):
	# TODO check if X and labels have the same dimension

	if isinstance(X, np.ndarray):
		X = torch.tensor(X).type(torch.float)
	elif isinstance(X, torch.Tensor):
		X = X.type(torch.float)
	else:
		raise TypeError(
			" Data format has to be either numpy array or torch tensor ")

	if labels is None:
		labels = torch.ones(X.shape[0]) * -1
	loader = DataLoader(_tsCaptum_loader(X, labels), shuffle=False, batch_size=batch_size)

	return loader
