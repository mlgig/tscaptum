import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

from ._tsCaptum_loader import _tsCaptum_loader


# TODO check to aeon, sktime, torch and captum how function in utils are called (with or without leading _) ?

def _equal_length_segmentation(n_chunks: int, n_channels: int, series_length: int):
	r"""
	function returning how to group time points into time Series accordingly to the given arguments
	To be noted that it operates channel-wise i.e. each channel is divided into 'n_chunks' chunks

	:param n_chunks:        number of chunks to be used

	:param n_channels:      number of channel of each instance in the dataset

	:param series_length:   length of each channel of each instance in the dataset

	:return:                a torch tensor representing how to group time points
	"""
	quotient, reminder = np.floor(series_length / n_chunks).astype(int), series_length % n_chunks

	first_group = np.array([[i + j * n_chunks for i in range(reminder)] for j in range(n_channels)])
	first_group = np.expand_dims(np.repeat(first_group, (quotient + 1), axis=1), 0)

	second_group = np.array([[i + j * n_chunks for i in range(reminder, n_chunks)] for j in range(n_channels)])
	second_group = np.expand_dims(np.repeat(second_group, quotient, axis=1), 0)

	final_group = np.concatenate((first_group, second_group), axis=-1)
	return torch.tensor(final_group).to(torch.int64)


def _normalise_result(X):
	"""
	function to normalize obtained saliency map

	:param X: the saliency map to be normalized

	:return: normalized version of X
	"""

	assert len(X.shape) == 3
	results = []
	for x in X:
		scaling_factor = 1 / max(np.abs(x.max()), np.abs(x.min()))
		results.append(scaling_factor * x)
	return results


def _check_labels(labels, predictor_type):
	r"""
	function checking the label argument provided to explain method and converting them into integer representation as
	required by captum

	:param labels:          provided labels

	:param predictor_type:  predictor's type i.e. classifier or regressor

	:return:                label encoder and relative integer indices
	"""
	if predictor_type == "classifier":
		# transform to numeric labels
		le = LabelEncoder()
		labels_idx = torch.tensor(le.fit_transform(labels)).type(torch.int64)

	elif predictor_type == "regressor":
		if labels is not None:
			raise ValueError(
				"specified labels when predictor type is regressor"
			)
		le = None
		labels_idx = None

	else:
		raise (
			" provided predictor type not recognized. Please specify whether is a classifier or regressor "
		)

	return le, labels_idx


def _check_convert_data_format(X, labels, batch_size):
	r"""
	function checking and converting provided samples and labels to explain method

	:param X:           sample to explain. Can be provided as numpy array or as torch tensor

	:param labels:      labels provided to explain method

	:param batch_size:  batch size provided to explain method

	:return:            data loader to be used in the explain method
	"""

	if X is None and labels is not None:
		if X.shape[0] != labels.shape[0]:
			# if both X and labels are provided having no matching dimensions
			raise ValueError(
				"provided samples and labels have different dimensions"
			)

	if isinstance(X, np.ndarray):
		X = torch.tensor(X).type(torch.float)
	elif isinstance(X, torch.Tensor):
		X = X.type(torch.float)
	else:
		raise TypeError(
			" Data format has to be either numpy array or torch tensor "
		)

	if labels is None:
		labels = torch.ones(X.shape[0]) * -1
	loader = DataLoader(_tsCaptum_loader(X, labels), shuffle=False, batch_size=batch_size)

	return loader
