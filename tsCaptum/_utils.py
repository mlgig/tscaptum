import numpy as np
import torch
from torch.utils.data import DataLoader
from ._tsCaptum_loader import _tsCaptum_loader

def _check_convert_data_format(X, labels,batch_size):

	# TODO check if X and labels have the same dimension

	if type(X) == np.ndarray:
		X = torch.tensor(X).type(torch.float)
	elif type(X) == torch.Tensor:
		X =  (X).type(torch.float)
	else:
		raise TypeError (
			" Data format has to be either numpy array or torch tensor ")

	if labels is not None:
		# TODO check this warning:
		#  UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
		#   labels = torch.tensor(labels).type(torch.int64)
		labels = torch.tensor(labels).type(torch.int64)
	else:
		labels = torch.ones(X.shape[0])*-1
	loader = DataLoader(_tsCaptum_loader(X,labels), shuffle=False, batch_size=batch_size)

	return loader