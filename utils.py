import torch
import numpy as np

# function to adapt sklearn/aeon classifiers to captum
def forward_classification(X_test : torch.Tensor, model):
	# convert X to pytorch tensor
	X_test_numpy = X_test.detach().numpy()
	# compute probability
	predictions = model.predict_proba(X_test_numpy)
	# return result as torch tensor as expected by captum attribution method
	return torch.tensor(predictions)

def forward_regression(X_test : torch.Tensor, model):
	# convert X to pytorch tensor
	X_test_numpy = X_test.detach().numpy()
	# use the model forward function
	predictions = model.predict(X_test_numpy)
	# return result as torch tensor as expected by captum attribution method
	return torch.tensor(predictions)


def get_groups(n_chunks,n_channels,series_length):
	"""
	function returning how to group time points into time Series accordingly to the given arguments
	To be noted that it operates channel-wise i.e. each channel is divided into "n_chunks" chunks
	:param n_chunks:        number of chunks to be used
	:param n_channels:      number of channel of each instance in the dataset
	:param series_length:   length of each channel of each instance in the dataset
	:return:                a numpy array representing how to group the time points
	"""
	groups = np.array([[i + j * n_chunks for i in range(n_chunks)] for j in range(n_channels)])
	groups = np.expand_dims(np.repeat(groups, np.ceil(series_length / n_chunks).astype(int), axis=1), 0)[:, :, :series_length]
	return torch.tensor( groups )
