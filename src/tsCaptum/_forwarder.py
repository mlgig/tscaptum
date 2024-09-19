import numpy as np
import torch
from ._ridge_classifier import proba4ridge

# documentation for each method!

class _Forwarder:
	"""
	class automatically detecting the right forward function to use with the captum explainers
	"""

	# TODO specify possible clf type as aeonClf | SktimeClf | ....

	def __init__(self, predictor, predictor_type):
		"""
		init function for this class

		:param predictor: the predictor to be explained
		:
		param predictor_type: (optional) the predictor variable's type i.e. regressor or classifier
		"""
		self.predictor = predictor

		if predictor_type == "regressor":
			self.raw_result_func = self.predictor.predict
			self.features_transform = None

		elif predictor_type == "ridge_classifier":
			self.raw_result_func, self.features_transform = proba4ridge(self.predictor)

		elif predictor_type == "classifier":
			self.raw_result_func = self.predictor.predict_proba
			self.features_transform = None

		else:
			raise " provided model not recognized. Please specify whether is a classifier or regressor "

	def forward(self, X):
		"""
		function adapting scikit-learn-like predictor to captum explainers

		:param X: instances to be explained

		:return: predictor's output as a torch tensor
		"""

		# convert X to pytorch tensor
		X_numpy: np.array = X.detach().numpy()

		# transform the feature if this is needed e.g. Rocket
		if self.features_transform!=None:
			X_numpy = self.features_transform(X_numpy)

		# use the model forward function
		preds: np.array = self.raw_result_func(X_numpy)
		# return result as torch tensor as expected by captum attribution method
		preds_torch: torch.tensor = torch.tensor(preds)

		return preds_torch