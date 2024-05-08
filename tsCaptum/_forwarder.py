import numpy as np
import torch

# documentation for each method!

class _Forwarder():

	# TODO specify possible clf type as aeonClf | SktimeClf | ....

	def __init__(self,predictor, predictor_type):
		# TODO should I make the following two variables private?
		self.predictor = predictor
		if predictor_type == "regressor":
			# TODO can I avoid it?
			self.raw_result_func = self.predictor.predict
		elif predictor_type == "classifier":
			self.raw_result_func = self.predictor.predict_proba
		else:
			raise (" provided model not recoginzed. Please specify whether is a classifier or regressor ")

	def forward(self, X ):

		# convert X to pytorch tensor
		# TODO force  each variable's type
		X_numpy : np.array = X.detach().numpy()
		# use the model forward function
		preds : np.array = self.raw_result_func(X_numpy)
		# return result as torch tensor as expected by captum attribution method
		preds_torch : torch.tensor = torch.tensor(preds)

		return preds_torch