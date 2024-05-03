import numpy as np
import torch

class _Forward_Regression():

	# TODO specify possible clf type as aeonClf | SktimeClf | ....
	# TODO is there a way to tell weather regressor or classifier? using aeon.registry.all_estimators or by checking whether there is
	#  a predict_proba or not?

	def __init__(self, clf ):
		# TODO should I make the following two variables private?
		self.clf = clf
		# TODO can I avoid it?
		self.raw_result_func = self.clf.predict

	def forward(self, X ):

		# convert X to pytorch tensor
		# TODO force  each variable's type
		X_numpy : np.array = X.detach().numpy()
		# use the model forward function
		preds : np.array = self.raw_result_func(X_numpy)
		# return result as torch tensor as expected by captum attribution method
		preds_torch : torch.tensor = torch.tensor(preds)

		return preds_torch