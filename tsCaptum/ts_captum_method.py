import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from captum.attr._utils.attribution import PerturbationAttribution
from captum.attr import (FeatureAblation as _FeatureAblationCaptum,
                         FeaturePermutation as _FeaturePermutationCaptum)

from ._forwarder import _Forwarder
from ._utils import  _check_convert_data_format


# TODO should I move this class in some .utils.py along with chunking code? NOPE
#  (btw modify s.t. the last segment isn't the short one) etc. ? Probably not
# TODO documentation for each method in each class

class _TSCaptum_Method():

	# TODO specify clf possible types as aeon | sktime | ...
	def __init__(self, explainer, predictor, predictor_type: str = None):

		# check argument values
		if not issubclass(explainer, PerturbationAttribution):
			raise (
				" provided explainer has to be an instance of 'captum.attr._utils.attribution.PerturbationAttribution' ")

		if not predictor_type in ["classifier", "regressor", None]:
			raise (
				" clf_type argument has to be either  'classifier' or 'regressor' ")

		# check whether is a classifier or a regressor      # TODO is it okay to do it in this way?
		self.predictor_type = predictor_type
		if self.predictor_type is None:
			self.predictor_type = "classifier"
			try:
				predictor.predict_proba
			except AttributeError:
				self.predictor_type = "regressor"

		# set also forward function and explainer to be used
		self._Forwarder = _Forwarder(predictor, self.predictor_type)
		self._explainer = explainer(self._Forwarder.forward)

	# TODO check how y is handled in captum attribute
	def explain(self, samples, labels=None, batch_size=8):

		# TODO batch size, default 8 or 16?
		# TODO add chunking

		# TODO to be moved in another function?
		if self.predictor_type == "classifier":
			# transform to numeric labels
			le = LabelEncoder()
			labels_idx = torch.tensor(le.fit_transform(labels))

		elif self.predictor_type == "regressor":
			labels_idx = labels

		else:
			raise (
				" provided regressor type not recognized. Please specify whether is a classifier or regressor ")

		loader = _check_convert_data_format(samples,labels_idx,batch_size)


		# TODO add tqdm?
		explanations = []
		with torch.no_grad():
			# TODO change in enumerate?
			for X,y in loader:
				if self.predictor_type == "classifier":
					current_exp = self._explainer.attribute(X, target=y)
				else:
					current_exp = self._explainer.attribute(X)
				explanations.append( current_exp.detach().numpy() )

		return np.concatenate(explanations)


class Feature_Ablation(_TSCaptum_Method):
	def __init__(self, clf, clf_type=None):
		super().__init__(_FeatureAblationCaptum, clf, clf_type)


class Feature_Permutation(_TSCaptum_Method):
	def __init__(self, clf, clf_type=None):
		super().__init__(_FeaturePermutationCaptum, clf, clf_type)
