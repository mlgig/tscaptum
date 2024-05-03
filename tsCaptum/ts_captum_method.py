import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from captum.attr._utils.attribution import PerturbationAttribution
from captum.attr import (FeatureAblation as _FeatureAblationCaptum,
                         FeaturePermutation as _FeaturePermutationCaptum)

from ._forward import _Forward


# TODO should I move this class in some .utils.py along with chunking code
#  (btw modify s.t. the last segment isn't the short one) etc. ? Probably not
# TODO documentation for each method in each class

class _TSCaptum_Method():

	# TODO specify clf possible types as aeon | sktime | ...
	def __init__(self, explainer, clf, clf_type: str = None):

		# check argument values
		if not issubclass(explainer, PerturbationAttribution):
			raise (
				" provided explainer has to be an instance of 'captum.attr._utils.attribution.PerturbationAttribution' ")

		if not clf_type in ["classifier", "regressor", None]:
			raise (
				" clf_type argument has to be either  'classifier' or 'regressor' ")

		# check whether is a classifier or a regressor      # TODO is it okay to do it in this way?
		self.clf_type = clf_type
		if self.clf_type is None:
			self.clf_type = "classifier"
			try:
				clf.predict_proba
			except AttributeError:
				self.clf_type = "regressor"

		# set also forward function and explainer to be used
		self._Forward_class = _Forward(clf, self.clf_type)
		self._explainer = explainer(self._Forward_class.forward)

	# TODO check how y is handled in captum attribute
	def explain(self, X: np.array, labels=None):

		# TODO handle both numpy and torch arrays?
		# TODO batch size, default 8 or 16?
		# TODO add chunking

		X = torch.tensor(X)
		if self.clf_type == "regressor":
			explanation = self._explainer.attribute(X)

		elif self.clf_type == "classifier":

			# transform to numeric labels
			le = LabelEncoder()
			labels_idx = torch.tensor(le.fit_transform(labels))

			# then explain
			explanation = self._explainer.attribute(X, target=labels_idx)

		else:
			raise (
				" provided model not recoginzed. Please specify whether is a classifier or regressor ")

		return explanation.detach().numpy()


class Feature_Ablation(_TSCaptum_Method):
	def __init__(self, clf, clf_type=None):
		super().__init__(_FeatureAblationCaptum, clf, clf_type)


class Feature_Permutation(_TSCaptum_Method):
	def __init__(self, clf, clf_type=None):
		super().__init__(_FeaturePermutationCaptum, clf, clf_type)
