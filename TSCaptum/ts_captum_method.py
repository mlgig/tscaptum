import numpy as np
import torch
from captum.attr._utils.attribution import PerturbationAttribution
from captum.attr import (FeatureAblation as _FeatureAblationCaptum,
                         FeaturePermutation as _FeaturePermutationCaptum)

from ._forward_regression import _Forward_Regression

# should I move this class in some .utils.py along with chunking code etc. ?
class _TSCaptum_Method():
	# TODO specify clf possible types as aeon | sktime | ...
	def __init__(self, explainer , clf):
		assert  issubclass(explainer,PerturbationAttribution)
		# TODO check whether is possible to determine clf is a classifier or a regressor e.g. check predict_proba or check aron/sktime rgister etc,
		self._Forward_class = _Forward_Regression(clf)
		self.explainer = explainer(self._Forward_class.forward)

	# TODO check how y is handled in captum attribute
	def explain(self,X: np.array  ,y=None):
		# TODO batch size, default 8 or 16?
		# TODO add chunking

		# handle both numpy and torch arrays?
		X = torch.tensor(X)
		explanation = self.explainer.attribute(X)
		return explanation.detach().numpy()

class Feature_Ablation(_TSCaptum_Method):
	def __init__(self,clf):
		super().__init__(_FeatureAblationCaptum,clf)

class Feature_Permutation(_TSCaptum_Method):
	def __init__(self,clf):
		super().__init__(_FeaturePermutationCaptum,clf)