import numpy as np
from tqdm import tqdm
import torch
import warnings

from captum.attr._utils.attribution import PerturbationAttribution
from captum.attr import (FeatureAblation as _FeatureAblationCaptum,
                         FeaturePermutation as _FeaturePermutationCaptum,
                         KernelShap as _KernelShapCaptum,
                         Lime as _LimeCaptum,
                         ShapleyValueSampling as _ShapleyValueSamplingCaptum)

from ._forwarder import _Forwarder
from ._utils import _check_convert_data_format, _check_labels
from ._utils import *


# TODO documentation for each method in each class

class _TSCaptum_Method():

	def __init__(self, explainer, predictor, predictor_type: str = None):

		# check argument values
		if not issubclass(explainer, PerturbationAttribution):
			raise (
				" provided explainer has to be an instance of 'captum.attr._utils.attribution.PerturbationAttribution' ")

		if not predictor_type in ["classifier", "regressor", None]:
			raise (
				" clf_type argument has to be either  'classifier' or 'regressor' ")

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

	def explain(self, samples, labels=None, batch_size=8, n_segments=10, normalise=False, baseline=0):
		"""
		main method to get a saliency map by the selected explainer
		:param samples:
		:param labels:
		:param batch_size:
		:param n_segments:
		:param normalise:
		:param baseline: the baseline which will substitute ts'values when ablated. It can be either a scalar or a TS
		(both as numpy array or torch.tensor)
		:return:
		"""

		n_2explain, n_channels, series_length = samples.shape
		le, labels_idx = _check_labels(labels, self.predictor_type)
		loader = _check_convert_data_format(samples, labels_idx, batch_size)

		explanations = []
		with tqdm(total=n_2explain) as pbar:
			with torch.no_grad():
				for n, (X, y) in enumerate(loader):

					kwargs = self._define_kwargs(baseline, n_channels, n_segments, series_length, y)
					current_exp = self._explainer.attribute(X, **kwargs)
					explanations.append( normalise_result(current_exp.detach().numpy()) ) if normalise else explanations.append( current_exp.detach().numpy() )
					pbar.update(batch_size)
		pbar.close()
		explanations = np.concatenate(explanations)
		return explanations


	def _define_kwargs(self, baseline, n_channels, n_segments, series_length, y):
		kwargs = {}

		if not isinstance(self, Feature_Permutation):
			if np.isscalar(baseline):
				kwargs['baselines'] = baseline
			elif type(baseline)==np.ndarray:
				kwargs['baselines'] = torch.tensor(baseline)
			elif type(baseline)==torch.Tensor:
				kwargs['baselines'] = baseline

		if self.predictor_type == "classifier":
			kwargs['target'] = y

		if n_segments != -1:
			groups = equal_length_segmentation(n_segments, n_channels, series_length)
			kwargs['feature_mask'] = groups

		return kwargs


class Feature_Ablation(_TSCaptum_Method):

	def __init__(self, clf, clf_type=None):
		super().__init__(_FeatureAblationCaptum, clf, clf_type)



class Feature_Permutation(_TSCaptum_Method):

	def __init__(self, clf, clf_type=None):
		super().__init__(_FeaturePermutationCaptum, clf, clf_type)

	def explain(self, samples, **kwargs):

		if 'batch_size' in  kwargs and kwargs['batch_size'] != 1:
			warnings.warn(
				"batch_size set to 2 as Feature Permutation require more than 1 sample to work"
			)
			kwargs['batch_size']=2

		if 'baselines' in kwargs:
			warnings.warn(
				"specified baseline will be ignored as Feature Permutation algorithm has its own baseline"
			)
		return super().explain(samples, **kwargs)


class Kernel_Shap(_TSCaptum_Method):

	def __init__(self, clf, clf_type=None):
		super().__init__(_KernelShapCaptum, clf, clf_type)

	def explain(self, samples, **kwargs):
		if 'batch_size' in  kwargs and kwargs['batch_size'] != 1:
			warnings.warn(
				"batch_size set to 1 as suggested by Captum for Lime and KernelSHAP"
			)
		kwargs['batch_size'] = 1
		return super().explain(samples, **kwargs)


class LIME(_TSCaptum_Method):

	def __init__(self, clf, clf_type=None):
		super().__init__(_LimeCaptum, clf, clf_type)

	def explain(self, samples, **kwargs):

		if 'batch_size' in  kwargs and kwargs['batch_size'] != 1:
			warnings.warn(
				"batch_size set to 1 as suggested by Captum for Lime and KernelSHAP"
			)
		kwargs['batch_size'] = 1
		return super().explain(samples, **kwargs)



class Shapley_Value_Sampling(_TSCaptum_Method):

	def __init__(self, clf, clf_type=None):
		super().__init__(_ShapleyValueSamplingCaptum, clf, clf_type)

