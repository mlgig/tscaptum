from tqdm import tqdm
import warnings

from captum.attr._utils.attribution import PerturbationAttribution
from captum.attr import (FeatureAblation as _FeatureAblationCaptum,
                         FeaturePermutation as _FeaturePermutationCaptum,
                         KernelShap as _KernelShapCaptum,
                         Lime as _LimeCaptum,
                         ShapleyValueSampling as _ShapleyValueSamplingCaptum)

from ._forwarder import _Forwarder
from ._utils import _check_convert_data_format, _check_labels, _normalise_result, _equal_length_segmentation
from ._utils import *


class _tsCaptum_Method:
	"""
	super class for all attribution methods
	"""

	def __init__(self, explainer, predictor, predictor_type: str = None):
		r"""
		init method for the superclass

		:param explainer:       the actual explainer that will be used for computing the saliency maps.
		Each subclass fix this argument as the corresponding Captum explainer

		:param predictor:       the predictor that will be explained
		:param predictor_type:  which type the predictor is i.e. classifier or regressor
		"""

		# check argument values
		if not issubclass(explainer, PerturbationAttribution):
			raise (
				" provided explainer has to be an instance of 'captum.attr._utils.attribution.PerturbationAttribution' ")

		if predictor_type not in ["classifier", "regressor", None]:
			raise (
				" clf_type argument has to be either  'classifier' or 'regressor' ")

		# in case predictor_type argument isn't provided tell it calling the predict_proba method:
		# if it's present the predictor is a classifier otherwise is a regressor
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
		r"""
		main method to get a saliency map by the selected explainer

		:param samples:     samples to be explained

		:param labels:      labels associated to samples in case of classification

		:param batch_size:  the batch_size to be used i.e. number of samples to be explained at the same time

		:param n_segments:  number of segments the timeseries is dived to. If you want to explain point-wise provide -1 as
		value #TODO should we put more information about how it works?

		:param normalise:   whether or not to normalise the result

		:param baseline:    the baseline which will substitute time series's values when ablated. It can be either a scalar
		(each time series's value is substituted by this scalar)  or a single time series
		(both as numpy array or torch.tensor)

		:return:    the saliency maps as a 3D tensor ( samples, channels, time points)
		"""

		# check arguments and get a DataLoader
		n_2explain, n_channels, series_length = samples.shape
		le, labels_idx = _check_labels(labels, self.predictor_type)
		loader = _check_convert_data_format(samples, labels_idx, batch_size)

		explanations = []
		with tqdm(total=n_2explain) as pbar:
			with torch.no_grad():
				for n, (X, y) in enumerate(loader):
					# fix kwargs for the relative captum method
					kwargs = self._define_kwargs(baseline, n_channels, n_segments, series_length, y)
					# get the current saliency maps, convert it to numpy array and store it to a temp list
					current_exps = self._explainer.attribute(X, **kwargs)
					explanations.append(
						_normalise_result(current_exps.detach().numpy())) if normalise \
						else explanations.append(current_exps.detach().numpy())
					pbar.update(batch_size)
		pbar.close()

		# convert the list to numpy array and return it as result
		explanations = np.concatenate(explanations)
		return explanations

	def _define_kwargs(self, baseline, n_channels, n_segments, series_length, y):
		r"""
		inner function that checking provided argument to explain, define the correct kwarg dictionary

		:return: kwarg dictionary for the relative captum method
		"""
		kwargs = {}

		# checking baseline
		if not isinstance(self, Feature_Permutation):
			if np.isscalar(baseline):
				kwargs['baselines'] = baseline
			elif type(baseline) is np.ndarray:
				kwargs['baselines'] = torch.tensor(baseline)
			elif type(baseline) is torch.Tensor:
				kwargs['baselines'] = baseline

		# labels
		if self.predictor_type == "classifier":
			kwargs['target'] = y

		# define feature mask looking at the desired number of segment
		if n_segments != -1:
			groups = _equal_length_segmentation(n_segments, n_channels, series_length)
			kwargs['feature_mask'] = groups

		return kwargs


class Feature_Ablation(_tsCaptum_Method):
	r"""
	Wrapper for Feature Ablation method
	"""

	def __init__(self, clf, clf_type=None):
		super().__init__(_FeatureAblationCaptum, clf, clf_type)


class Feature_Permutation(_tsCaptum_Method):
	r"""
	Wrapper for feature permutation method
	"""

	def __init__(self, clf, clf_type=None):
		super().__init__(_FeaturePermutationCaptum, clf, clf_type)

	def explain(self, samples, **kwargs):
		r"""
		extending _tsCaptum_Method's explain by fixing some arguments as defined by Captum implementation.
		For Feature Permutation takes care of the batch_size > 2 and baseline which can't be provided

		:param samples: samples to be explained

		:param kwargs:  additional arguments

		:calling:       _tsCaptum_Method'e explain
		"""
		if 'batch_size' in kwargs and kwargs['batch_size'] != 1:
			warnings.warn(
				"batch_size set to 2 as Feature Permutation require more than 1 sample to work"
			)
			kwargs['batch_size'] = 2

		if 'baseline' in kwargs:
			warnings.warn(
				"specified baseline will be ignored as Feature Permutation algorithm has its own baseline"
			)
		return super().explain(samples, **kwargs)


class Kernel_Shap(_tsCaptum_Method):
	r"""
	Wrapper for KernelSHAP method
	"""

	def __init__(self, clf, clf_type=None):
		super().__init__(_KernelShapCaptum, clf, clf_type)

	def explain(self, samples, **kwargs):
		r"""
		extending _tsCaptum_Method'e explain by fixing some arguments as defined by Captum implementation.
		For KernelSHAP takes care of the batch_size suggested to be equal to 1

		:param samples: samples to be explained

		:param kwargs:  additional arguments

		:calling:       _tsCaptum_Method's explain
		"""
		if 'batch_size' in kwargs and kwargs['batch_size'] != 1:
			warnings.warn(
				"batch_size set to 1 as suggested by Captum for Lime and KernelSHAP"
			)
		kwargs['batch_size'] = 1
		return super().explain(samples, **kwargs)


class LIME(_tsCaptum_Method):
	r"""
	Wrapper for LIME method
	"""

	def __init__(self, clf, clf_type=None):
		super().__init__(_LimeCaptum, clf, clf_type)

	def explain(self, samples, **kwargs):
		r"""
		extending _tsCaptum_Method'e explain by fixing some arguments as defined by Captum implementation.
		For LIME takes care of the batch_size suggested to be equal to 1

		:param samples: samples to be explained

		:param kwargs:  additional arguments

		:calling:       _tsCaptum_Method's explain
		"""
		if 'batch_size' in kwargs and kwargs['batch_size'] != 1:
			warnings.warn(
				"batch_size set to 1 as suggested by Captum for Lime and KernelSHAP"
			)
		kwargs['batch_size'] = 1
		return super().explain(samples, **kwargs)


class Shapley_Value_Sampling(_tsCaptum_Method):
	r"""
	Wrapper for Shapley Value Sampling method. Most of the time this is the best approximation of the intractable
	Shapley values
	"""

	def __init__(self, clf, clf_type=None):
		super().__init__(_ShapleyValueSamplingCaptum, clf, clf_type)
