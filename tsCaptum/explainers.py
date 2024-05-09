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
from ._utils import _check_convert_data_format, _check_labels, equal_length_segmentation


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

	def explain(self, samples, labels=None, batch_size=8, n_segments=-1):

		# TODO add chunking
		# TODO to normalize results ?
		n_2explain,n_channels, series_length = samples.shape
		le, labels_idx = _check_labels(labels,self.predictor_type)
		loader = _check_convert_data_format(samples,labels_idx,batch_size)


		explanations = []
		with tqdm(total=n_2explain) as pbar:
			with torch.no_grad():
				for n,(X,y) in enumerate(loader):
					# kwargs
					kwargs = {}
					if self.predictor_type == "classifier":
						kwargs['target']=y
					if n_segments!=-1:
						groups = equal_length_segmentation(n_segments,n_channels,series_length)
						kwargs['feature_mask'] = groups

					current_exp = self._explainer.attribute(X, **kwargs)
					explanations.append( current_exp.detach().numpy() )
					pbar.update(batch_size)
		pbar.close()
		explanations = np.concatenate(explanations)
		return explanations


class Feature_Ablation(_TSCaptum_Method):
	def __init__(self, clf, clf_type=None):
		super().__init__(_FeatureAblationCaptum, clf, clf_type)


class Feature_Permutation(_TSCaptum_Method):
	def __init__(self, clf, clf_type=None):
		super().__init__(_FeaturePermutationCaptum, clf, clf_type)

	def explain(self, samples, labels=None, batch_size=8,n_segments=-1):
		# TODO avoid warning mentioning itself
		if batch_size == 1:
			warnings.warn(" batch_size set to 2 as Feature Permutation require more than 1 sample to work")
		return super().explain(samples=samples,labels=labels,batch_size=2,n_segments=n_segments)

class Kernel_Shap(_TSCaptum_Method):
	def __init__(self, clf, clf_type=None):
		super().__init__(_KernelShapCaptum, clf, clf_type)

	def explain(self ,samples, labels=None, batch_size=8,n_segments=-1):
		# TODO avoid warning mentioning itself
		if batch_size != 1:
			warnings.warn(" batch_size set to 1 as suggested by Captum for Lime and KernelSHAP")
		return super().explain(samples=samples,labels=labels,batch_size=1,n_segments=n_segments)

class LIME(_TSCaptum_Method):
	def __init__(self, clf, clf_type=None):
		super().__init__(_LimeCaptum, clf, clf_type)

	def explain(self, samples, labels=None, batch_size=8,n_segments=-1):
		# TODO avoid warning mentioning itself
		if batch_size != 1:
			warnings.warn(" batch_size set to 1 as suggested by Captum for Lime and KernelSHAP")
		return super().explain(samples=samples,labels=labels,batch_size=1,n_segments=n_segments)

class SHAP(_TSCaptum_Method):
	def __init__(self, clf, clf_type=None):
		super().__init__(_ShapleyValueSamplingCaptum, clf, clf_type)

