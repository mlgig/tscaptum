# tsCaptum - A library for univariate and multivariate time series explanation

tsCaptum is a library that adapts the popular post-hoc attribution methods implemented in the Captum
framework to work with time series. Unlike previous libraries tsCaptum is :
1) Fully compatible with scikit-learn and popular time-series frameworks such as aeon and sktime (e.g., so it is easy to combine ROCKET with SHAP).
2) Takes advantage of TS locality by using time series segmentation (so SHAP runs fast with long time series and suffers less from vanishing attribution weights) 
3) It's extremely easy to use requiring almost no effort by the users

```
from aeon.classification.convolution_based import RocketClassifier
clf = RocketClassifier(n_jobs=-1)
clf.fit(MP_X_train,MP_y_train)

from tsCaptum.explainers import Feature_Ablation
myFA = Feature_Ablation(clf)
exp = myFA.explain(samples=CMJ_X_test_samples, labels=CMJ_y_test_samples, n_segments=10, normalise=False, baseline=0)

from tsCaptum.explainers import Shapley_Value_Sampling as SHAP
mySHAP = SHAP(clf)
exp = mySHAP.explain(CMJ_X_test_samples, labels=CMJ_y_test_samples,  n_segments=10, normalise=False, baseline=0)

```

More examples can be found in the [examples_tscaptum notebook](https://github.com/mlgig/tscaptum/blob/main/examples_tscaptum.ipynb) with detailed steps for training, prediction and 
explanation.
Some of the attribution methods available in captum have been described and quantitatively evaluated for time series explanation in our recent [ECMLPKDD 2024 paper](https://github.com/mlgig/xai4mtsc_eval_actionability/tree/main?tab=readme-ov-file).

## Installation using pip
```
pip install tsCaptum
```

In case you have problems installing torch (e.g. you use Linux as OS and you want a lighter installation) use 
```
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install tsCaptum
```

## Citation
In case you use this library please cite:
```
@misc{tsCaptum,
    author = {Davide Italo Serramazza, Thach Le Nguyen, Georgiana Ifrim},
    title = {A Short Tutorial for Multivariate Time Series Explanation using tsCaptum},
    journal = {Software Impacts},
    year = {2024},
    publisher = {Elsevier},
    note = {In Press},
    url = { https://github.com/mlgig/tscaptum},
}
```
