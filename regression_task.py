import numpy as np
import torch

# following classifiers tested
from aeon.transformations.collection.convolution_based import Rocket, MiniRocketMultivariate, HydraTransformer
from aeon.datasets import load_basic_motions, load_regression, load_from_tsfile
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV, LinearRegression
from captum.attr import FeatureAblation,ShapleyValueSampling


# function to adapt sklearn/aeon classifiers to captum
def forward(X_test : torch.Tensor, model):
	# convert X to pytorch tensor
	X_test_numpy = X_test.detach().numpy()
	# use the model forward function
	predictions = model.predict(X_test_numpy)
	# return result as torch tensor as expected by captum attribution method
	return torch.tensor(predictions)


# load data
X_train, y_train = load_from_tsfile("./data/AppliancesEnergy_TRAIN.ts")
X_test, y_test = load_from_tsfile("./data/AppliancesEnergy_TEST.ts")

# define, train and test your classifier. If you wish to use another one just edit the following line
clf =  make_pipeline( MiniRocketMultivariate(), StandardScaler(), LinearRegression())
clf.fit(X_train,y_train)
# in this case LinearRegression.score() returns "coefficient of determination R^2"
score = clf.score(X_test,y_test)
print(clf, score)

# instantiate your attribution method
explainer = FeatureAblation(forward)
# then call the method attribute. Two arguments:
# 1) instances you'd like to explain (pass it as torch tensor)
# 2) the initially define forward function has a second argument which is the model classifying, in our case clf
attrs  = explainer.attribute( torch.tensor( X_test ), additional_forward_args=clf )
attrs_numpy = attrs.detach().numpy()
print( attrs_numpy, X_test.shape, attrs_numpy.shape)
