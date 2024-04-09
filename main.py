import numpy as np
import torch

# following classifiers tested
from aeon.transformations.collection.convolution_based import Rocket, MiniRocketMultivariate, HydraTransformer
from aeon.classification.shapelet_based import ShapeletTransformClassifier,RDSTClassifier
from aeon.classification.interval_based import QUANTClassifier, TimeSeriesForestClassifier
from aeon.classification.dictionary_based import MUSE

from aeon.datasets import load_basic_motions
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV
from sklearn.preprocessing import LabelEncoder
from captum.attr import FeatureAblation,ShapleyValueSampling
from aeon.registry import all_estimators


# function to adapt sklearn/aeon classifiers to captum
def forward(X_test : torch.Tensor, model):
    # convert X to pytorch tensor
    X_test_numpy = X_test.detach().numpy()
    # compute probability
    predictions = model.predict_proba(X_test_numpy)
    # return result as torch tensor as expected by captum attribution method
    return torch.tensor(predictions)


# load data
X_train, y_train = load_basic_motions(split="train")
X_test, y_test = load_basic_motions(split="test")

# transform label into numerical ones as done with NNs.
# It's possible to use the method inverse_transform to switch back from int to string encoding
le = LabelEncoder()
y_train_labels = le.fit_transform(y_train)
y_test_labels = le.transform(y_test)

# define, train and test your classifier. If you wish to use another one just edit the following line
clf =  make_pipeline( MiniRocketMultivariate(), StandardScaler(), LogisticRegressionCV())
#clf = TimeSeriesForestClassifier()
clf.fit(X_train,y_train_labels)
accuracy = clf.score(X_test,y_test_labels)
print(clf, accuracy)

# instantiate your attribution method
explainer = FeatureAblation(forward)
# then call the method attribute. Two arguments:
# 1) instances you'd like to explain (pass it as torch tensor)
# 2) the initially define forward function has a second argument which is the model classifying, in our case clf
attrs  = explainer.attribute( torch.tensor( X_test ), additional_forward_args=clf )
print(attrs,attrs.shape, torch.sum(attrs), torch.unique(attrs) )
