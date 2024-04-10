import numpy as np
import torch

# following classifiers tested
from aeon.transformations.collection.convolution_based import Rocket, MiniRocketMultivariate, HydraTransformer
from aeon.datasets import  load_from_tsfile
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV, LinearRegression
from captum.attr import FeatureAblation,ShapleyValueSampling
from utils import forward_regression, get_groups

# load data
X_train, y_train = load_from_tsfile("./data/AppliancesEnergy_TRAIN.ts")
X_test, y_test = load_from_tsfile("./data/AppliancesEnergy_TEST.ts")

# define, train and test your classifier. If you wish to use another one just edit the following line
clf =  make_pipeline( MiniRocketMultivariate(), StandardScaler(), LinearRegression(n_jobs=-1))
clf.fit(X_train,y_train)
# in this case LinearRegression.score() returns "coefficient of determination R^2"
score = clf.score(X_test,y_test)
print(clf, score)

# instantiate your attribution method
explainer = FeatureAblation(forward_regression)

# then call the method attribute. Three arguments:
# 1) instances you'd like to explain (pass it as torch tensor)
# 2) the initially define forward function has a second argument which is the model classifying, in our case clf
# 3) feature masks i.e. chunking: with the following lines you're computing the tensor representing how to group
# features. If you want point-wise explanations you need to remove feature_mask parameter
chunks = get_groups(n_chunks=10,n_channels=X_test.shape[1],series_length=X_test.shape[2])

attrs  = explainer.attribute( torch.tensor( X_test ), additional_forward_args=clf, feature_mask= chunks)
attrs_numpy = attrs.detach().numpy()
print( "saliency maps:",attrs_numpy,
       "\n dataset dimension:", X_test.shape,"\t attribution tensor dimension:", attrs_numpy.shape,
       "\t unique values in it:", np.unique(attrs_numpy).size )
