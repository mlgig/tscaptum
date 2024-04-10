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
from utils import forward_classification, get_groups


# load data
X_train, y_train = load_basic_motions(split="train")
X_test, y_test = load_basic_motions(split="test")

# transform label into numerical ones as done with NNs.
# It's possible to use the method inverse_transform to switch back from int to string encoding
le = LabelEncoder()
y_train_labels = le.fit_transform(y_train)
y_test_labels = le.transform(y_test)

# define, train and test your classifier. If you wish to use another one just edit the following line
clf =  make_pipeline( MiniRocketMultivariate(), StandardScaler(), LogisticRegressionCV(n_jobs=-1))
#clf = TimeSeriesForestClassifier()
clf.fit(X_train,y_train_labels)
accuracy = clf.score(X_test,y_test_labels)
print(clf, accuracy)

# instantiate your attribution method
explainer = FeatureAblation(forward_classification)
# then call the method attribute. Two arguments:
# 1) instances you'd like to explain (pass it as torch tensor)
# 2) the initially define forward function has a second argument which is the model classifying, in our case clf
# 3) feature masks i.e. chunking: with the following lines you're computing the tensor representing how to group
# features. If you want point-wise explanations you need to remove feature_mask parameter
chunks = get_groups(n_chunks=10,n_channels=X_test.shape[1],series_length=X_test.shape[2])

attrs = explainer.attribute( torch.tensor(X_test) ,
                target=torch.tensor(y_test_labels), additional_forward_args=clf , feature_mask=chunks)
print("saliency maps:",attrs,
    "\n tensor dimension:", attrs.shape,"\t unique values:", torch.unique(attrs).size() )
