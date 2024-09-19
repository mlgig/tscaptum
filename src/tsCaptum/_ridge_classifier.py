import numpy as np
from scipy.special import softmax
from sklearn.linear_model import *
from sklearn.pipeline import Pipeline
from aeon.classification.convolution_based import HydraClassifier, MultiRocketHydraClassifier, RocketClassifier


def isRidgeInstance(clf):
    """
    function to identify if a pipeline contains Ridge as classifier
    :param clf: classifier to be analyzed
    :return:
    """

    ridge_instances = [  HydraClassifier, MultiRocketHydraClassifier, RocketClassifier]
    ridge_classifiers = [ RidgeClassifier,RidgeClassifierCV]

    if type(clf) in ridge_instances:
        return True
    elif type(clf) == Pipeline and type(clf[-1]) in ridge_classifiers:
        return True

    return False


def proba4ridge(clf):
    """
    function to extract probabilities out of Ridge Classifier
    :param clf: Ridge Classifier to get probabilities of
    :return:
    """

    # first identify the feature transformation part and the final Ridge classifier
    feature_transform_foo, ridge = None, None

    if type(clf) == Pipeline :
        # case it's a sklearn pipeline having Ridge on top
        feature_transform_foo, ridge = clf[:-1].transform , clf[-1]
    elif type(clf)==RocketClassifier:
        # case it's a aeon's RocketClassifier
        feature_transform_foo, ridge = clf.pipeline_[:2].transform , clf.pipeline_[-1]
    elif type(clf)==HydraClassifier:
        # case it's a aeon's HydraClassifier
        feature_transform_foo, ridge = clf._clf[:2].transform ,  clf._clf[-1]
    elif type(clf)==MultiRocketHydraClassifier:
        # case it's aeon classifier combining Hydra and MultiRocket features
        ridge = clf.classifier

        feature_transform_foo = lambda  x :\
            np.concatenate(
                (
                    clf._scale_hydra.transform (
                    clf._transform_hydra.transform(x)),

                clf._scale_multirocket.transform(
                    clf._transform_multirocket.transform (x))
                ),
                axis=1
            )

    # after having identified the Ridge classifier extract probas
    if np.unique(clf.classes_).shape[0]<=2:
        # in case of binary classification use directly the distance from the hyperplane
        proba_foo = ridge._predict_proba_lr
    else:
        # otherwise use a softmax
        proba_foo = lambda X :  softmax(
            ridge.decision_function(X),
            axis=-1)

    return proba_foo, feature_transform_foo