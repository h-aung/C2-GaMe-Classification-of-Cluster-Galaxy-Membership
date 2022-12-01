"""
Module for C2-GaMe (Classification of Cluster Galaxy Membership) model.

This module gives access to trained Random Forest (RF), k-Nearest-Neighbors (kNN),
and Linear Support Vector Classifier (SVC) models, trained with MDPL2 projected phase-space
(2D) data.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from joblib import load
from dataclasses import dataclass

TRAINED_RF_CLF_PATH = "classifiers/clf_rf.joblib"
TRAINED_KNN_CLF_PATH= "classifiers/clf_knn.joblib"

@dataclass
class RF:
    """
    Random Forest model using 100 decision trees. This class gives access to the trained RF classifier, as well
    as two methods which use the classifier to make predictions.

    One method outputs deterministic predictions, and the other one outputs probabilistic
    predictions.
    """
    classifier: RandomForestClassifier = load(TRAINED_RF_CLF_PATH)
    
    def predict_det(self, data: pd.DataFrame) -> np.ndarray:
        """
        Use the trained RF classifier to predict the labels in the given data.

        Outputs deterministic classification for each row in data (either orbiting: 0,
        infalling: 1, or interloper: 2).

        Columns that must exist in the data DataFrame:
            d2d: 2d radius
            v: LOS velocity
            mratio: mass ratio
            ssfr: specific star formation rate
        """
        return self.classifier.predict(data)
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Use the trained RF classifier to predict the labels in the given data.

        Outputs probabilistic classification for each row in data. Each item in the output
        array is a list of three probabilities between 0 and 1: 
            [p(orbiting), p(infalling), p(interloper)]
        
        Columns that must exist in the data DataFrame:
            d2d: 2d radius
            v: LOS velocity
            mratio: mass ratio
            ssfr: specific star formation rate
        """
        return self.classifier.predict_proba(data)


@dataclass
class KNN:
    """
    k-Nearest-Neighbots model using 15 neighbors. This class gives access to the 
    trained KNN classifier, as well as two methods which use the classifier to make 
    predictions.

    One method outputs deterministic predictions, and the other one outputs probabilistic
    predictions.
    """
    classifier: KNeighborsClassifier= load(TRAINED_KNN_CLF_PATH)
    
    def predict_det(self, data: pd.DataFrame) -> np.ndarray:
        """
        Use the trained KNN classifier to predict the labels in the given data.

        Outputs deterministic classification for each row in data (either orbiting: 0,
        infalling: 1, or interloper: 2).

        Columns that must exist in the data DataFrame:
            d2d: 2d radius
            v: LOS velocity
            mratio: mass ratio
            ssfr: specific star formation rate
        """
        return self.classifier.predict(data)
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Use the trained KNN classifier to predict the labels in the given data.

        Outputs probabilistic classification for each row in data. Each item in the output
        array is a list of three probabilities between 0 and 1: 
            [p(orbiting), p(infalling), p(interloper)]
        
        Columns that must exist in the data DataFrame:
            d2d: 2d radius
            v: LOS velocity
            mratio: mass ratio
            ssfr: specific star formation rate
        """
        return self.classifier.predict_proba(data)
