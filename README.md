# Classification of Galaxy Cluster Membership with Machine Learning
Classification of Galaxy Cluster Membership with Machine Learning code for importing trained models and generating figures.

## C2GaMe.py
This module includes classes for RF, KNN, and Logistic Regression which give access to the trained classifiers, as well as methods to make deterministic and probabilistic predictions.

See the C2GaMe_example.py file (and below) for an example of how to import and use the module.

```
"""
This module gives examples of how to use the C2GaMe module.
"""
import C2GaMe
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# gain access to RF model class which uses sSFR as an input feature
RF = C2GaMe.RF(sSFR=True)

# gain access to the classifier object itself
RF_clf: RandomForestClassifier = RF.classifier

example_data = pd.DataFrame({
    'd2d': np.arange(10),
    'v': np.arange(10),
    'ssfr': np.arange(10)
})

# use the RF object to make deterministic predictions
RF.predict_det(example_data)

# use the RF object to make probabilistic predictions
RF.predict_proba(example_data)

# gain access to the KNN model class which does not use sSFR as an input feature
KNN = C2GaMe.KNN()


# gain access to the classifier object itself
KNN_clf: KNeighborsClassifier = KNN.classifier

example_data_no_ssfr = pd.DataFrame({
    'd2d': np.arange(10),
    'v': np.arange(10),
})

# use the RF object to make deterministic predictions
KNN.predict_det(example_data_no_ssfr)

# use the RF object to make probabilistic predictions
KNN.predict_proba(example_data_no_ssfr)


```

## C2GaMe_Figures.ipynb

To access and download files necessary to generate the figures in the C2GaMe_Figures.ipynb file, please see the "Figure Files" folder at [https://drive.google.com/drive/folders/1qjy32e_gtoafJlajz9N8v14aPLFbhX8o?usp=sharing](https://drive.google.com/drive/folders/1SxetDYJoWt_sLOc9spp3FttuM-cjRg29?usp=sharing)

Once downloaded, edit the filepaths in the "Filepaths" section of the Jupyter Notebook to point to the appropriate files.

## C2-GaMe-Statistical-Metrics.ipynb

This notebook shows computation of each of the statistical metrics presented in the paper's tables. Also, this notebook contains code for the generation of the ROC curves (Fig 2) and calibration plots (Fig 3) figures. The saved data is in the "Saved Predictions" folder at [https://drive.google.com/drive/folders/1qjy32e_gtoafJlajz9N8v14aPLFbhX8o?usp=sharing](https://drive.google.com/drive/folders/1SxetDYJoWt_sLOc9spp3FttuM-cjRg29?usp=sharing) for download. 

Once downloaded, edit filepaths appropriately.

## C2GaMe Classifiers

To access and download the RF, KNN, and Logistic Regression classifiers used in the C2GaMe module, please see the "Classifiers" folder at [https://drive.google.com/drive/folders/1qjy32e_gtoafJlajz9N8v14aPLFbhX8o?usp=sharing](https://drive.google.com/drive/folders/1SxetDYJoWt_sLOc9spp3FttuM-cjRg29?usp=sharing).

Once downloaded, place them in a folder called "classifiers" in the same directory as the C2GaMe.py module.
