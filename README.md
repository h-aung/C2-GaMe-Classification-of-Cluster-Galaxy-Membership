# Classification of Galaxy Cluster Membership with Machine Learning
Classification of Galaxy Cluster Membership with Machine Learning code for importing trained models and generating figures.

## C2GaMe.py
This module includes classes for RF, KNN, and SVC which give access to the trained classifiers, as well as methods to make deterministic and probabilistic predictions.

See the C2GaMe_example.py file (and below) for an example of how to import and use the module.

```
import C2GaMe
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# gain access to RF model class
RF = C2GaMe.RF()

# gain access to the classifier object itself
RF_clf: RandomForestClassifier = RF.classifier

example_data = pd.DataFrame({
    'd2d': np.arange(10),
    'v': np.arange(10),
    'mratio': np.arange(10),
    'ssfr': np.arange(10)
})

# use the RF object to make deterministic predictions
RF.predict_det(example_data)

# use the RF object to make probabilistic predictions
RF.predict_proba(example_data)

# gain access to the KNN model class
KNN = C2GaMe.KNN()


# gain access to the classifier object itself
KNN_clf: KNeighborsClassifier = KNN.classifier

# use the RF object to make deterministic predictions
KNN.predict_det(example_data)

# use the RF object to make probabilistic predictions
KNN.predict_proba(example_data)

```

## C2GaMe_Figures.ipynb

To access and download files necessary to generate the figures in the C2GaMe_Figures.ipynb file, please see the "Figure Files" folder at [https://drive.google.com/drive/folders/1qjy32e_gtoafJlajz9N8v14aPLFbhX8o?usp=sharing](https://drive.google.com/drive/folders/1SxetDYJoWt_sLOc9spp3FttuM-cjRg29?usp=sharing)

Once downloaded, edit the filepaths in the "Filepaths" section of the Jupyter Notebook to point to the appropriate files.
