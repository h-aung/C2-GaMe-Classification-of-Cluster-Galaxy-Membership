# Classification of Galaxy Cluster Membership with Machine Learning
Classification of Galaxy Cluster Membership with Machine Learning code for importing trained models and generating figures. This code accompanies the paper https://arxiv.org/abs/2205.01700.

## Using Your Own Data
The module `C2GaMe.py` includes classes for RF, KNN, and Logistic Regression which give access to the trained classifiers, as well as methods to make deterministic and probabilistic predictions.

See Example.ipynb notebook for an example of how to import and use the module.

To access and download the RF, KNN, and Logistic Regression classifiers used in the C2GaMe module, please see the "Classifiers" folder at [this OneDrive link](https://yaleedu-my.sharepoint.com/:f:/g/personal/han_aung_yale_edu/EpS074iVw_VBqgufJznFBnYBUzPiH9-gq5ACuDx8kkX6tQ?e=XOf2rH).

Once downloaded, place them in a folder called "classifiers" in the same directory as the C2GaMe.py module.


## Reproducing the Results

The two jupyter notebooks `C2GaMe_Figures.ipynb` and `C2-GaMe-Statistical-Metrics.ipynb` show how to reproduce the results in the paper. We also provide processed data (after making predictions with the classifiers) in the same [link](https://yaleedu-my.sharepoint.com/:f:/g/personal/han_aung_yale_edu/EpS074iVw_VBqgufJznFBnYBUzPiH9-gq5ACuDx8kkX6tQ?e=XOf2rH).

#### C2-GaMe_Figures.ipynb

To access and download files necessary to generate the figures in the C2GaMe_Figures.ipynb file, please see the "Figure Files" folder at [previous link](https://yaleedu-my.sharepoint.com/:f:/g/personal/han_aung_yale_edu/EpS074iVw_VBqgufJznFBnYBUzPiH9-gq5ACuDx8kkX6tQ?e=XOf2rH)

Once downloaded, edit the filepaths in the "Filepaths" section of the Jupyter Notebook to point to the appropriate files.

#### C2-GaMe-Statistical-Metrics.ipynb

This notebook shows computation of each of the statistical metrics presented in the paper's tables. Also, this notebook contains code for the generation of the ROC curves (Fig 2) and calibration plots (Fig 3) figures. The saved data is in the "Saved Predictions" folder at [previous link](https://yaleedu-my.sharepoint.com/:f:/g/personal/han_aung_yale_edu/EpS074iVw_VBqgufJznFBnYBUzPiH9-gq5ACuDx8kkX6tQ?e=XOf2rH) for download. 

Once downloaded, edit filepaths appropriately.

