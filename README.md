# Kidney Stone Detection Using AI

The dataset can be downloaded from the Kaggle link provided below:

https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone?resource=download

Dataset contains 12,446 images these are the combination of cyst, normal, stone, and tumor images. There are no null values in the dataset so no need to remove any data.
Normal: 5077, Cyst: 3709, Tumor: 2283, Stone: 1377 images.

Used Normalization and Data argumentation to preprocess the data. Data Argumentation techniques such as rotate, zoom, horizontal flip, adjusting the brightness.

The dataset is split in 80% for training and 20% for testing. Implemented KNN, Random Forest, Decision Trees, SVM, CNN algorithms.

Decided to move forward considering KNN and CNN algorithms. Used PCA as dimensionality reduction technique and tried with different k values and different hyperparameters combination to get the better performance. Implemented CNN with many layers and also included dropout layers ,maxpooling  layers and early stopping technique .
