# Kidney Stone Detection Using AI

The dataset can be downloaded from the Kaggle link provided below:

https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone?resource=download

After downloading the dataset to your local you can use Jupyter or google colab to test.If using google colab mount the drive to the folder where the dataset is present.

Dataset contains 12,446 images these are the combination of cyst, normal, stone, and tumor images. There are no null values in the dataset so no need to remove any data.
Normal: 5077, Cyst: 3709, Tumor: 2283, Stone: 1377 images.

Used Normalization and Data argumentation to preprocess the data. Data Argumentation techniques such as rotate, zoom, horizontal flip, adjusting the brightness.

The dataset is split in 80% for training and 20% for testing. Implemented KNN, Random Forest, Decision Trees, SVM, CNN algorithms.

Decided to move forward considering KNN and CNN algorithms. Used PCA as dimensionality reduction technique and tried with different k values and different hyperparameters combination to get the better performance. Implemented CNN with many layers and also included dropout layers ,maxpooling  layers and early stopping technique .

Requirements for KidneyDetectionApp to run:
Import all the packages below 
Flask
tensorflow
numpy
Pillow
scikit-image
matplotlib
keras
split-folders
scikit-learn
seaborn
streamlit

commands to run :
python app.py

Want to train the model and generate a new kid_desease_classification_model_CNN.h5 :
python  train_save_model_CNN.py 

If you are using Mac with m1 chip 
Pre-requisites:
You need to set up a virtual environment 
conda config –set auto_activate_base false
conda activate mlp
python app.py

IDE’s – 
PyCharm
Visual Studio Code
