#  MMD-Assignment #2: SVM-using-SGD

## Team: Emin Guliev, Justus Rass & Christian Wiskott.

### Requirements
- Python 3
- Numpy
- Matplotlib
- Jupyter
- Seaborn
- Tqdm

The DataLoader module assumes that the folder "datasets" is in the root folder. If that's not the case,
pass your file-path to the DataLoader keyword-argument >path<.

### Description
Application and comparison of a LinearSVM model using regular vs. Random Fourier Features on two toydata sets as well
as the MNIST dataset. Includes a parallel implementation of the SVM + comparison of runtimes.

The included jupyther-notebooks provide all results used in the report and can be executed using a jupyther environment.