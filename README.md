# QuorainsincereQuestions
Quora insincere Question classification using RNN Models (Kaggle dataset

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

This project uses Python 3, along with Jupyter Notebook. The following libraries are necessary for running the notebook:
* Pandas
* Numpy
* PyTorch
* Scikit-Learn

Packages used by this project can also be installed as a Conda Environment using the provided Requirements.txt file.

## Project Motivation<a name="motivation"></a>

For this project, I was interested in exploring a dataset from start to finish, with the following steps:
1. Data Preprocessing to clean up raw data into meaningful features
2. Training a LSTM Based Neural Network on the data to obtain results
3. Tuning the model with different configurations and layers to improve results

## File Descriptions <a name="files"></a>

The main code for this project is divided into 5 notebooks. Each notebook is properly documented with Markdown cells to indicate the steps taken.

* `Data Investigation.ipynb` - This notebook does an initial analysis of the raw data.
* `Data Cleaning.ipynb` - In this notebook, the raw data is cleaned up, with steps taken to make the text records more meaningful for training.
* `Data Augmentation.ipynb` - In this notebook, the existing data is augmented with additional records to reduce label bias.
* `Model Training - Basic Model.ipynb` - In this notebook, a basic model with LSTM layers is ued for training and obtaining results on the model.
* `Model Training - Different Models.ipynb` - In this notebook, different model and training configurations are used to see how results from the basic model can be improved upon.

The code and results are also posted on Medium as a [blog post](https://medium.com/ml2vec/learning-insincerity-of-quora-questions-using-lstm-networks-f866ea51957e).

Data for the project is not included, and can be downloaded directly from Kaggle [here](https://www.kaggle.com/c/quora-insincere-questions-classification/overview). To properly run the notebooks, they must be kept in the `data` directory as files `data/train.csv` and `data/test.csv`. 

## Results<a name="results"></a>

The final model trained obtained a score of `0.771` on the training F1 metric and a `0.601` on Validation F1. The model with the most improvement had the following configuration:

Neural Model with a 128-dim LSTM Layer, Attention Layer, and 2 Fully Connected Layers

300 Dimensional Fast Text word Embeddings

Adam Optimizer with a 0.003 Learning Rate and no weight decay

Trained on 5 Epochs


More detailed findings can be found at the post available [here](https://medium.com/ml2vec/learning-insincerity-of-quora-questions-using-lstm-networks-f866ea51957e).

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credit to Kaggle for providing the data. You can find the Licensing for the data and other descriptive information from [Kaggle](https://www.kaggle.om). This code is free to use.
