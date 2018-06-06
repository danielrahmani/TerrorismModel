# TerrorismData

In this project the Global Terrorism Database (GTD) is studied. The ultimate goal of the project is to use attack type, weapons used, description of the attack, etc. to build a model that can predict what group may have been responsible for an incident. 

The data set is stored in data.zip file. Download and unzip the file to get the dataset (terrorist_data.csv).

There are three main notebooks in this project and each one can be run independently:

1. Terrorism_model
2. Terrorism_deep
3. Terrorism_Visualization


## Terrorism_model
The Terrorism_ML notebook is the major notebook in this project, in which various machine learning models such as XGBoost Classifier, Logistic Regression Classifier, C-Support Vector Classifier, etc. are built and their accuracy are compared.

For imputing nulls in the dataset:
1. Text features such as summary are imputed by empty.
2. Nominal/categorical features are imputed by just a different new value. (e.g. -9 in binary features)
3. Numerical features are imputed using fancyimpute, but a new binary feature is created for each numerical features with 1 if the value was missing and 0 otherwise.

The text features that contain explanatory information (summary and scite1) are stemmed using nltk package. After stemming all other text features are merged and stopwords, punctuations, numbers and white spaces are removed.

Text features are vectorized using TfidfVectorizer and scaled by MaxAbsScaler.

Categorical features are also binarized using LabelBinarizer.

All the number features are also scaled by MaxAbsScaler.

The XGBoost classifier is used to find the most important features and models are trained using important features afterwards.

For hyperparameter selection I have used hyperopt package which applies the Bayesian optimization on the hyperparameter space to find hyperparameters that minimize the loss (maximize the accuracy). At the end predictions are made using VotingClassifier. 

## Terrorism_deep
In the Terrorism_deep notebook two deep neural model are built using Keras. In one model, only numerical features are used and in another model all features are used. The accuracy of models are compared. 

The preprocessing is more or less the same as the Terrorism_ML notebook. One major difference is that instead of using TfidfVectorizer for text features, pre_trained GloVe embedding is used for explanatory features (summary and scite1) and onehot encoder is used for other text features. Keras Functional API is used to build a network with three inputs. Two embedding layers for text features and one dense layer for numerical features.

To use GloVe pretrained package download the smallest package of embeddings  (822Mb), called "glove.6B.zip" from https://nlp.stanford.edu/projects/glove/. After downloading and unzipping, you will see a few files, one of which is "glove.6B.100d.txt", which contains a 100-dimensional version of the embedding. We will use this file.


## Terrorism_Visualization
In the visulaization notebook (Terrorism_Visualization), various types of basuc analysis are performed and the results are visualized. Analysis and visualization on word frequency, word cloud, statistical information (e.g. the most deadly year, country with highest number of fatalities), world map of fatalities, missing values, etc. are presented in this notebook.

### Comparison
There is a similar GitHub project by Chris Cronin at https://ccronin51.github.io/StartupML%20Application%20Project.html#StartupML on the same dataset. I got some ideas specially on using pipelines from this project, but my appraoch for data postprocessing, imputation, stemming, handling the categorical features, hyperparameter selection, etc. are completely different. Here is a comparison between the accuracy of two projects:

| Models        | Accuracy (Cronin)  | Accuracy (Rahmani)  |
| ------------- |:-------------:| :-----:|
| Decision Tree    | 0.9478| 0.9621 |
| Random Forest       | 0.9472     |  0.9689  |
| Logistic Regression | 0.9337      |  0.9763   |
| SVC | 0.8856| 0.97134|
| Naive Bayes | 0.8384| 0.6149|
| KNN |0.7218 |0.8701 | 
| deep | -- | 0.9328|
| Hard Vote | 0.9328| 0.9779|
| Soft Vote | 0.9265 | 0.9777|


### References
Here is the list of references and libraries which I got ideas from them or used them:

Codebook: http://apps.start.umd.edu/gtd/downloads/dataset/Codebook.pdf 

XGBoost: http://xgboost.readthedocs.io/en/latest/

Hyperopt: https://github.com/hyperopt

nltk: https://www.nltk.org/

GloVe: https://nlp.stanford.edu/projects/glove/

missingo: https://github.com/ResidentMario/missingno 

word cloud: https://github.com/amueller/word_cloud

Fancy Impute: https://github.com/iskandr/fancyimpute

Ideas on visualization: https://www.kaggle.com/abigaillarion/terrorist-attacks-in-united-states

For more details on keras functional API which covers the multiple input model look at: https://machinelearningmastery.com/keras-functional-api-deep-learning/

For more details on embedding layer in keras look at: https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

For the imputation strategy: http://vikas.sindhwani.org/KDDCup-jmlr09.pdf

Bayesian optimization: https://thuijskens.github.io/2016/12/29/bayesian-optimisation/#parameter-selection-of-a-support-vector-machine




