# Charity-Funding-Predictor

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. I use the features in the dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.
CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:


* EIN and NAME—Identification columns

* APPLICATION_TYPE—Alphabet Soup application type

* AFFILIATION—Affiliated sector of industry

* CLASSIFICATION—Government organization classification

* USE_CASE—Use case for funding

* ORGANIZATION—Organization type

* STATUS—Active status

* INCOME_AMT—Income classification

* SPECIAL_CONSIDERATIONS—Special consideration for application

* ASK_AMT—Funding amount requested

* IS_SUCCESSFUL—Was the money used effectively


## Steps:

## Step 1: Preprocess the Data
### Using my knowledge of Pandas and scikit-learn’s StandardScaler(), I preprocess the dataset. This step prepares me for Step 2, where I compile, train, and evaluate the neural network model.

* Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in the dataset:
1.  What variable(s) are the target(s) for your model?
2.  What variable(s) are the feature(s) for your model?
* Drop the EIN and NAME columns.
* Determine the number of unique values for each column.
* For columns that have more than 10 unique values, I determine the number of data points for each unique value.
* Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.
* Use pd.get_dummies() to encode categorical variables.



## Step 2: Compile, Train, and Evaluate the Model
### Using my knowledge of TensorFlow, I design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. I think about how many inputs there are before determining the number of neurons and layers in your model. Once I completed that step, I compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.


* Continue using the Jupyter Notebook in which I performed the preprocessing steps from Step 1.
* Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
* Create the first hidden layer and choose an appropriate activation function.
* Add a second hidden layer with an appropriate activation function.
* Create an output layer with an appropriate activation function.
* Check the structure of the model.
* Compile and train the model.
* Evaluate the model using the test data to determine the loss and accuracy.
* Save and export my results to an HDF5 file. Name the file AlphabetSoupCharity.h5.



## Step 3: Optimize the Model
### Using my knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.
* Using any or all of the following methods to optimize your model:
* Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
1. Dropping more or fewer columns.
2. Creating more bins for rare occurrences in columns.
3. Increasing or decreasing the number of values for each bin.
* Add more neurons to a hidden layer.
* Add more hidden layers.
* Use different activation functions for the hidden layers.
* Add or reduce the number of epochs to the training regimen.
* Create a new Jupyter Notebook file and name it AlphabetSoupCharity_Optimzation.ipynb.
* Import my dependencies and read in the charity_data.csv to a Pandas DataFrame.
* Preprocess the dataset like I did in Step 1, and adjust for any modifications that came out of optimizing the model.
* Design a neural network model, and adjust for modifications that will optimize the model to achieve higher than 75% accuracy.
* Save and export my results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.



## Step 4: Write a Report on the Neural Network Model
### For this part of project, I write a report on the performance of the deep learning model I created for AlphabetSoup.
### The report contains the following:
* Overview of the analysis: Explain the purpose of this analysis.
* Results: Using bulleted lists and images to support your answers, address the following questions.

#### Data Preprocessing

1. What variable(s) are the target(s) for the model?
2. What variable(s) are the features for the model?
3. What variable(s) should be removed from the input data because they are neither targets nor features?



#### Compiling, Training, and Evaluating the Model

1. How many neurons, layers, and activation functions did I select for your neural network model, and why?
2. Was I able to achieve the target model performance?
3. What steps did you take in your attempts to increase model performance?



### Summary
