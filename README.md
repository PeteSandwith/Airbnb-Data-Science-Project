# Airbnb-Data-Science-Project
An industry grade data science project. A pipeline was created to systematically clean and perform feature engineering on airbnb data samples. Multiple machine learning models were trained; hyperparameters of the models were tuned to improve performance and iterative selection was used to add features to enhance the accuracy of the models. Tools used to deploy this system include AWS, PyTorch, Pandas, SkLearn and Tensorboard.

## Milestone 1
- A python script tabular_data.py was written to systematically load and clean the data relating to airbnb property listings. To manipulate the data set we chose to use the python library pandas, which is an extremely popular tool for data analysis and manipulation. By loading the dataset as a pandas dataframe actions can be performed on the whole dataset, based on certain conditions. For example, any items that had a missing value in the column relating to the number of beds were given a value of 1:
```
  dataframe['beds'].where(~dataframe['beds'].isna(), 1, inplace = True)
```
