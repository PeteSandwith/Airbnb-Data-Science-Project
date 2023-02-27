# Airbnb-Data-Science-Project
An industry grade data science project. A pipeline was created to systematically clean and perform feature engineering on airbnb data samples. Multiple machine learning models were trained; hyperparameters of the models were tuned to improve performance and iterative selection was used to add features to enhance the accuracy of the models. Tools used to deploy this system include AWS, PyTorch, Pandas, SkLearn and Tensorboard.

## Milestone 1
- A python script tabular_data.py was written to systematically load and clean the data relating to airbnb property listings. To manipulate the data set we chose to use the python library pandas, which is an extremely popular tool for data analysis and manipulation. By loading the dataset as a pandas dataframe actions can be performed on the whole dataset, based on certain conditions. For example, any items that had a missing value in the column relating to the number of beds were given a value of 1:
```
dataframe['beds'].where(~dataframe['beds'].isna(), 1, inplace = True)
```
whilst the presence of missing data in some columns meant that the entire row of the table had to be removed:
```
dataframe.dropna(axis = 0, how = 'any', subset = ['Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating'], inplace = True)
```
- The description column contained text data that was saved in the form of a list of strings. A specific function was created to parse this data so that it could be saved as a single string object with no erroneous whitespace:
```
def parses_description_strings(string):
    try:
        #Uses ast package to parse the string description into a list
        description_as_list = ast.literal_eval(string)
        description_as_list.pop(0)
        #Removes empty quotes from the list
        number_empty_quotes = description_as_list.count('')
        for index in range(0,number_empty_quotes):
            description_as_list.remove('')
        #Turns the list into a single string containing the description
        cleaned_description = ' '.join(description_as_list)
        return cleaned_description
    except: 
        return string
```
