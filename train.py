from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core.model import Model # for Model Deserialization Opt#1
from sklearn.externals import joblib

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

# ds = ### YOUR CODE HERE ###
# # ds = ### Refactored CODE begins BELOW ###
dflow_dprep_link = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
# # Showing the file
# %pfile dflow_dprep_link # !data_urls
ds = TabularDatasetFactory.from_delimited_files(path=dflow_dprep_link)
if "outputs" not in os.listdir():
    os.mkdir("./outputs")
# x, y = clean_data(### YOUR DATA OBJECT HERE ###)
# from train import clean_data
# # ds = ### Refactored CODE ends ABOVE ###
# x, y = clean_data(ds)

# TODO: Split data into train and test sets.

### YOUR CODE HERE ###a
# # Split ### Refactored CODE begins BELOW ###
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state = 0)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) # .2 # , random_state = 32
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 32) # (X, y, , )
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=32)
""" 
x_train.reset_index(inplace=True, drop=True)
x_test.reset_index(inplace=True, drop=True)
y_train.reset_index(inplace=True, drop=True)
y_test.reset_index(inplace=True, drop=True)
y_train.values.flatten()
 """
#run = Run.get_context() # Alt # !## # !## # !##
""" 
import json
ds = json.loads(dflow_dprep_link)['ds']
ds = np.array(ds)
foresights_result = model.predict(ds)
foresights_result.tolist()
foresights_result
 """
# # Split ### Refactored CODE ends ABOVE ###

# run = Run.get_context()

def clean_data(data):
    # Dict for cleaning data
    months = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
    weekdays = {"mon":1, "tue":2, "wed":3, "thu":4, "fri":5, "sat":6, "sun":7}

    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    jobs = pd.get_dummies(x_df.job, prefix="job")
    x_df.drop("job", inplace=True, axis=1)
    x_df = x_df.join(jobs)
    x_df["marital"] = x_df.marital.apply(lambda s: 1 if s == "married" else 0)
    x_df["default"] = x_df.default.apply(lambda s: 1 if s == "yes" else 0)
    x_df["housing"] = x_df.housing.apply(lambda s: 1 if s == "yes" else 0)
    x_df["loan"] = x_df.loan.apply(lambda s: 1 if s == "yes" else 0)
    contact = pd.get_dummies(x_df.contact, prefix="contact")
    x_df.drop("contact", inplace=True, axis=1)
    x_df = x_df.join(contact)
    education = pd.get_dummies(x_df.education, prefix="education")
    x_df.drop("education", inplace=True, axis=1)
    x_df = x_df.join(education)
    x_df["month"] = x_df.month.map(months)
    x_df["day_of_week"] = x_df.day_of_week.map(weekdays)
    x_df["poutcome"] = x_df.poutcome.apply(lambda s: 1 if s == "success" else 0)

    y_df = x_df.pop("y").apply(lambda s: 1 if s == "yes" else 0)
    # # Making clean_data() iterable?
    """ 
    Traceback (most recent call last):
    File "train.py", line 109, in <module>
    x, y = clean_data(ds)
    TypeError: 'NoneType' object is not iterable
     """
    # return x_df
    return x_df, y_df


# # Cleaning & Spitting AFTER defining above clean_data function; BEFORE above main() for difining run.log
""" 
Traceback (most recent call last):
  File "train.py", line 30, in <module>
    x, y = clean_data(ds)
NameError: name 'clean_data' is not defined
 """
x, y = clean_data(ds)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 32) # (X, y, , )
run = Run.get_context()
""" 
Traceback (most recent call last):
  File "train.py", line 107, in <module>
    main()
  File "train.py", line 98, in main
    run.log("Regularization Strength:", np.float(args.C))
NameError: name 'run' is not defined
 """

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()

