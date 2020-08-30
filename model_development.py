from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import numpy as np
import cdsw

#new cdsw methods
def track_dataset(df, source, transformation, tag):
  df = df #this could be a pandas, spark dataframe, or a hive table, etc. or tracked transformation
  df_columns = df.columns #it would be nice to have metadata about the columns in the dataset
  df_source = source #the data this dataset was created from
  df_transformation = tran
  metadata = metadata
  
def track_transformation(source, target, trasnformation, custom_metadata):
  source = source #input tracked datasets or tracked transformations
  target = target #output tracked datasets or tracked transformations
  transformation = transformation #could be a SQL statement, or a python method, should be able to be passed pretty much anything


@cdsw.model_training
def development_pipeline(args):

  
  
  #Loading from csv
  X = pd.read_csv("X.csv")
  y = pd.read_csv("y.csv")

  #Tracking the original datasets
  cdsw.track_dataset(X)
  cdsw.track_dataset(y)

  #Splitting in training and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
  cdsw.track_transformation(source=[X,y], target=[X_train, X_test, y_train, y_test], "train_test_split", {"test_size":0.33, "random_state":0})

  #Now tracking training and test sets as part of Atlas flow
  cdsw.track_dataset(X_train, source=X, transformation="train_test_split", tag="training")
  cdsw.track_dataset(X_test, source=X, transformation="train_test_split", tag="test")
  cdsw.track_dataset(y_train, source=y, transformation="train_test_split", tag="training")
  cdsw.track_dataset(y_test, source=y, transformation="train_test_split", tag="test")

  #Creating Pipeline Object - sequence of Sklearn objects, 
  #The first (scaler) is a tranformation the second (logistic regression) is the classifier i.e. the actual ML model
  #However, the entire pipeline is exported and treated as the deployed model. Not just the actual ML model
  pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())])

  pipe.fit(X_train, y_train)
  pipe.score(X_test, y_test) #outputs a simple float 
  
  predictions = pipe.predict(X_test)
  
  cdsw.track_transformation(source=[X_train, y_train], 
                            target="LogisticRegression", 
                            transformation="StandardScaler",
                            custom_metadata=_
                           )
  
  cdsw.track_transformation(source="StandardScaler", 
                          target=predictions, 
                          transformation="LogisticRegression",
                          custom_metadata=pipe.score(X_test, y_test) #outputs a simple float 
                         )
  
  cdsw.track_dataset(predictions, source="LogisticRegression", transformation="LogisticRegression", tag="predictions")
  
  


