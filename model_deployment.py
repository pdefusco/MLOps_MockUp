import sklearn
import numpy as np
from joblib import load
import cdsw

pipeline = load('model_ready.joblib')

print(pipeline.named_steps)

scaler = pipeline.named_steps['scaler']
clf = pipeline.named_steps['lr']

@cdsw.model_metrics
def predict(args):
  
  data = pd.DataFrame(args, index=[0])
  prediction = pipeline.predict(data)
  probability = clf.predict_proba(data)
  
  #Track individual inputs -- Already available
  cdsw.track_metric('input_data', data)
  
  # Track probability -- Already available
  cdsw.track_metric('probability', int(probability[0][0]))
  
  # Track prediction -- Already available
  cdsw.track_metric('prediction', int(prediction[0]))
  
  
  return {'prediction':prediction, 
         'probability': probability}