
##Sagemaker iniitiation package 
import sagemaker
import boto3
import os
import numpy as np
from sagemaker.predictor import csv_serializer
from sagemaker.amazon.amazon_estimator import get_image_uri 
from sagemaker.session import s3_input, Session

## relevant model and metric packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import datetime as dt
import pickle as pkl
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


##Bucket initiation and  location tag commands 

##1
bucket = 'bucketname/'
prefix = 'foldername/'

data = pd.read_csv('s3://' + bucket + prefix + 'data.csv')

##2
bucket_name = 'healthlakepit'
my_region = boto3.session.Session().region_name 
print(my_region)


s3 = boto3.resource('s3')
try:
    if  my_region == 'us-east-1':
        s3.create_bucket(Bucket=bucket_name)      
        print('S3 bucket created successfully')
except Exception as e:
    print('S3 error: ',e)

##3
prefix = 'xgboost-Algo'
output_path ='s3://{}/{}/output'.format(bucket_name, prefix)
print(output_path)\


## Test Split 
train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.8 * len(model_data))])
print(train_data.shape, test_data.shape)

##Concart_1

##A very helpful code I found, to move your OUTPUT_LABEL to the first column of your dataset is this:
cols = list(data)
cols.insert(0, cols.pop(cols.index('OUTPUT_LABEL')))
data= data.loc[:, cols]


## Concart_2
pd.concat([train_data['target'], train_data.drop(['id'], axis=1)],axis=1).to_csv('train.csv', index=False, header=False)

pd.concat([test_data['target'], test_data.drop(['id'], axis=1)], axis=1).to_csv('test.csv', index=False, header=False)


boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')

boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'test/test.csv')).upload_file('test.csv')


s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')
s3_input_test = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/test'.format(bucket_name, prefix), content_type='csv')


## Algo- container call in sagemaker 

container = get_image_uri(boto3.Session().region_name,'xgboost',repo_version='1.0-1')

hyperparameters = {
        "max_depth":"2",
        "eta":"0.2",
        "gamma":"4",
        "min_child_weight":"6",
        "subsample":"0.8",
        "objective":"binary:logistic",
        "num_round":100 ,
        "eval_metric":'auc'
        }

# construct a SageMaker estimator that calls the xgboost-container
estimator = sagemaker.estimator.Estimator(image_uri=container, 
                                          hyperparameters=hyperparameters,
                                          role=sagemaker.get_execution_role(),
                                          train_instance_count=1, 
                                          train_instance_type='ml.m4.xlarge', 
                                          train_volume_size=5, # 5 GB 
                                          output_path=output_path,
                                          train_use_spot_instances=True,
                                          train_max_run=300,
                                          train_max_wait=600)


## Sagemaker Fit fucntion call 

estimator.fit({'train': s3_input_train,'validation': s3_input_test})

##Deploy Function

xgb_predictor = estimator.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')


## Array for Prediction array 
test_data_array = test_data.drop(['id'], axis=1).values                #load the data into an array
#x gb_predictor.content_type = 'text/csv'                                           # set the data type for an inference

xgb_predictor.serializer = csv_serializer                                         # set the serializer type
predictions = xgb_predictor.predict(test_data_array).decode('utf-8')              # predict!
predictions_array = np.fromstring(predictions[1:], sep=',')                       # and turn the prediction into an array
print(predictions_array.shape)

##Print Prediction_array
predictions_array

## confusion matrix
cm = pd.crosstab(index=test_data['target'], columns=np.round(predictions_array), rownames=['Observed'], colnames=['Predicted'])
tn = cm.iloc[0,0]; fn = cm.iloc[1,0]; tp = cm.iloc[1,1]; fp = cm.iloc[0,1]; p = (tp+tn)/(tp+tn+fp+fn)*100
print("\n{0:<20}{1:<4.1f}%\n".format("The Classification Rate: ", p))
print("{0:<15}{1:<15}{2:>8}".format("Predicted", "without Disease", "With Disease"))
print("Observed")
print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("without Disease", tn/(tn+fn)*100,tn, fp/(tp+fp)*100, fp))
print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("With Disease", fn/(tn+fn)*100,fn, tp/(tp+fp)*100, tp))

##Sagemaker Endpoint Deleting 
sagemaker.Session().delete_endpoint(xgb_predictor.endpoint)
bucket_to_delete.objects.all().delete()
bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)