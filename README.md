# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? 
I choose the resnet18 model since it is a pre-trained model and I can add some layers onto it to improve the model for my classification.

Give an overview of the types of parameters and their ranges used for the hyperparameter search
I choose to tune the hyper-paramter: learning rate and batch-size. The learning rate is between 0.001 to 0.1 and it is an continuous variable. And the batch-size have the values between 75,100,125.

Remember that your README should:
- Include a screenshot of completed training jobs
![avatar](/picture1.jpg)


- Logs metrics during the training process
![avatar](/picture2.jpg)

- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs
![avatar](/picture3.jpg)


## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker
There seems nothing anomalous with the debugging output. However, the loss for the testing job is larger than the training job. And it might be because there are not much training data which would cause overfitting.

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?
There seems nothing problematic with the model.

**TODO** Remember to provide the profiler html/pdf file in your submission.
![avatar](/picture4.jpg)


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
After deploying the model, I extract the test set from S3. And then I randomly pick up one picture and do some data preprocessing, then I do the prediction through the deployment.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.
![avatar](/picture5.jpg)






Some references:
https://github.com/watson28/cnn-dog-project/blob/master/dog_app.ipynb

http://machinememos.com/python/keras/artificial%20intelligence/machine%20learning/transfer%20learning/dog%20breed/neural%20networks/convolutional%20neural%20network/tensorflow/image%20classification/imagenet/2017/07/11/dog-breed-image-classification.html

https://medium.com/@kenneth.ca95/a-guide-to-transfer-learning-with-keras-using-resnet50-a81a4a28084b

https://github.com/yaircarmon/semisup-adv/blob/master/generate_pseudolabels.py

https://stackoverflow.com/questions/36205481/read-file-content-from-s3-bucket-with-boto3

https://github.com/htinaunglu/DogBreed-Classification-With-Amazon-Sagemaker/blob/main/train_model.py
