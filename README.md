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
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search
* I use pretrained incepton v3 model. This is a classical model used for image classification. 
* I fine tune "lr": ContinuousParameter(0.001, 0.1), "batch-size": CategoricalParameter([32, 64, 128, 256, 512]), "epochs": IntegerParameter(2,15), "momentum": ContinuousParameter(0.001, 0.9999). 
* I Use hyperband search to find the best hyperparameter.

![alt text](image/hyperparameter_tune_job.png?raw=true)

![alt text](image/train_with_debugger.png?raw=true)

Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker
* initiate hook in the main function. 
* Add hook in train and test function. 
* In the notebook, add rules and configurations to estimator.

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.
The GPU is well used, which is good. The loss reduces significant in the beginning, but not moving much latter. We can reduce learning rate and try more epoches, or use a scheduler to change learning rate dynamically. 

![alt text](image/Screenshot 2023-01-22 at 12.33.48 PM.png?raw=true)

## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.
* write an inference.py file as the entry point of the deployment. 
* overwrite predictor call back class. (overwrite serialization and deserialization)
* Setup PyTorchModel parameters, including moedel data dir, framework version, python version, entry point file, and preditor callback function. 
* deploy with instace type and ContentType
* load the image to create the payload, then predict and get prediction result

![alt text](image/endpoint.png?raw=true)

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
