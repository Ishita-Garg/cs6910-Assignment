**Pretrained_CNN_models.ipynb:**

(Contains code for questions 1 to 3)

•	**_Downloading and unzipping section:_** Downloads the data from drive into the content folder of Colab and then properly distributes into training, validation (10% of training dataset) and test datasets. The RAM usually gets full and for training the model, runtime needs to be restarted (and downloading and unzipping section are skipped).  

•	**_Preprocessing section:_** Training, validation and test dataset images are normalized and resized to 224x224 images.

•	Classes of the subset of inaturalist dataset is defined.

•	**_Multiple Pre-Trained models section:_** Contains model defining functions for multiple pre-trained models addressing the facts that size of the images is not same as ImageNet data and that output has only 10 classes. The pre-trained models considered here are:
-	Resnet50
-	VGG16
-	Alexnet
-	Squeezenet
-	Densenet

•	**_Defining training function and data loaders section:_** Contains train_model() function which trains the model on the inaturalist training data. It has the following parameters:
-	model – from the options mentioned above.
-	criteria – the loss function which is taken as cross entropy here.
-	optimizer – adam is used here
-	number of epochs
-	device – cuda

•	**_Training section:_** The models are training using the train_model function. The train_model() function is set such that it first trains the model and then evaluates based on validation dataset. So first any pre-trained model is obtained compatible with inaturalist dataset, then criteria and optimizer are defined and the the train_model function is called with appropriate parameters for training the model on training data and evaluating on the validation set.

•	**_Hyperparameter tuning section:_** Contains hyperparameter dictionary and train() function for sweep. The hyperparameters considered for sweep are:
-	Model – resnet, alexnet, vgg, densenet, squeezenet
-	Learning rates
-	Freeze_percent – Percent of neurons to be freezed
-	Beta 1 values
-	Batch size -  of train loader

•	**_Evaluation on test data section_**: Testing accuracy of model on the test by calling the evaluate() function.
