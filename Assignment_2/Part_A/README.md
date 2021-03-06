**1)	CNN_for_inatuaralist_dataset.ipynb:**

(Contains code for questions 1 and 2)

•	**_Downloading and unzipping section_**: Downloads the data from drive into the content folder of Colab and then properly distributes into training, validation (10% of training dataset) and test datasets. The RAM usually gets full and for training the model, runtime needs to be restarted (and downloading and unzipping section are skipped).  

•	**_Preprocessing section_**: Training, validation and test dataset images are normalized and resized to 224x224 images.

•	Classes of the subset of inaturalist dataset is defined.

•	**_Constructing CNN section_**: Contains ConvNet class which has following two functions:
-	__init__() which takes dropout probability, list of number of kernels in each layer, activation function, list of kernel size of each layer and number of neurons in dense layer as parameters and constructs the architecture of CNN based on these parameters.
-	forward() which applies the CNN on the given input.

• Contains train_model() function which trains the CNN model on the training data. It has the following parameters:
-	model – which takes the model architecture created from ConvNet class.
-	criteria – the loss function which is taken as cross entropy here.
-	optimizer – adam is used here
-	number of epochs
-	device

•	**_Defining data loaders and training model section_**: Train, validation and test data loaders are defined with suitable batch size and then training is done. The train_model() function is set such that it first trains the model and then evaluates based on validation dataset. So first convnet model is constructed, then criteria and optimizer are defined and the the train_model function is called with appropriate parameters for training the model on training data and evaluating on validation set.

•	**_Hyperparameter tuning section_**: Contains train() function for sweep. The hyperparameters considered for sweep are:
-	Number of filters
-	Dropout probability
-	Learning rates
-	Activation functions
-	Beta 1 values

**2)	Evaluating_CNN_on_test_data.ipynb:**

(Contains code for question 4)

•	**_Downloading and unzipping section_**: Downloads the data from drive into the content folder of Colab and then properly distributes into training, validation (10% of training dataset) and test datasets. The RAM usually gets full and for training the model, runtime needs to be restarted (and downloading and unzipping section are skipped).  

•	**_Preprocessing section_**: Training, validation and test dataset images are normalized and resized to 224x224 images.

•	Classes of the subset of inaturalist dataset is defined.

•	**_Constructing CNN section_**: Contains ConvNet class which has following two functions:
-	__init__() which takes dropout probability, list of number of kernels in each layer, activation function, list of kernel size of each layer and number of neurons in dense layer as parameters and constructs the architecture of CNN based on these parameters.
-	forward() which applies the CNN on the given input.

• Contains train_model() function which trains the CNN model on the training data. It has the following parameters:
-	model – which takes the model architecture created from ConvNet class.
-	criteria – the loss function which is taken as cross entropy here.
-	optimizer – adam is used here
-	number of epochs
-	device

•	**_Defining data loaders and training model section_**: Train, validation and test data loaders are defined with suitable batch size and then training is done. The train_model() function is set such that it first trains the model and then evaluates based on validation dataset. So first convnet model is constructed, then criteria and optimizer are defined and the the train_model function is called with appropriate parameters for training the model on training data and evaluating on validation set.

•	**_Evaluating best model section_**: Parameters corresponding best validation accuracy obtained in sweep are taken to train the (best) model and then test accuracy of that model is calculated by calling the evaluate() function.

•	**_Plotting grid for test data and its prediction section_**: Contains code for plotting 6x5 grid containing sample images from the test data and predictions made by the best model.

•	**_Visualizing all filters section_**: Contains code for visualising all the filters in the first layer of the best model for a random image from the test set. Creates a 8x8 grid as number of filters in first layer of best model is 64.

**3)	Guided_back_propogation.ipynb:**

(Contains code for question 5)

•	**_Downloading and unzipping section_**: Downloads the data from drive into the content folder of Colab and then properly distributes into training, validation (10% of training dataset) and test datasets. The RAM usually gets full and for training the model, runtime needs to be restarted (and downloading and unzipping section are skipped).  

•	**_Preprocessing section_**: Training, validation and test dataset images are normalized and resized to 224x224 images.

•	Classes of the subset of inaturalist dataset is defined.

•	**_Constructing CNN section_**: Contains ConvNet class which has following two functions:
-	__init__() which takes dropout probability, list of number of kernels in each layer, activation function, list of kernel size of each layer and number of neurons in dense layer as parameters and constructs the architecture of CNN based on these parameters.
-	forward() which applies the CNN on the given input.

• Contains train_model() function which trains the CNN model on the training data. It has the following parameters:
-	model – which takes the model architecture created from ConvNet class.
-	criteria – the loss function which is taken as cross entropy here.
-	optimizer – adam is used here
-	number of epochs
-	device

•	**_Defining data loaders and training model section_**: Train, validation and test data loaders are defined with suitable batch size and then training is done. The train_model() function is set such that it first trains the model and then evaluates based on validation dataset. So first convnet model is constructed, then criteria and optimizer are defined and the the train_model function is called with appropriate parameters for training the model on training data and evaluating on validation set.

•	**_Guided backpropogation section_**: Contains guidedBackprop class which has the following function:
- register_hooks() - for storing and manipulating gradients of intermediate nodes of the model.
- capture() - takes input and its labels and reconstructs the image from the model.

•	standardize() function - takes input as parameter and normalizes it.

•	After that, a few images are selected and back propogation is done and then standardize function is called with those images as inputs and then the grid is plotted.
