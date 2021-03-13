**1)	Neural_Network_for_Fashion_MNIST.ipynb :**

(Contains code for questions 1 to 7)

    •	Fashion MNIST dataset is downloaded and images are plotted for each class and logged in wandb.
    
    •	Contains a class NeuralNet whose object can be created by passing these parameters :-
        -	Size_of_Input (int): which is the dimension of one input i.e. the number of neurons in input layer (784 in Fashion MNIST dataset)
        -	Number_of_Neuron_each_layer (list): contains the number of neurons in each layer of the network – hidden and output layers.
        -	Number_of_Layers (int): equal to the count of layers in the network (L = (L-1) hidden layers + 1 output layer)
        -	Activation_function (string): name of activation function – sigmoid, tanh, ReLU
        -	typeOfInit (string): weight initializer – random or xavier
        -	L2reg_const (int): weight decay with default value set to 0
        
    •	All the above variables and weights and bias are initialized when the object of this class is created.
    
    •	The class contains the following functions :-
        -	forward_propogation() with one argument – Input
        -	back_propogation() with arguments – pre-activation, activation, output from forward_propogation, true label (Y) and input.
        -	optimize() for implementing the gradient descent variants – sgd, momentum, nag, rmsprop, adam and nadam.
        -	val_loss_and_accuracy() with argument – validation data and labels.
        -	test() with argument – test data and labels – for calculating test accuracy.
        
    •	The model can be trained by creating an object of the NeuralNet class using the above mentioned arguments and then calling the optimize function with the following parameters :-
        -	X (array) : train input data (train images data)
        -	Y (array) : train output data (train labels)
        -	val_images (array) : validation input data
        -	val_labels (array) : validation labels
        -	optimizer (string) : sgd, momentum, nag, rmsprop, adam and nadam.
        -	learning_rate (float) : learning rate for the algorithms
        -	max_epochs (int)
        -	batch_size (int) 
        
    •	The trained model can be tested by calling test() function with test data and label as arguments and it returns the test accuracy.
    
    •	In the hyperparameter tuning section :
        -	sweep configuration is defined with various choices of the hyperparameters.
        -	train function is defined for sweeping
        -	sweep_id is set using wandb.sweep function
        -	sweep is run for 50 times using random search as there are a lot of hyperparameter combinations.
        
    •	At the end of notebook confusion matrix is plotted (logged in wandb) for the best identified hyperparameter combination.
    
    
**2)	Comparing_cross_entropy_and_squared_error_loss.ipynb :**

(Contains code for question 8)

    •	Contains a class NeuralNet whose object can be created by passing these parameters :-
        -	Size_of_Input (int): which is the dimension of one input i.e. the number of neurons in input layer (784 in Fashion MNIST dataset)
        -	Number_of_Neuron_each_layer (list): contains the number of neurons in each layer of the network – hidden and output layers.
        -	Number_of_Layers (int): equal to the count of layers in the network (L = (L-1) hidden layers + 1 output layer)
        -	Activation_function (string): name of activation function – sigmoid, tanh, ReLU
        -	typeOfInit (string): weight initializer – random or xavier
        -	L2reg_const (int): weight decay with default value set to 0
        
    •	All the above variables and weights and bias are initialized when the object of this class is created.
    
    •	The class contains the following functions :-
        -	forward_propogation() with one argument – Input
        -	back_propogation() with arguments – pre-activation, activation, output from forward_propogation, true label (Y), input and loss function.
        -	optimize() for implementing the gradient descent variants – sgd, momentum, nag, rmsprop, adam and nadam.
        -	val_loss_and_accuracy() with argument – validation data, validation labels and loss function.
        -	test() with argument – test data, test labels and loss function – for calculating test accuracy.
        
    •	The model can be trained by creating an object of the NeuralNet class using the above mentioned arguments and then calling the optimize function with the following parameters :-
        -	X (array) : train input data (train images data)
        -	Y (array) : train output data (train labels)
        -	val_images (array) : validation input data
        -	val_labels (array) : validation labels
        -	optimizer (string) : sgd, momentum, nag, rmsprop, adam and nadam.
        -	learning_rate (float) : learning rate for the algorithms
        -	max_epochs (int)
        -	batch_size (int) 
        -	loss_type (string) : squared_error or cross_entropy
        
    •	The trained model can be tested by calling test() function with test data and label as arguments and it returns the test accuracy.
    
    •	In the hyperparameter tuning section :
        -	sweep configuration is defined with various choices of the hyperparameters including the option of loss function.
        -	train function is defined for sweeping
        -	sweep_id is set using wandb.sweep function
        -	sweep is run using grid search as the number of hyperparameter combinations are reduced using inferences drawn from sweep in the questions before.


**3)	Neural_Network_for_MNIST_dataset.ipynb :**

(Contains code for question 10)

    •	MNIST dataset is downloaded and data is processed.
    
    •	Contains a class NeuralNet whose object can be created by passing these parameters :-
        -	Size_of_Input (int): which is the dimension of one input i.e. the number of neurons in input layer (784 in Fashion MNIST dataset)
        -	Number_of_Neuron_each_layer (list): contains the number of neurons in each layer of the network – hidden and output layers.
        -	Number_of_Layers (int): equal to the count of layers in the network (L = (L-1) hidden layers + 1 output layer)
        -	Activation_function (string): name of activation function – sigmoid, tanh, ReLU
        -	typeOfInit (string): weight initializer – random or xavier
        -	L2reg_const (int): weight decay with default value set to 0
        
    •	All the above variables and weights and bias are initialized when the object of this class is created.
    
    •	The class contains the following functions :-
        -	forward_propogation() with one argument – Input
        -	back_propogation() with arguments – pre-activation, activation, output from forward_propogation, true label (Y) and input.
        -	optimize() for implementing the gradient descent variants – sgd, momentum, nag, rmsprop, adam and nadam.
        -	val_loss_and_accuracy() with argument – validation data and labels.
        -	test() with argument – test data and labels – for calculating test accuracy.

    •	The model can be trained by creating an object of the NeuralNet class using the above mentioned arguments and then calling the optimize function with the following parameters :-
        -	X (array) : train input data (train images data)
        -	Y (array) : train output data (train labels)
        -	val_images (array) : validation input data
        -	val_labels (array) : validation labels
        -	optimizer (string) : sgd, momentum, nag, rmsprop, adam and nadam.
        -	learning_rate (float) : learning rate for the algorithms
        -	max_epochs (int)
        -	batch_size (int) 
        
    •	The trained model can be tested by calling test() function with test data and label as arguments and it returns the test accuracy.
    
    •	The model is trained and tested for hyperparameter combinations corresponding to the top 3 accuracies obtained in case of Fashion MNIST dataset.
