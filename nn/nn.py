# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike


# Neural Network Class Definition
class NeuralNetwork:
    """
    This is a neural network class that generates a fully connected Neural Network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            This list of dictionaries describes the fully connected layers of the artificial neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}] will generate a
            2 layer deep fully connected network with an input dimension of 64, a 32-dimension hidden layer
            and an 8-dimensional output.
        lr: float
            Learning Rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            This list of dictionaries describing the fully connected layers of the artificial neural network.
    """
    def __init__(self,
                 nn_arch, #List[Dict[str, Union(int, str)]],
                 lr: float,
                 seed: int,
                 batch_size: int,
                 epochs: int,
                 loss_function: str):
        # Saving architecture
        self.arch = nn_arch
        # Saving hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size
        # Initializing the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD!! IT IS ALREADY COMPLETE!!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """
        # seeding numpy random
        np.random.seed(self._seed)
        # defining parameter dictionary
        param_dict = {}
        # initializing all layers in the NN
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            # initializing weight matrices
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            # initializing bias matrices
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1
        return param_dict

    def _single_forward(self,
                        W_curr: ArrayLike,
                        b_curr: ArrayLike,
                        A_prev: ArrayLike,
                        activation: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        #based on how things are initialized, we actually need to transpose A_prev, not W_curr
        
        A_prevT = np.transpose(A_prev)
        
        Z_curr_noT = np.matmul(W_curr, A_prevT) + b_curr
        Z_curr = np.transpose(Z_curr_noT) #dimensions = n_observations x # output neurons
        
        if activation == "relu":
            A_curr = self._relu(Z_curr) #call relu activation function
        elif activation == "sigmoid":
            A_curr = self._sigmoid(Z_curr) #call sigmoid activation function
        
        return (A_curr, Z_curr)

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        cache = {} #cache to be filled later
        A_prev = X #assign inputs to the previous A matrix
        cache["A0"] = A_prev #add it to the cache as A0
        
        #probably want to create a for loop where you loop through each layer of the model and get Z and A matrices for each layer
        #length of nn_arch = # of layers
        for idx, layer in enumerate(self.arch):
            layer_idx = idx+1
            
            curr_activation = layer['activation'] #get activation function type for the current layer
            param_dict = self._param_dict #get current parameters
            
            W_curr = param_dict['W' + str(layer_idx)] #get current weights (which on first pass will be randomly initialized)
            b_curr = param_dict['b' + str(layer_idx)] #get current bias terms
            
            A_curr,Z_curr = self._single_forward(W_curr, b_curr, A_prev, curr_activation)
            
            cache["A"+str(layer_idx)] = A_curr #store current A matrix in cache dictionary
            cache["Z"+str(layer_idx)] = Z_curr #store current Z matrix in cache dictionary
            
            
            #update variables for next iteration
            A_prev = A_curr
            
        output = A_curr #output should be final A values of corresponding to output layer
        
        return output,cache

    def _single_backprop(self,
                         W_curr: ArrayLike,
                         b_curr: ArrayLike,
                         Z_curr: ArrayLike,
                         A_prev: ArrayLike,
                         dA_curr: ArrayLike,
                         activation_curr: str) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        m = A_prev.shape[0] #get the number of examples
        
        if activation_curr=="sigmoid":
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        elif activation_curr=="relu":
            dZ_curr = (self._relu_backprop(dA_curr, Z_curr)).astype(float)        
        
        dW_curr = np.dot(dZ_curr.T, A_prev)
        db_curr = np.transpose(np.sum(dZ_curr, axis=0, keepdims=True))
        dA_prev = np.dot(dZ_curr, W_curr)
        
        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values. #from the forward prop iteration
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        grad_dict = {} #initialize an empty grad_dict, to be filled with backprop gradients
        
        #get dA_prev
        if self._loss_func=="mean squared error":
            dA_prev = self._mean_squared_error_backprop(y, y_hat)
        if self._loss_func=="binary cross entropy":
            dA_prev = self._binary_cross_entropy_backprop(y, y_hat)
            
        #print(dA_prev)
        
        #reverse the nn architecture for backprop and iterate through each layer
        for last_layer,layer in reversed(list(enumerate(self.arch))):
            curr_layer = last_layer+1
            param_dict = self._param_dict
            
            dA_curr = dA_prev
            
            #get the inputs necessary for _single_backprop method
            curr_activation = layer["activation"] #activation function of the current layer
            A_prev = cache["A"+str(last_layer)] #inputs of the previous layer
            Z_curr = cache["Z"+str(curr_layer)] #transformed inputs of the current layer
            W_curr = param_dict["W"+str(curr_layer)] #weights of the current layer
            b_curr = param_dict["b"+str(curr_layer)] #bias terms of the current layer
            
            #single backprop through current layer
            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, curr_activation)
            
            #update grad_dict
            grad_dict['dW' + str(curr_layer)] = dW_curr
            grad_dict['db' + str(curr_layer)] = db_curr
        
        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and thus does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.

        Returns:
            None
        """
        # W = W - alpha*dW
        # b = b - alpha*db
        for idx, layer in enumerate(self.arch):
            
            layer_idx = idx+1
#             print("layer_idx: ", layer_idx)
#             print("layer: ", layer)
            
#             print('W' + str(layer_idx))
#             print(self._param_dict['W' + str(layer_idx)])
            
#             print('b' + str(layer_idx))
#             print(self._param_dict['b' + str(layer_idx)])
            
            # updating weight params
            self._param_dict['W' + str(layer_idx)] = self._param_dict['W' + str(layer_idx)] + (self._lr * grad_dict['dW' + str(layer_idx)])
            #updating bias params
            self._param_dict['b' + str(layer_idx)] = self._param_dict['b' + str(layer_idx)] + (self._lr * grad_dict['db' + str(layer_idx)])
            
        return None

    def fit(self,
            X_train: ArrayLike,
            y_train: ArrayLike,
            X_val: ArrayLike,
            y_val: ArrayLike) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network via training for the number of epochs defined at
        the initialization of this class instance.
        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        #initializing empty lists and starting values
        per_epoch_loss_train = []
        per_epoch_loss_val = []
        iteration = 1
        
        while iteration < self._epochs:
            #print(iteration)
            #!!don't forget to expand the dimensions of the y_vectors prior to inputting them into the fit function!!
            #get number of dimensions in the X_train and y_train datasets
            dim_X_train = X_train.shape[1]
            dim_y_train = y_train.shape[1]
            
            # Add y_values as the last column vector in X_train
            shuffle_arr = np.concatenate([X_train, y_train], axis=1)
            # In place shuffle
            np.random.shuffle(shuffle_arr)
            X_train = shuffle_arr[:, 0:dim_X_train] #separate out the inputs
            y_train = shuffle_arr[:, dim_X_train:dim_X_train+dim_y_train] #separate out the outputs
            #divide number of observations by the batch size to get number of batches
            num_batches = int(X_train.shape[0]/self._batch_size) + 1
            #split the X and y data into their respective number of batches
            X_batch = np.array_split(X_train, num_batches)
            y_batch = np.array_split(y_train, num_batches)
            
            #keep track of the per batch loss so you can take the mean at the end
            per_batch_train_loss = []
            per_batch_val_loss = []
            
            for X_train, y_train in zip(X_batch, y_batch):
                #forward pass through the network
                y_hat, cache = self.forward(X_train)
                #calculate loss
                if self._loss_func=="mean squared error":
                    loss = self._mean_squared_error(y_train, y_hat)
                elif self._loss_func=="binary cross entropy":
                    loss = self._binary_cross_entropy(y_train, y_hat)
                per_batch_train_loss.append(loss) #append the current training loss
                #per_epoch_loss_train.append(loss)
                
                #backpropagation pass through the network
                grad_dict = self.backprop(y_train, y_hat, cache)
                #update parameter values
                self._update_params(grad_dict) #update parameter values
                
                #validation pass
                y_pred, val_cache = self.forward(X_val) #make prediction based on current weights
                #calculate loss
                if self._loss_func=="mean squared error":
                    val_loss = self._mean_squared_error(y_val, y_pred)
                elif self._loss_func=="binary cross entropy":
                    val_loss = self._binary_cross_entropy(y_val, y_pred)
                per_batch_val_loss.append(val_loss) #append the current validation loss
                #per_epoch_loss_val.append(val_loss)
                    
            iteration+=1 #update iteration number
            per_epoch_loss_train.append(np.mean(per_batch_train_loss)) #append epoch's mean training loss
            per_epoch_loss_val.append(np.mean(per_batch_val_loss)) #append epoch's mean validation loss
            
        
        return per_epoch_loss_train, per_epoch_loss_val
        

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network model.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat, cache = self.forward(X) #run forward pass on the data to get outputs
        
        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = 1/(1+(1/np.exp(Z))) #sigmoid function, or 1/(1+e^-Z)
        return nl_transform

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = np.maximum(0,Z) #this mimics the ReLu function which is 0 at x<0 and linearly increasing at x>0
        return nl_transform

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        A = self._sigmoid(Z)
        dZ = np.multiply(A,(1-A)) #should this actually be np.multiply? or dot product?
        return dZ

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        return Z>0 #will return True if Z>0

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        #flatten input vectors, because they should be 1-dimensional
        y = y.flatten()
        y_hat = y_hat.flatten()
        m = y.shape[0]
        
        #add a really small number to each y_hat values so there are no issues with taking the log of zero
        y_hat = y_hat - 0.00000001
        #calculate loss
        loss = -(1/m)*np.sum(y.dot(np.log(y_hat)) + (1-y).dot(np.log(1-y_hat)))
        return loss

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        m = len(y)
        y=y.flatten()
        y_hat=y_hat.flatten()
        #add a really small number to y_hat values so there are no issues with dividing by zero
        y_hat = y_hat - 0.00000001
        return (1/m)*(np.divide(y,y_hat) + np.divide(1-y,1-y_hat)) #-np.mean((y/y_hat) + ((1-y)/(1-y_hat))) #removed negative

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        loss = np.mean(np.square(y-y_hat))
        return loss

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        m = len(y)
        dA = -2*(y-y_hat)*(1/m) #took away negative sign
        return dA

    def _loss_function(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Loss function, computes loss given y_hat and y. This function is
        here for the case where someone would want to write more loss
        functions than just binary cross entropy.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.
        Returns:
            loss: float
                Average loss of mini-batch.
        """
        
        pass

    def _loss_function_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        This function performs the derivative of the loss function with respect
        to the loss itself.
        Args:
            y (array-like): Ground truth output.
            y_hat (array-like): Predicted output.
        Returns:
            dA (array-like): partial derivative of loss with respect
                to A matrix.
        """
        pass
