import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import StandardScaler
import pickle
from urllib.request import urlretrieve
import os
import tarfile
from sklearn.metrics import confusion_matrix
import time 
from itertools import product

#unpickle CIFAR-10 dataset
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#load cifar10
def load_cifar10():
    #check if CIFAR-10 dataset already downloaded if not then download it
    dataDirectory = 'cifar-10-batches-py'
    if not os.path.exists(dataDirectory):
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        filename = 'cifar-10-python.tar.gz'
        urlretrieve(url, filename)
        
        #extracting dataset
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall()
        os.remove(filename)
        print("CIFAR-10 dataset downloaded and extracted.")
    #load train data
    y_train = [] # collects all image data arrays
    x_train = [] #collects all training labels
    for i in range(1,6):
        batch = unpickle(f'{dataDirectory}/data_batch_{i}') #unpickles each batch file
        x_train.append(batch[b'data']) #add training data
        y_train.extend(batch[b'labels']) #add training labels

    x_train = np.concatenate(x_train)
    y_train = np.array(y_train)

    #load test data
    testBatch = unpickle(f'{dataDirectory}/test_batch')
    y_test = np.array(testBatch[b'labels'])
    x_test = testBatch[b'data']

    return (x_train, y_train), (x_test, y_test)

def filterSelectedClasses(x, y, selectedClasses):
    mask = np.isin(y.flatten(), selectedClasses) #creates a mask for selected classes to be kept
    yFilter = y[mask]
    xFilter = x[mask]
    #remapl labels in the 0-4 range for logical iteration
    labelMapping = {originalLabel: newLabel for newLabel, originalLabel in enumerate(selectedClasses)}
    yRemap = np.array([labelMapping[label] for label in yFilter])

    return xFilter, yRemap.reshape(-1, 1) #reshaping to a column vector so that it works with one-hot encoding later

#apply the filrer function so that all filtered xs and ys include the desired classes only
#ADD AT THE VERY BOTTOM



#one-hot encoding as MLPs require this format for cross-entropy loss calculation
'''[0, 0, 1, 0, 0]  # Class 2 each row represents a sample for this sample outpt would be class 2, one hot makes all other class probavbilities = 0
[1, 0, 0, 0, 0]  # Class 0  
[0, 1, 0, 0, 0]  # Class 1
[0, 0, 0, 0, 1]  # Class 4
[0, 0, 0, 1, 0]  # Class 3/ which is now truck class(9)'''
def one_hot_encoding(y, num_classes):
    oneHot = np.zeros((len(y), num_classes))
    for i, value in enumerate(y):
        oneHot[i, value] = 1
    return oneHot

# Data Augmentation Functions
def rand_horz_flip(image, p=0.5): #50% probability of flipping, flip operations require 2D spatial structure to operate properly
    if np.random.random() < p:
        img_r = image[:1024].reshape(32, 32)[:, ::-1].flatten()
        img_g = image[1024:2048].reshape(32, 32)[:, ::-1].flatten()
        img_b = image[2048:].reshape(32, 32)[:, ::-1].flatten()
        return np.concatenate([img_r, img_g, img_b])
    return image

def rand_rotation(image, max_angle=15): #rotation operations require 2D spatial structure to operate properly
    from scipy.ndimage import rotate
    angle = np.random.uniform(-max_angle, max_angle)
    #rotate each channel separately and flatten back
    img_r = rotate(image[:1024].reshape(32, 32), angle, reshape=False, mode='nearest').flatten()
    img_g = rotate(image[1024:2048].reshape(32, 32), angle, reshape=False, mode='nearest').flatten()
    img_b = rotate(image[2048:].reshape(32, 32), angle, reshape=False, mode='nearest').flatten()
    return np.concatenate([img_r, img_g, img_b])

def rand_crop_and_resize(image, crop_ratio=0.87): #crop and resize operations require 2D spatial structure to operate properly
    from scipy.ndimage import zoom
    width, height = 32, 32
    #crop sizes for width and height
    crop_w = int(width * crop_ratio)
    crop_h = int(height * crop_ratio)
    zoom_factor = (height/crop_h, width/crop_w) #zoom_factor defines how much to scale the cropped image back to original size
    #crop starting positions
    top = np.random.randint(0, height - crop_h + 1)
    left = np.random.randint(0, width - crop_w + 1)
    img_r = image[:1024].reshape(32, 32)[top:top+crop_h, left:left+crop_w]
    img_g = image[1024:2048].reshape(32, 32)[top:top+crop_h, left:left+crop_w]
    img_b = image[2048:].reshape(32, 32)[top:top+crop_h, left:left+crop_w]
    resized_r = zoom(img_r, zoom_factor, order=1).flatten()
    resized_g = zoom(img_g, zoom_factor, order=1).flatten()
    resized_b = zoom(img_b, zoom_factor, order=1).flatten()
    return np.concatenate([resized_r, resized_g, resized_b])

def rand_brightness(image, delta=0.2): #delta defines the range of brightness adjustment
    factor = 1 + np.random.uniform(-delta, delta)
    return np.clip(image * factor, 0, 1)

def rand_contrast(image, delta=0.3): #delta defines the range of contrast adjustment
    mean = np.mean(image)
    factor = 1 + np.random.uniform(-delta, delta) 
    return np.clip((image - mean) * factor + mean, 0, 1) #image - mean calculates deviation from mean brightness of image, then * factor scales this deviation + mean shifts it back to original brightness range

def gaussian_noise(image, std=0.03): #adds gaussian noise
    noise = np.random.normal(0, std, image.shape) #noise array with shape of input image flattened gets applied to each pixel in the flattened image
    return np.clip(image + noise, 0, 1) #clips values to [0, 1] range to avoid overflow

def rand_hue(image, max_shift=0.1): #max shift defines the range of hue adjustment, the value is kept at 0.1 to ensure realistic color variations so that they aren't too obscure
    shift_value = np.random.uniform(-max_shift, max_shift)
    # Extract RGB channels
    img_r = image[:1024]
    img_g = image[1024:2048]
    img_b = image[2048:]
    #hue shift is applied by mixing channels
    shifted_r = np.clip(img_r + shift_value * (img_g - img_r), 0, 1)
    shifted_g = np.clip(img_g + shift_value * (img_b - img_g), 0, 1)
    shifted_b = np.clip(img_b + shift_value * (img_r - img_b), 0, 1)
    return np.concatenate([shifted_r, shifted_g, shifted_b])

def augment_image(image, augmentation_probability=0.5):
    augmented_image = image.copy()
    if np.random.random() < augmentation_probability:
        augmented_image = rand_horz_flip(augmented_image, p=0.5)
    if np.random.random() < augmentation_probability:
        augmented_image = rand_rotation(augmented_image, max_angle=15)
    if np.random.random() < augmentation_probability:
        augmented_image = rand_crop_and_resize(augmented_image, crop_ratio=0.87)
    if np.random.random() < augmentation_probability:
        augmented_image = rand_contrast(augmented_image, delta=0.3)
    if np.random.random() < augmentation_probability:
        augmented_image = rand_brightness(augmented_image, delta=0.2)
    if np.random.random() < augmentation_probability:
        augmented_image = rand_hue(augmented_image, max_shift=0.1)
    if np.random.random() < augmentation_probability:
        augmented_image = gaussian_noise(augmented_image, std=0.03)
    return augmented_image

def augment_batch(X_batch, y_batch, augment_probability=0.8):
    augmentedX = []
    for i in range(len(X_batch)):
        if np.random.random() > augment_probability:
            augmentedX.append(X_batch[i])
        else:
            augmented_image = augment_image(X_batch[i], augmentation_probability=0.5)
            augmentedX.append(augmented_image)
    return np.array(augmentedX), y_batch

#STEP 3: MLP Class using Standard SGD optimizer and ADAM optimizer with Batch Normalization and Dropout
class MLPClassifier:
    def __init__(self, hidden_layers=[512, 256, 128, 64], learning_rate=0.001, epochs=100, batch_size=128, 
                 hidden_activation='relu', loss_function='cross_entropy', optimizer='adam',
                 beta1=0.9, beta2=0.999, epsilon=1e-8, early_stopping=True, patience=10, 
                 weight_decay=0.0, dropout_rate=0.0, use_batch_norm=False): #default network configuration ans general settings
        self.weights = []
        self.biases = []
        self.epochs = epochs #amount of training iterations
        self.batch_size = batch_size #amount of samples per gradient update
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.hidden_activation = hidden_activation #'relu', 'tanh'
        self.loss_function = loss_function #'cross_entropy', 'mse'
        self.optimizer = optimizer #'standard gradient descent' or 'adam'
        self.beta1 = beta1 #exponential decay rates for normal gradient estimates in ADAM optimizer
        self.beta2 = beta2 #exponential decay rates for squared gradients in ADAM optimizer
        self.epsilon = epsilon #to prevent division by zero in ADAM optimizer
        self.early_stopping = early_stopping #whether to use early stopping or not
        self.bestValLoss = float('inf') #tracks best validation loss
        self.bestWeights = None #stores best weights
        self.bestBiases = None #stores best biases
        self.finalEpoch = 0 #stores final epoch number after early stopping to report when training stopped
        self.patience = patience #used for early stopping
        self.patience_counter = 0 #tracks number of epochs without improvement
        self.weight_decay = weight_decay #L2 regularization strength
        self.dropout_rate = dropout_rate #dropout probability
        self.use_batch_norm = use_batch_norm #whether to use batch normalization
        self.m1_weights = [] #first moment estimates for weights in ADAM | ADAM OPTIMIZER EXPLAINED IN REPORT
        self.m2_weights = [] #second moment estimates for weights in ADAM | ADAM OPTIMIZER EXPLAINED IN REPORT
        self.m1_biases = [] #first moment estimates for biases in ADAM | ADAM OPTIMIZER EXPLAINED IN REPORT
        self.m2_biases = [] #second moment estimates for biases in ADAM | ADAM OPTIMIZER EXPLAINED IN REPORT
        self.t = 0 #time step for ADAM optimizer
        self.summary = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []} #stores training performances
        
        # Batch Normalization parameters
        self.bn_gamma = [] #learnable scale parameters
        self.bn_beta = [] #learnable shift parameters
        self.bn_movingAvgMean = [] #running mean used during testing and computed during training
        self.bn_movingAvgVar = [] #running variance used during testing and computed during training
        self.bn_momentum = 0.9 #momentum for moving averages
        self.m1_bn_gamma = [] #first moment estimates for BN gamma in ADAM
        self.m2_bn_gamma = [] #second moment estimates for BN gamma in ADAM
        self.m1_bn_beta = [] #first moment estimates for BN beta in ADAM
        self.m2_bn_beta = [] #second moment estimates for BN beta in ADAM

    #relevant activation functions with their derivatives
    def relu(self, x):
        return np.maximum(0, x)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
    def der_relu(self, x):
        return np.where(x > 0, 1, 0)
    
    def der_tanh(self, x):
        return 1 - np.tanh(x)**2
    
    def der_softmax(self, x):
        s = self.softmax(x)
        return s * (1 - s)
    
    def get_activation(self, x, activation):
        if activation == 'relu':
            return self.relu(x)
        elif activation == 'tanh':
            return self.tanh(x)
        else:
            raise ValueError(f"Unknown activation function: {activation}")
    
    def get_activation_derivative(self, x, activation):
        if activation == 'relu':
            return self.der_relu(x)
        elif activation == 'tanh':
            return self.der_tanh(x)
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
    def weight_initilization(self, input_neurons, output_neurons): #initializes weights and biases for INPUT LAYER
        layer_neurons = [input_neurons] + self.hidden_layers + [output_neurons] #amount of neurons in each layer

        for i in range(len(layer_neurons) - 1): #iterates through each layer to initialize its weights and biases
            weight = np.random.randn(layer_neurons[i], layer_neurons[i+1]) * np.sqrt(2.0 / layer_neurons[i]) #He weight initialization as it is suitable for ReLU activation
            bias = np.zeros((1, layer_neurons[i+1])) #Initializing biases to 0
            self.weights.append(weight)
            self.biases.append(bias)
            
            # Initialize Batch Normalization parameters (only for hidden layers)
            if self.use_batch_norm and i < len(layer_neurons) - 2:
                self.bn_gamma.append(np.ones((1, layer_neurons[i+1]))) #scale parameter initialized to 1
                self.bn_beta.append(np.zeros((1, layer_neurons[i+1]))) #shift parameter initialized to 0
                self.bn_movingAvgMean.append(np.zeros((1, layer_neurons[i+1]))) #running mean initialized to 0
                self.bn_movingAvgVar.append(np.ones((1, layer_neurons[i+1]))) #running variance initialized to 1
            
            # Initialize ADAM moment vectors with zeros
            if self.optimizer == 'adam':
                self.m1_weights.append(np.zeros_like(weight))
                self.m2_weights.append(np.zeros_like(weight))
                self.m1_biases.append(np.zeros_like(bias))
                self.m2_biases.append(np.zeros_like(bias))
                
                # Initialize ADAM moments for BN parameters
                if self.use_batch_norm and i < len(layer_neurons) - 2:
                    self.m1_bn_gamma.append(np.zeros((1, layer_neurons[i+1]))) #first moment estimates for BN gamma in ADAM
                    self.m2_bn_gamma.append(np.zeros((1, layer_neurons[i+1]))) #second moment estimates for BN gamma in ADAM
                    self.m1_bn_beta.append(np.zeros((1, layer_neurons[i+1]))) #first moment estimates for BN beta in ADAM
                    self.m2_bn_beta.append(np.zeros((1, layer_neurons[i+1]))) #second moment estimates for BN beta in ADAM

    def calc_loss(self, y_true, y_pred):
        if self.loss_function == 'cross_entropy':
            return self.cross_entropyLoss(y_true, y_pred)
        else:
            return self.mseLoss(y_true, y_pred)

    def cross_entropyLoss(self, y_true, y_pred):
        epsilon = 1e-8 #small epsilon to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon) #clips predictions to avoid log(0)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1)) #cross-entropy loss calculation for every sample in the current batch hence the mean
        return loss

    def mseLoss(self, y_true, y_pred): #usually not optimal for multi-class classification tasks but will still be used for comparisons
        loss = np.mean(np.square(y_true - y_pred)) 
        return loss
    
    def calc_accuracy(self, y_true, y_pred):
        pred = np.argmax(y_pred, axis=1)
        trueClasses = np.argmax(y_true, axis=1)
        accuracy = np.mean(pred == trueClasses)
        return accuracy
    
    #Training
    def mlp_training(self, X, y, validation_split=0.2, verbose=True, use_augmentation=True): #verbose is used to monitor learning progress, it is set to false during hyperparameter tuning
        split_idx = int(len(X) * (1 - validation_split)) #split index specifies index where data is split into training and validation sets
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]   

        batchAmount = len(X_train) // self.batch_size #number of batches per epoch

        self.weight_initilization(X.shape[1], y.shape[1]) #initialize weights and biases

        for epoch in range(self.epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            
            #to avoid first-class-seen bias, shuffling training data will be necessary, TALK ABOUT THE IMPORTANCE OF THIS IN REPORT
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for batch_idx in range(batchAmount):
                #create batches with size = self.batch_size for training
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size
                batch_X = X_train_shuffled[start_idx:end_idx]
                batch_y = y_train_shuffled[start_idx:end_idx]
                
                # Apply data augmentation
                if use_augmentation:
                    batch_X, batch_y = augment_batch(batch_X, batch_y, augment_probability=0.8)

                #STEP 3.1 Forward pass
                activations, pre_activ_values, dropout_masks, batchNormStorage = self.forward_pass(batch_X, training=True) #

                #STEP 3.2 Calculate loss and accuracy for this batch
                batch_loss = self.calc_loss(batch_y, activations[-1])
                
                # Add L2 regularization penalty
                l2_penalty = 0 #Computes the sum of all squared weights across all layers, adds this penalty to the loss function, scaled by weight_decay / 2 finally During backpropagation, this causes gradients to include an extra term:
                for weight in self.weights:
                    l2_penalty += np.sum(weight ** 2) #sum of squared weights across all layers
                batch_loss += (self.weight_decay / 2) * l2_penalty #L2 regularization term added to loss
                
                batch_accuracy = self.calc_accuracy(batch_y, activations[-1])
                
                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy

                #STEP 3.3 Backpropagation
                weight_gradients = [] #list that stores weight gradients
                bias_gradients = [] #list that stores bias gradients
                bn_gamma_gradients = [] #list that stores batch norm gamma gradients
                bn_beta_gradients = [] #list that stores batch norm beta gradients
                m = batch_X.shape[0] #number of samples in batch
                
                # Output layer error
                # for mse_loss testing change the EQUATION BELOW
                dZ = activations[-1] - batch_y #derivative of cross-entropy loss w.r.t pre-activation output
                # Quick overview showing Mathematical proof, more on the report itself (z = pre-activation output):
                # Cross-entropy: L = -Σ(y_true * log(y_pred))
                # Softmax: y_pred = exp(z_i) / Σ(exp(z_j))
                # Using chain rule: ∂L/∂z_k = Σ_i (∂L/∂y_i * ∂y_i/∂z_k)
                # Softmax derivative is a Jacobian: ∂y_i/∂z_k = y_i(δ_ik - y_k)
                # Result (after cancellation): ∂L/∂z = y_pred - y_true == activations[-1] - y_true

                #start backprop
                for i in range(len(self.weights) - 1, -1, -1):
                    # loss gradient w.r.t weights: ∂L/∂W = (1/m) * A^T · δ + weight_decay * W (L2 regularization)
                    dW = np.dot(activations[i].T, dZ) / m + self.weight_decay * self.weights[i]
                    # loss gradient w.r.t biases: ∂L/∂b = (1/m) * Σ(δ), sum dZs across batch samples
                    db = np.sum(dZ, axis=0, keepdims=True) / m
                    
                    weight_gradients.insert(0, dW)
                    bias_gradients.insert(0, db)
                    
                    if i > 0: #prevents trying to compute dZ for input layer since there are no weights before it
                        dZ = np.dot(dZ, self.weights[i].T) * self.get_activation_derivative(pre_activ_values[i-1], self.hidden_activation)
                        
                        # Backprop through dropout
                        if dropout_masks[i-1] is not None: #used for hidden layers only
                            dZ = dZ * dropout_masks[i-1] / (1 - self.dropout_rate) #modifies the loss derivative with respect to preactivation hence needs modification using dropout mask
                        
                        # Backprop through Batch Normalization
                        if self.use_batch_norm and batchNormStorage[i-1] is not None: #if batch norm is used and this layer has it
                            Z, Z_norm, batch_mean, batch_var = batchNormStorage[i-1]
                            
                            dgamma = np.sum(dZ * Z_norm, axis=0, keepdims=True)
                            dbeta = np.sum(dZ, axis=0, keepdims=True)
                            bn_gamma_gradients.insert(0, dgamma) #insert BN gamma gradient at the beginning of the list
                            bn_beta_gradients.insert(0, dbeta) #insert BN beta gradient at the beginning of the list
                            
                            dZ_norm = dZ * self.bn_gamma[i-1]
                            dvar = np.sum(dZ_norm * (Z - batch_mean) * -0.5 * np.power(batch_var + 1e-8, -1.5), axis=0, keepdims=True)
                            dmean = np.sum(dZ_norm * -1.0 / np.sqrt(batch_var + 1e-8), axis=0, keepdims=True) + dvar * np.sum(-2.0 * (Z - batch_mean), axis=0, keepdims=True) / m
                            dZ = dZ_norm / np.sqrt(batch_var + 1e-8) + dvar * 2.0 * (Z - batch_mean) / m + dmean / m
                        elif self.use_batch_norm: #when backpropagating through output layer batch norm isn't used
                            bn_gamma_gradients.insert(0, None) 
                            bn_beta_gradients.insert(0, None)

                # Update weights and biases using selected optimizer
                if self.optimizer == 'adam':
                    self.update_parameters_adam(weight_gradients, bias_gradients, bn_gamma_gradients, bn_beta_gradients)
                else:  # Default to SGD
                    self.update_parameters_sgd(weight_gradients, bias_gradients, bn_gamma_gradients, bn_beta_gradients)

            #average training loss and accuracy values
            avg_loss = epoch_loss / batchAmount
            avg_accuracy = epoch_accuracy / batchAmount
            
            #Validation metrics
            validationActivations, _, _, _ = self.forward_pass(X_val, training=False)
            validationLoss = self.calc_loss(y_val, validationActivations[-1])
            validationAccuracy = self.calc_accuracy(y_val, validationActivations[-1])
            
            #provides summary for performance tracking
            self.summary['loss'].append(avg_loss)
            self.summary['accuracy'].append(avg_accuracy)
            self.summary['val_loss'].append(validationLoss)
            self.summary['val_accuracy'].append(validationAccuracy)
            
            # Early stopping check
            if self.early_stopping: #is early stopping enabled?
                if validationLoss < self.bestValLoss:
                    #save best weights and reset patience
                    self.bestValLoss = validationLoss
                    self.bestWeights = [w.copy() for w in self.weights]
                    self.bestBiases = [b.copy() for b in self.biases]
                    self.patience_counter = 0
                else:
                    #if no improvement: increment patience counter
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        self.finalEpoch = epoch + 1
                        if verbose:
                            print(f"\nEarly stopping triggered at epoch {self.finalEpoch}")
                            print(f"Best validation loss: {self.bestValLoss:.4f} at epoch {self.finalEpoch - self.patience}")
                        #store best weights and biases
                        self.weights = self.bestWeights
                        self.biases = self.bestBiases
                        break
            
            if verbose and (epoch + 1) % 5 == 0: #print every 5 epochs for deep model monitoring every 5 epochs here because it takes longer to run so I needed more frequent updates to ensure my model was running properly
                optimizer = 'ADAM' if self.optimizer == 'adam' else 'SGD'
                #print temporal results for monitoring training progress
                print(f"Epoch {epoch + 1}/{self.epochs} [{optimizer}] - Loss: {avg_loss:.4f} - Accuracy: {avg_accuracy:.4f} - Val Loss: {validationLoss:.4f} - Val Accuracy: {validationAccuracy:.4f}") 
    
    def forward_pass(self, X, training=False):
        activations = [X]
        pre_activValues = [] #stores pre-activation values for each layer
        dropout_masks = [] #bool array with same shape as activations to track which neurons were dropped out
        batchNormStorage = [] #stores batch normalization's scale and shift parameters (gamma and beta) for backpropagation
        
        for i in range(len(self.weights)):
            Z = np.dot(activations[i], self.weights[i]) + self.biases[i]
            
            if i != len(self.weights) - 1:
                # Apply Batch Normalization before activation (only for hidden layers)
                if self.use_batch_norm:
                    if training:
                        # Training mode: compute batch statistics
                        batch_mean = np.mean(Z, axis=0, keepdims=True) #mean across batch for each feature
                        batch_var = np.var(Z, axis=0, keepdims=True) #variance across batch for each feature
                        Z_norm = (Z - batch_mean) / np.sqrt(batch_var + 1e-8) #normalize the activations
                        Z_bn = self.bn_gamma[i] * Z_norm + self.bn_beta[i] #scale and shift
                        
                        # Update running statistics for inference
                        self.bn_movingAvgMean[i] = self.bn_momentum * self.bn_movingAvgMean[i] + (1 - self.bn_momentum) * batch_mean #the running mean defines a moving average of batch means
                        self.bn_movingAvgVar[i] = self.bn_momentum * self.bn_movingAvgVar[i] + (1 - self.bn_momentum) * batch_var #the running variance defines a moving average of batch variances
                        
                        
                        batchNormStorage.append((Z, Z_norm, batch_mean, batch_var)) #append to the batch norm storage all necessary values for backpropagation
                        Z = Z_bn
                    else:
                        # Inference mode: use running statistics
                        Z_norm = (Z - self.bn_movingAvgMean[i]) / np.sqrt(self.bn_movingAvgVar[i] + 1e-8) #normalize already computed values for moving avg mean and variance
                        Z = self.bn_gamma[i] * Z_norm + self.bn_beta[i] #update the pre-activation values using scale and shift parameters
                        batchNormStorage.append(None)
                else:
                    batchNormStorage.append(None) #no batch norm used when disabled
                
                pre_activValues.append(Z)
                A = self.get_activation(Z, self.hidden_activation)
                
                # Apply dropout
                if training and self.dropout_rate > 0:
                    dropout_mask = (np.random.rand(*A.shape) > self.dropout_rate).astype(float) #dropout mask is assigned by randomly generating values between 0 and 1 and if they are larger than the dropout rate then we set them to 1 else 0 and finally convert them to bools, if false then drop
                    dropout_masks.append(dropout_mask)
                    A = A * dropout_mask / (1 - self.dropout_rate) #scales activations to maintain expected value
                else:
                    dropout_masks.append(None) #no dropout mask needed during inference or if dropout rate is 0
            else:
                pre_activValues.append(Z)
                A = self.softmax(Z)
                dropout_masks.append(None)
                batchNormStorage.append(None)
            
            activations.append(A)
        
        return activations, pre_activValues, dropout_masks, batchNormStorage
    
    def update_parameters_adam(self, weight_gradients, bias_gradients, bn_gamma_gradients=None, bn_beta_gradients=None):
        self.t += 1  # Increment timestep
        
        for i in range(len(self.weights)):
            # Update biased first moment estimate (momentum), beta1 is the weighted% of previous gradients and 1-beta1 weighted% of current gradient
            self.m1_weights[i] = self.beta1 * self.m1_weights[i] + (1 - self.beta1) * weight_gradients[i]
            self.m1_biases[i] = self.beta1 * self.m1_biases[i] + (1 - self.beta1) * bias_gradients[i]
            
            # Update biased second moment estimate, remember beta2% of previous squared gradients and 1-beta2% of current squared gradient beta2 is always larger because it captures long-term trends
            self.m2_weights[i] = self.beta2 * self.m2_weights[i] + (1 - self.beta2) * (weight_gradients[i] ** 2)
            self.m2_biases[i] = self.beta2 * self.m2_biases[i] + (1 - self.beta2) * (bias_gradients[i] ** 2)
            
            #Problem: At the start (timestep t=1), momentum and variance estimates are biased toward zero (since they initialized at zero)            
            #Solution: Divide by (1 - beta^t) to correct this as at the start the denominator is small and increases over time reducing the bias
            m_bias_corrected_weight = self.m1_weights[i] / (1 - self.beta1 ** self.t)
            m_bias_corrected_bias = self.m1_biases[i] / (1 - self.beta1 ** self.t)
            
            #Compute bias-corrected second moment estimate
            v_bias_corrected_weight = self.m2_weights[i] / (1 - self.beta2 ** self.t)
            v_bias_corrected_bias = self.m2_biases[i] / (1 - self.beta2 ** self.t)
            
            #Update parameters
            self.weights[i] -= self.learning_rate * m_bias_corrected_weight / (np.sqrt(v_bias_corrected_weight) + self.epsilon)
            self.biases[i] -= self.learning_rate * m_bias_corrected_bias / (np.sqrt(v_bias_corrected_bias) + self.epsilon)
        
        # Update Batch Normalization parameters
        if self.use_batch_norm and bn_gamma_gradients is not None: #if batch norm is used and gradients are initialized which happens when backpropagating through hidden layers
            for i in range(len(self.bn_gamma)):
                if bn_gamma_gradients[i] is not None:
                    self.m1_bn_gamma[i] = self.beta1 * self.m1_bn_gamma[i] + (1 - self.beta1) * bn_gamma_gradients[i]
                    self.m2_bn_gamma[i] = self.beta2 * self.m2_bn_gamma[i] + (1 - self.beta2) * (bn_gamma_gradients[i] ** 2)
                    m_gamma_corrected = self.m1_bn_gamma[i] / (1 - self.beta1 ** self.t) #similar logic as weight decays for bias-corrected estimates 
                    v_gamma_corrected = self.m2_bn_gamma[i] / (1 - self.beta2 ** self.t)
                    self.bn_gamma[i] -= self.learning_rate * m_gamma_corrected / (np.sqrt(v_gamma_corrected) + self.epsilon)
                    
                    self.m1_bn_beta[i] = self.beta1 * self.m1_bn_beta[i] + (1 - self.beta1) * bn_beta_gradients[i]
                    self.m2_bn_beta[i] = self.beta2 * self.m2_bn_beta[i] + (1 - self.beta2) * (bn_beta_gradients[i] ** 2)
                    m_beta_corrected = self.m1_bn_beta[i] / (1 - self.beta1 ** self.t)
                    v_beta_corrected = self.m2_bn_beta[i] / (1 - self.beta2 ** self.t)
                    self.bn_beta[i] -= self.learning_rate * m_beta_corrected / (np.sqrt(v_beta_corrected) + self.epsilon)
    
    def update_parameters_sgd(self, weight_gradients, bias_gradients, bn_gamma_gradients=None, bn_beta_gradients=None):
        for i in range(len(self.weights)): 
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
        
        if self.use_batch_norm and bn_gamma_gradients is not None:
            for i in range(len(self.bn_gamma)): #runs for each hidden layer with batch norm
                if bn_gamma_gradients[i] is not None:
                    self.bn_gamma[i] -= self.learning_rate * bn_gamma_gradients[i] #update BN gamma
                    self.bn_beta[i] -= self.learning_rate * bn_beta_gradients[i] #update BN beta

    def evaluate(self, X, y):
        predictions = self.predict(X)
        loss = self.calc_loss(y, predictions)
        accuracy = self.calc_accuracy(y, predictions)
        return loss, accuracy
    
    def predict(self, X):
        activations, _, _, _ = self.forward_pass(X, training=False) #just call forward pass since the network's parameters have now been fully trained and are ready to accept new unseen test data
        return activations[-1] #ret output layer activations which are essentially the predicted class probabilities

np.random.seed(40) #NEVER CHANGE!!
#STEP 1: Data extraction and filteration
(x_train, y_train), (x_test, y_test) = load_cifar10()
selectedClasses = [0, 1, 8, 9, 5]  # airplane, automobile, ship, truck, dog
selectedClassNames = ['airplane', 'automobile', 'ship', 'truck', 'dog']

x_trainFiltered, y_trainFiltered = filterSelectedClasses(x_train, y_train, selectedClasses)
x_testFiltered, y_testFiltered = filterSelectedClasses(x_test, y_test, selectedClasses)

#STEP 2: Data Preprocessing (Normalization, One-Hot Encoding and standardizing) 
x_trainNorm = x_trainFiltered.astype('float32') / 255.0
x_testNorm = x_testFiltered.astype('float32') / 255.0  

scaler = StandardScaler()
y_trainOHE = one_hot_encoding(y_trainFiltered, len(selectedClasses))
y_testOHE = one_hot_encoding(y_testFiltered, len(selectedClasses))

x_testScaled = scaler.fit_transform(x_testNorm)
x_trainScaled = scaler.fit_transform(x_trainNorm)  

#Deep Model Configuration with Batch Normalization, Dropout, and Data Augmentation
deep_model = MLPClassifier(
    hidden_layers=[1024, 1024, 512, 512, 256, 256, 128, 128],  # 8 hidden layers for 10 total layers
    learning_rate=0.001,
    epochs=160,
    batch_size=256,
    hidden_activation='relu',
    loss_function='cross_entropy',
    optimizer='adam',
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    early_stopping=True,
    patience=100,  # High patience to allow model to converge with augmentation fluctuations
    weight_decay=0.0,  # L2 regularization (set to 0.0 as per notebook)
    dropout_rate=0.1,  # 10% dropout rate
    use_batch_norm=True  # Enable batch normalization
)

# Calculate approximate parameters
total_params_approx = (3072 * 1024) + (1024 * 1024) + (1024 * 512) + (512 * 512) + (512 * 256) + (256 * 256) + (256 * 128) + (128 * 128) + (128 * 5)
print(f"Approximate total parameters: {total_params_approx:,}")
print(f"Training data shape: {x_trainScaled.shape}")
print(f"Training labels shape: {y_trainOHE.shape}")
print(f"Model Architecture: {deep_model.hidden_layers}")
print(f"Optimizer: ADAM (beta1={deep_model.beta1}, beta2={deep_model.beta2})")
print(f"Early Stopping: Enabled (patience={deep_model.patience})")
print(f"Dropout Rate: {deep_model.dropout_rate}")
print(f"Batch Normalization: {'Enabled' if deep_model.use_batch_norm else 'Disabled'}")
print(f"Data Augmentation: Enabled")
print("\n" + "="*50)

# Start training with data augmentation
deep_model.mlp_training(
    x_trainScaled,  
    y_trainOHE, 
    validation_split=0.2, 
    verbose=True,
    use_augmentation=True
)

print("\n" + "="*50)
print("Training completed!")

#checks if loss is decreasing
recentLossValues = deep_model.summary['loss'][-5:]  # Last 5 epochs
loss_decreasing = all(recentLossValues[i] >= recentLossValues[i+1] for i in range(len(recentLossValues)-1))

#analysis for final loss values
print(f"\nLoss Analysis:")
print(f"Final Training Loss: {deep_model.summary['loss'][-1]:.4f}")
print(f"Loss trend in last 5 epochs: {'Decreasing' if loss_decreasing else 'Not consistently decreasing'}")


#STEP 4: Evaluate on test set
print("\nEvaluating: ")
test_loss, test_accuracy = deep_model.evaluate(x_testScaled, y_testOHE)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

#STEP 5: Interpretation and Visualisation of Results
#Plot training and validation loss and accuracy over epochs, confusion matrix and print out class-wise performance
plt.figure(figsize=(15, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(deep_model.summary['loss'], label='Training Loss', color='blue')
plt.plot(deep_model.summary['val_loss'], label='Validation Loss', color='red')
plt.title('Deep Model Loss Over Time (ADAM + BN + Dropout + Augmentation)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(deep_model.summary['accuracy'], label='Training Accuracy', color='blue')
plt.plot(deep_model.summary['val_accuracy'], label='Validation Accuracy', color='red')
plt.title('Deep Model Accuracy Over Time (ADAM + BN + Dropout + Augmentation)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Make predictions on test set for detailed analysis
y_pred = deep_model.predict(x_testScaled)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_testOHE, axis=1)

#class-wise accuracy
print("\nClass-wise Performance:")
for i in range(5):
    class_mask = y_true_classes == i
    class_accuracy = np.mean(y_pred_classes[class_mask] == y_true_classes[class_mask])
    print(f"Class {i} ({selectedClassNames[i]}): {class_accuracy:.4f}")

# Confusion matrix visualization
cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(8, 6))
sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=selectedClassNames, 
            yticklabels=selectedClassNames)
plt.title('Confusion Matrix - Deep Model with Batch Norm & Dropout')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(f"\nFINAL RESULTS WITH DEEP MODEL (ADAM + BN + DROPOUT + AUGMENTATION):")
print(f"Training Accuracy: {deep_model.summary['accuracy'][-1]:.4f}")
print(f"Validation Accuracy: {deep_model.summary['val_accuracy'][-1]:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

#HYPERPARAMETER TUNING CODE COMMENTED OUT
'''
layer_config = [1024, 1024, 512, 512, 256, 256, 128, 128] #for hyperparameter tuning in the deep model I avoided including layers and instead came up with this layer configuration by performing research since the tuning would take a very long time if it were to be done with many layer configs
loss_function = 'cross_entropy'
activation = 'relu' #same as layer configs

# Training parameters
tuning_epochs = 30
batch_size = 256
learning_rate = 0.001
validation_split = 0.2

# Regularization combinations to test - ALL POSSIBLE COMBINATIONS
regularization_configs = [
    {
        'name': 'Dropout (0.1) + Data Augmentation',
        'dropout_rate': 0.1,
        'weight_decay': 0.0,
        'use_augmentation': True,
        'use_batch_norm': False
    },
    {
        'name': 'Dropout (0.1) + L2 Regularization',
        'dropout_rate': 0.1,
        'weight_decay': 0.0001,
        'use_augmentation': False,
        'use_batch_norm': False
    },
    {
        'name': 'Data Augmentation + L2 Regularization',
        'dropout_rate': 0.0,
        'weight_decay': 0.0001,
        'use_augmentation': True,
        'use_batch_norm': False
    },
    {
        'name': 'Dropout (0.1) + Data Augmentation + L2',
        'dropout_rate': 0.1,
        'weight_decay': 0.0001,
        'use_augmentation': True,
        'use_batch_norm': False
    },
    {
        'name': 'Dropout (0.15) + Data Augmentation',
        'dropout_rate': 0.15,
        'weight_decay': 0.0,
        'use_augmentation': True,
        'use_batch_norm': False
    },
    
    {
        'name': 'Batch Norm + Dropout (0.1)',
        'dropout_rate': 0.1,
        'weight_decay': 0.0,
        'use_augmentation': False,
        'use_batch_norm': True
    },
    {
        'name': 'Batch Norm + Data Augmentation',
        'dropout_rate': 0.0,
        'weight_decay': 0.0,
        'use_augmentation': True,
        'use_batch_norm': True
    },
    {
        'name': 'Batch Norm + L2 Regularization',
        'dropout_rate': 0.0,
        'weight_decay': 0.0001,
        'use_augmentation': False,
        'use_batch_norm': True
    },
    
    {
        'name': 'Batch Norm + Dropout (0.1) + Data Augmentation',
        'dropout_rate': 0.1,
        'weight_decay': 0.0,
        'use_augmentation': True,
        'use_batch_norm': True
    },
    {
        'name': 'Batch Norm + Dropout (0.1) + L2',
        'dropout_rate': 0.1,
        'weight_decay': 0.0001,
        'use_augmentation': False,
        'use_batch_norm': True
    },
    {
        'name': 'Batch Norm + Data Augmentation + L2',
        'dropout_rate': 0.0,
        'weight_decay': 0.0001,
        'use_augmentation': True,
        'use_batch_norm': True
    },
    
    {
        'name': 'Batch Norm + Dropout (0.1) + Data Augmentation + L2 (ALL)',
        'dropout_rate': 0.1,
        'weight_decay': 0.0001,
        'use_augmentation': True,
        'use_batch_norm': True
    },
    
    {
        'name': 'Batch Norm + Dropout (0.15) + Data Augmentation + L2 (ALL)',
        'dropout_rate': 0.15,
        'weight_decay': 0.0001,
        'use_augmentation': True,
        'use_batch_norm': True
    }
]

# Storage for results
comparison_results = []
best_accuracy = 0.0
best_config = None

start_time = time.time()

# Loop through each regularization combination
for config_idx, reg_config in enumerate(regularization_configs, 1):
    print(f"\n{'='*70}")
    print(f"Configuration {config_idx}/{len(regularization_configs)}: {reg_config['name']}")
    print(f"{'='*70}")
    print(f"   Architecture: {layer_config}")
    print(f"   Loss Function: {loss_function}")
    print(f"   Activation: {activation}")
    print(f"   Dropout Rate: {reg_config['dropout_rate']}")
    print(f"   Weight Decay (L2): {reg_config['weight_decay']}")
    print(f"   Data Augmentation: {'Yes' if reg_config['use_augmentation'] else 'No'}")
    print(f"   Batch Normalization: {'Yes' if reg_config['use_batch_norm'] else 'No'}")
    print(f"   Optimizer: ADAM")
    
    config_start_time = time.time()
    
    try:
        # Create model with current regularization configuration
        mlp_model = MLPClassifier(
            hidden_layers=layer_config,
            learning_rate=learning_rate,
            epochs=tuning_epochs,
            batch_size=batch_size,
            hidden_activation=activation,
            loss_function=loss_function,
            optimizer='adam',
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            early_stopping=True,
            patience=15,  # Increased from 10 to allow more tolerance for augmentation fluctuations
            weight_decay=reg_config['weight_decay'],
            dropout_rate=reg_config['dropout_rate'],
            use_batch_norm=reg_config['use_batch_norm']
        )
        
        # Train with appropriate augmentation setting
        if reg_config['use_augmentation']:
            print("\nTraining with data augmentation enabled...")
        else:
            print("\nTraining without data augmentation...")
        
        mlp_model.mlp_training(
            x_trainScaled, 
            y_trainOHE, 
            validation_split=validation_split,
            verbose=True,  # Show progress every 10 epochs
            use_augmentation=reg_config['use_augmentation']
        )
        
        # Evaluate on full test set
        test_loss, test_accuracy = mlp_model.evaluate(x_testScaled, y_testOHE)
        
        config_time = time.time() - config_start_time
        
        # Store results
        config_result = {
            'name': reg_config['name'],
            'dropout_rate': reg_config['dropout_rate'],
            'weight_decay': reg_config['weight_decay'],
            'use_augmentation': reg_config['use_augmentation'],
            'use_batch_norm': reg_config['use_batch_norm'],
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'final_train_acc': mlp_model.summary['accuracy'][-1],
            'final_val_acc': mlp_model.summary['val_accuracy'][-1],
            'final_train_loss': mlp_model.summary['loss'][-1],
            'final_val_loss': mlp_model.summary['val_loss'][-1],
            'total_params': sum(w.size for w in mlp_model.weights) + sum(b.size for b in mlp_model.biases),
            'training_time': config_time,
            'epochs_trained': len(mlp_model.summary['loss'])
        }
        comparison_results.append(config_result)
        
        print(f"\nConfiguration completed in {config_time:.1f}s")
        print(f"   Final Test Accuracy: {test_accuracy:.4f}")
        print(f"   Final Test Loss: {test_loss:.4f}")
        print(f"   Epochs Trained: {config_result['epochs_trained']}")
        
        # Check if this is the best
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_config = config_result.copy()
            print(f"NEW BEST CONFIGURATION!")
        
    except Exception as e:
        print(f"Error with configuration: {e}")
        import traceback
        traceback.print_exc()
        continue

total_time = time.time() - start_time

print(f"\n{'='*70}")
print("REGULARIZATION COMPARISON COMPLETED!")
print(f"{'='*70}")
print(f"Total Time: {total_time:.1f} seconds ({total_time/60:.2f} minutes)")
print()

# Display results table
print("RESULTS SUMMARY:")
print("-" * 110)
print(f"{'Configuration':<40} {'Test Acc':<12} {'Val Acc':<12} {'Train Acc':<12} {'Epochs':<10} {'Time (s)':<10}")
print("-" * 110)
for result in comparison_results:
    print(f"{result['name']:<40} {result['test_accuracy']:<12.4f} {result['final_val_acc']:<12.4f} "
          f"{result['final_train_acc']:<12.4f} {result['epochs_trained']:<10} {result['training_time']:<10.1f}")
print("-" * 110)

if best_config:
    print(f"\nBEST CONFIGURATION:")
    print(f"   Name: {best_config['name']}")
    print(f"   Test Accuracy: {best_config['test_accuracy']:.4f} ({best_config['test_accuracy']*100:.2f}%)")
    print(f"   Validation Accuracy: {best_config['final_val_acc']:.4f}")
    print(f"   Training Accuracy: {best_config['final_train_acc']:.4f}")
    print(f"   Test Loss: {best_config['test_loss']:.4f}")
    print(f"   Dropout Rate: {best_config['dropout_rate']}")
    print(f"   Weight Decay (L2): {best_config['weight_decay']}")
    print(f"   Data Augmentation: {'Yes' if best_config['use_augmentation'] else 'No'}")
    print(f"   Batch Normalization: {'Yes' if best_config['use_batch_norm'] else 'No'}")
    print(f"   Training Time: {best_config['training_time']:.1f}s")
    print(f"   Epochs Trained: {best_config['epochs_trained']}")
'''