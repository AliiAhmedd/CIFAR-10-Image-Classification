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

#STEP 3: MLP Class using Standard SGD optimizer and ADAM optimizer
class MLPClassifier:
    def __init__(self, hidden_layers=[512, 256, 128, 64], learning_rate=0.001, epochs=100, batch_size=128, 
                 hidden_activation='relu', loss_function='cross_entropy', optimizer='adam',
                 beta1=0.9, beta2=0.999, epsilon=1e-8, early_stopping=True, patience=10): #default network configuration ans general settings
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
        self.m1_weights = [] #first moment estimates for weights in ADAM | ADAM OPTIMIZER EXPLAINED IN REPORT
        self.m2_weights = [] #second moment estimates for weights in ADAM | ADAM OPTIMIZER EXPLAINED IN REPORT
        self.m1_biases = [] #first moment estimates for biases in ADAM | ADAM OPTIMIZER EXPLAINED IN REPORT
        self.m2_biases = [] #second moment estimates for biases in ADAM | ADAM OPTIMIZER EXPLAINED IN REPORT
        self.t = 0 #time step for ADAM optimizer
        self.summary = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []} #stores training performances

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
        elif activation == 'sigmoid':
            return self.sigmoid(x)
        elif activation == 'tanh':
            return self.tanh(x)
        else:
            raise ValueError(f"Unknown activation function: {activation}")
    
    def get_activation_derivative(self, x, activation):
        if activation == 'relu':
            return self.der_relu(x)
        elif activation == 'sigmoid':
            return self._sigmoid_derivative(x)
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
            
            # Initialize ADAM moment vectors with zeros
            if self.optimizer == 'adam':
                self.m1_weights.append(np.zeros_like(weight))
                self.m2_weights.append(np.zeros_like(weight))
                self.m1_biases.append(np.zeros_like(bias))
                self.m2_biases.append(np.zeros_like(bias))

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
    def mlp_training(self, X, y, validation_split=0.2, verbose=True): #verbose is used to monitor learning progress, it is set to false during hyperparameter tuning
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

                #STEP 3.1 Forward pass
                activations, pre_activ_values = self.forward_pass(batch_X)

                #STEP 3.2 Calculate loss and accuracy for this batch
                batch_loss = self.calc_loss(batch_y, activations[-1])
                batch_accuracy = self.calc_accuracy(batch_y, activations[-1])
                
                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy

                #STEP 3.3 Backpropagation
                weight_gradients = [] #list that stores weight gradients
                bias_gradients = [] #list that stores bias gradients
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
                    # loss gradient w.r.t biases: ∂L/∂b = (1/m) * Σ(δ), sum dZs across batch samples
                    db = np.sum(dZ, axis=0, keepdims=True) / m
                    # loss gradient w.r.t weights: ∂L/∂W = (1/m) * A^T · δ, where A is activation from previous layer
                    dW = np.dot(activations[i].T, dZ) / m
                    
                    weight_gradients.insert(0, dW)
                    bias_gradients.insert(0, db)
                    
                    if i > 0: #prevents trying to compute dZ for input layer since there are no weights before it measure how the weight changes as the loss gets minimized
                        # Propagate the error backwards: δ = (δ_next · W_next^T)
                        dZ = np.dot(dZ, self.weights[i].T) * self.get_activation_derivative(pre_activ_values[i-1], self.hidden_activation)

                # Update weights and biases using selected optimizer
                if self.optimizer == 'adam':
                    self.update_parameters_Adam(weight_gradients, bias_gradients)
                else:  # Default to SGD
                    self.update_parameters_Grad_Desc(weight_gradients, bias_gradients)

            #average training loss and accuracy values
            avg_loss = epoch_loss / batchAmount
            avg_accuracy = epoch_accuracy / batchAmount
            
            #Validation metrics
            validationActivations, _ = self.forward_pass(X_val) #pre_activation values will not be needed for validation
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
                    self.best_weights = [w.copy() for w in self.weights]
                    self.best_biases = [b.copy() for b in self.biases]
                    self.patience_counter = 0
                else:
                    #if no improvement: increment patience counter
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        self.final_epoch = epoch + 1
                        if verbose:
                            print(f"\nEarly stopping triggered at epoch {self.final_epoch}")
                            print(f"Best validation loss: {self.bestValLoss:.4f} at epoch {self.final_epoch - self.patience}")
                        #dtore best weights and biases
                        self.weights = self.best_weights
                        self.biases = self.best_biases
                        break
            
            if verbose and (epoch + 1) % 10 == 0: #print every 10 epochs but first makes sure that verbose is true, because if it isn't then this means we are in hyperparameter tuning mode
                optimizer = 'ADAM' if self.optimizer == 'adam' else 'SGD'
                #print temporal results for monitoring training progress
                print(f"Epoch {epoch + 1}/{self.epochs} [{optimizer}] - Loss: {avg_loss:.4f} - Accuracy: {avg_accuracy:.4f} - Val Loss: {validationLoss:.4f} - Val Accuracy: {validationAccuracy:.4f}") 
    
    def forward_pass(self, X):
        activations = [X]
        pre_activValues = []
        
        for i in range(len(self.weights)):
            pre_activ_value = np.dot(activations[i], self.weights[i]) + self.biases[i]
            pre_activValues.append(pre_activ_value) 
            
            if i != len(self.weights) - 1:
                A = self.get_activation(pre_activ_value, self.hidden_activation)  #hidden layers can use either relu or tanh
            else:
                A = self.softmax(pre_activ_value)  #output layer = softmax activation
            
            activations.append(A)
        
        return activations, pre_activValues #activations will be used in backpropagation to calculate the output layer error and pre_activation will be used to calculate derivatives of activation functions
    
    def update_parameters_Adam(self, weight_gradients, bias_gradients): #
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
    
    def update_parameters_Grad_Desc(self, weight_gradients, bias_gradients):
        for i in range(len(self.weights)): 
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]

    def evaluateBatch(self, X, y):
        predictions = self.predict(X)
        loss = self.calc_loss(y, predictions)
        accuracy = self.calc_accuracy(y, predictions)
        return loss, accuracy
    
    def predict(self, X):
        activations, _ = self.forward_pass(X) #just call forward pass since the network's parameters have now been fully trained and are ready to accept new unseen test data
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

#Optimal Configuration from Hyperparameter Tuning
mlp_model_adam = MLPClassifier(
    hidden_layers=[1536, 768, 384, 192],
    learning_rate=0.001,
    epochs=100,  # Max epochs, but early stopping may stop sooner
    batch_size=256,
    hidden_activation='relu',
    loss_function='cross_entropy',
    optimizer='adam',  # Using ADAM optimizer
    beta1=0.9, #0.9 is the standard value for beta1 because it provides a good balance between stability and responsiveness to recent gradients, this is equivalent to saying remember 90% of previous direction and 10% of current direction
    beta2=0.999, #0.999 is commonly used for beta2 as it effectively captures long-term trends in the squared gradients, helping to stabilize the learning process.
    epsilon=1e-8,
    early_stopping=True,  # Enable early stopping
    patience=20  #this variable is very useful for hyperparameter tuning but for the final implementation I would make my patience high so that the model can train for a long time and to ensure convergence
)
# Calculate approximate parameters
total_params_approx = (3072 * 1536) + (1536 * 768) + (768 * 384) + (384 * 192) + (192 * 5)
print(f"Approximate total parameters: {total_params_approx:,}")
print("\n" + "="*50)

print(f"Optimizer: ADAM (beta1={mlp_model_adam.beta1}, beta2={mlp_model_adam.beta2})")
print(f"Early Stopping: Enabled (patience={mlp_model_adam.patience})")

mlp_model_adam.mlp_training(
    x_trainScaled,  
    y_trainOHE, 
    validation_split=0.2, 
    verbose=True
)

#checks if loss is decreasing
recentLossValues = mlp_model_adam.summary['loss'][-5:]  # Last 5 epochs
loss_decreasing = all(recentLossValues[i] >= recentLossValues[i+1] for i in range(len(recentLossValues)-1))

#analysis for final loss values to, this helped in optimizing the epochs value as I train
print(f"\nLoss Analysis:")
print(f"Final Training Loss: {mlp_model_adam.summary['loss'][-1]:.4f}")
print(f"Loss trend in last 5 epochs: {'Decreasing' if loss_decreasing else 'Not consistently decreasing'}")


#STEP 4: Evaluate and predict on test set
print("\nEvaluating: ")
test_loss, test_accuracy = mlp_model_adam.evaluateBatch(x_testScaled, y_testOHE)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

#STEP 5: Interpretation and Visualisation of Results
#Plot training and validation loss and accuracy over epochs, confusion matrix and print out class-wise performance
plt.figure(figsize=(15, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(mlp_model_adam.summary['loss'], label='Training Loss (ADAM)', color='blue')
plt.plot(mlp_model_adam.summary['val_loss'], label='Validation Loss (ADAM)', color='red')
plt.title('Model Loss Over Time with ADAM Optimizer')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(mlp_model_adam.summary['accuracy'], label='Training Accuracy (ADAM)', color='blue')
plt.plot(mlp_model_adam.summary['val_accuracy'], label='Validation Accuracy (ADAM)', color='red')
plt.title('Model Accuracy Over Time with ADAM Optimizer')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Make predictions on test set for detailed analysis
y_pred = mlp_model_adam.predict(x_testScaled)
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
plt.title('Confusion Matrix (ADAM Optimizer)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(f"\nFINAL RESULTS WITH ADAM OPTIMIZER:")
print(f"Training Accuracy: {mlp_model_adam.summary['accuracy'][-1]:.4f}")
print(f"Validation Accuracy: {mlp_model_adam.summary['val_accuracy'][-1]:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

#HYPERPARAMETER TUNING CODE COMMENTED OUT
'''
print("HYPERPARAMETER TUNING IN PROGRESS:")
inputDimension = x_trainScaled.shape[1]  # 3072 features

layer_configs = [
    [2048, 1024, 512, 256],
    [1536, 768, 384, 192],
    [1536, 768, 384],
    [1024, 512, 256, 128],
    [1024, 512, 256],
    [512, 256, 128, 64]
]
activations = ['relu', 'tanh']
loss_functions = ['cross_entropy', 'mse'] #This is for the case where more than one loss derivatives have been calculated and can be used within the code which is not the case as when I did this before I realized that the model performed very poorly with MSE so I just removed the redundant code
# OPTIMIZED training hyperparameters
tuning_epochs = 100
batch_size = 256
learning_rate = 0.001
validation_split = 0.2

# Storage for results
results = [] #for final printing of all configurations
current_config = 0
bestConfig = None
total_configurations = len(layer_configs) * len(loss_functions) * len(activations)
bestAccuracy = 0.0 #stores best accuracy found so far

start_time = time.time()
for layer_idx, base_layers in enumerate(layer_configs):

    hidden_layers = [int(nodes) for nodes in base_layers]
    
    for loss_function in loss_functions:
        for activation in activations:
            current_config += 1
            
            print(f"\nConfiguration {current_config}/{total_configurations}")
            print(f"   Layers: {hidden_layers}")
            print(f"   Loss: {loss_function}")
            print(f"   Activation: {activation}")
            print(f"   Optimizer: ADAM")
            
            config_start_time = time.time()
            
            try:
                # Create and train model with ADAM optimizer
                mlp_model = MLPClassifier(
                    hidden_layers=hidden_layers,
                    learning_rate=learning_rate,
                    epochs=tuning_epochs,
                    batch_size=batch_size,
                    hidden_activation=activation,
                    loss_function=loss_function,
                    optimizer='adam'  # Using ADAM optimizer
                )
                
                # Train model
                mlp_model.mlp_training(
                    x_trainScaled, 
                    y_trainOHE, 
                    validation_split=validation_split,
                    verbose=False #verbose set to false here to not print epoch info during hyperparameter tuning
                )
                
                # Quick evaluation on smaller test subset for speed
                test_subset_size = 1000
                test_indices = np.random.choice(len(x_testScaled), test_subset_size, replace=False)
                x_test_subset = x_testScaled[test_indices]
                y_test_subset = y_testOHE[test_indices]
                
                test_loss, test_accuracy = mlp_model.evaluate(x_test_subset, y_test_subset)
                
                config_time = time.time() - config_start_time
                
                # Store results
                config_result = {
                    'layers': hidden_layers.copy(),
                    'num_layers': len(hidden_layers),
                    'total_params': sum(w.size for w in mlp_model.weights) + sum(b.size for b in mlp_model.biases),
                    'loss_function': loss_function,
                    'activation': activation,
                    'optimizer': 'adam',
                    'test_accuracy': test_accuracy,
                    'test_loss': test_loss,
                    'final_train_acc': mlp_model.summary['accuracy'][-1],
                    'final_val_acc': mlp_model.summary['val_accuracy'][-1],
                    'training_time': config_time
                }
                results.append(config_result)
                
                # Check if this is the best so far
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    bestConfig = config_result.copy()
                    print(f"NEW BEST! Test Accuracy: {test_accuracy:.4f} (in {config_time:.1f}s)")
                else:
                    print(f"Test Accuracy: {test_accuracy:.4f} (in {config_time:.1f}s)")
                
                # Progress update with time estimates
                elapsed = time.time() - start_time
                avg_time = elapsed / current_config
                remaining = avg_time * (total_configurations - current_config)
                print(f"Progress: {current_config/total_configurations*100:.1f}% | "
                        f"Elapsed: {elapsed:.0f}s | ETA: {remaining:.0f}s")
                
            except Exception as e:
                print(f"Error with configuration: {e}")
                continue

total_time = time.time() - start_time
print(f"\nHyperparameter tuning completed in {total_time/60:.1f} minutes.")

results.sort(key=lambda x: x['test_accuracy'], reverse=True) #sort results by test accuracy

print(f"Average time per config: {total_time/len(results) if results else 0:.1f} seconds")
print()

# Display all results SORTED by accuracy
print("ALL CONFIG RESULTS (sorted by accuracy):")
print("-" * 90)
for i, result in enumerate(results, 1):
    print(f"{i}. Test Accuracy: {result['test_accuracy']:.4f} | "
          f"Layers: {result['layers']} | "
          f"Activation: {result['activation']} | "
          f"Time: {result['training_time']:.1f}s")
    print(f"   Train: {result['final_train_acc']:.4f} | "
          f"Val: {result['final_val_acc']:.4f} | "
          f"Params: {result['total_params']:,}")
    print()


# Performance summary
avg_accuracy = np.mean([r['test_accuracy'] for r in results])
std_accuracy = np.std([r['test_accuracy'] for r in results])

if bestConfig:
    best_acc_str = f"{bestConfig['test_accuracy']:.4f} ({bestConfig['test_accuracy']*100:.2f}%)"
    best_layers_str = str(bestConfig['layers'])
else:
    best_acc_str = 'N/A'
    best_layers_str = 'None'

performance_summary = f"""
OPTIMIZED HYPERPARAMETER TUNING RESULTS WITH ADAM:
Best Configuration: {best_layers_str}
Best Test Accuracy: {best_acc_str}
Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}
Configurations Tested: {len(results)}
Total Time: {total_time:.1f} seconds ({total_time/60:.2f} minutes)
Optimizer: ADAM
"""
print(performance_summary)
'''