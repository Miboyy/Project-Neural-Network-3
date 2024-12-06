import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error

# Load dataset
try:
    df = pd.read_csv('Testing_salary.csv', encoding='latin-1')
except FileNotFoundError:
    print("Error: Testing_salary.csv not found.")
    exit()

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Apply Label Encoding to categorical columns
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Separate features and target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

# Feature Scaling (Normalization)
sc_X = MinMaxScaler()
X = sc_X.fit_transform(X)
sc_y = MinMaxScaler()
y = sc_y.fit_transform(y)

# Neural Network Architecture (using numpy)
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.layers = []
        prev_size = input_size
        for size in hidden_sizes:
            self.layers.append({
                'weights': np.random.rand(prev_size, size) - 0.5, 
                'bias': np.random.rand(1, size) - 0.5
            })
            prev_size = size
        self.layers.append({
            'weights': np.random.rand(prev_size, output_size) - 0.5, 
            'bias': np.random.rand(1, output_size) - 0.5
        })

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        activations = [inputs]
        for layer in self.layers:
            z = np.dot(activations[-1], layer['weights']) + layer['bias']
            a = self.sigmoid(z)
            activations.append(a)
        return activations

    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            activations = self.forward(inputs)
            output = activations[-1]

            error = targets - output
            cost = np.mean(np.square(error))
            
            # Hanya print setiap 100 epoch untuk mengurangi output
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Cost: {cost}")

            delta = error * self.sigmoid_derivative(output)
            for i in reversed(range(len(self.layers))):
                layer = self.layers[i]
                if i > 0:
                    prev_activations = activations[i]
                else:
                    prev_activations = inputs
                
                layer['weights'] += learning_rate * np.dot(prev_activations.T, delta)
                layer['bias'] += learning_rate * np.sum(delta, axis=0, keepdims=True)
                
                if i > 0:
                    delta = np.dot(delta, layer['weights'].T) * self.sigmoid_derivative(activations[i])

    def predict(self, inputs):
        activations = self.forward(inputs)
        return activations[-1]

# Initialize and Train the Neural Network
input_size = X.shape[1]
hidden_sizes = [10, 5]
output_size = 1
nn = NeuralNetwork(input_size, hidden_sizes, output_size)

# Train the network
nn.train(X, y, epochs=2000, learning_rate=0.001)

# Predict (on the same dataset for demonstration)
predictions = nn.predict(X)

# Inverse transform the scaled values
predictions = sc_y.inverse_transform(predictions)
y_original = sc_y.inverse_transform(y)

# Print results
print("\nPredictions:", predictions.flatten())
print("\nActual Values:", y_original.flatten())

# Calculate Mean Squared Error
mse = mean_squared_error(y_original, predictions)
print(f"\nMean Squared Error: {mse}")