# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective of this experiment is to develop a neural network regression model using a dataset created in Google Sheets with one numeric input and one numeric output. Regression is a supervised learning technique used to predict continuous values. A neural network is chosen because it can effectively learn both linear and non-linear relationships between input and output by adjusting its weights during training.

The model is trained using backpropagation to minimize a loss function such as Mean Squared Error (MSE). During each iteration, the training loss is calculated and updated. The training loss vs iteration plot is used to visualize the learning process of the model, where a decreasing loss indicates that the neural network is learning properly and converging toward an optimal solution.

## Neural Network Model

<img width="1115" height="695" alt="image" src="https://github.com/user-attachments/assets/1a3163ce-4b0f-4fa5-9b13-61396699d3a9" />


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Dhanappriya
### Register Number: 212224230056
```python
#creating model class
# Name:DHANAPPRIYA S
# Register Number:212224230056
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here since it's a regression task
        return x

 

# Initialize the Model, Loss Function, and Optimizer
dhana = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(dhana.parameters(), lr=0.001)
#Function to train model
# Name:DHANAPPRIYA S
# Register Number:212224230056
def train_model(dhana, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(dhana(X_train), y_train)
        loss.backward()
        optimizer.step()

        dhana.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```
## Dataset Information
<img width="274" height="523" alt="image" src="https://github.com/user-attachments/assets/2dc3fa2d-c453-4512-8dc9-6dcbbfce96a7" />

## OUTPUT

<img width="856" height="514" alt="image" src="https://github.com/user-attachments/assets/e5f62159-54c8-415b-b146-2c93261c5992" />

### Training Loss Vs Iteration Plot

<<img width="836" height="763" alt="image" src="https://github.com/user-attachments/assets/ece5f266-9ce1-403f-93d2-3fb88c75a980" />



### New Sample Data Prediction

<img width="854" height="230" alt="image" src="https://github.com/user-attachments/assets/52fd0fca-86cc-4cc2-8619-b6d269c3568d" />

## RESULT

Thus the neural network regression model is developed using the given dataset.
