import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Load dataset
data_path = os.path.join(os.getcwd(), "All_Max_Scour_Slab.txt")
all_samp = pd.read_csv(data_path, delimiter='\t')

# Separating the points into train and test sets based on the tags
test_tags = ['S28', 'S22', 'S42', 'S14', 'S33']
test_set = all_samp[all_samp['Tag'].isin(test_tags)].reset_index(drop=True).drop('Tag', axis=1)
train_set = all_samp[~all_samp['Tag'].isin(test_tags)].reset_index(drop=True).drop('Tag', axis=1)

# Standardizing the variables
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = train_set.iloc[:, 0:4].values
y_train = train_set['S(m)'].values
X_test = test_set.iloc[:, 0:4].values
y_test = test_set['S(m)'].values

X_scaled_train = torch.tensor(scaler_X.fit_transform(X_train), dtype=torch.float32)
y_scaled_train = torch.tensor(scaler_y.fit_transform(y_train.reshape(-1, 1)), dtype=torch.float32)

# Keep the original inputs in a tensor (before scaling)
X_original_train = torch.tensor(X_train, dtype=torch.float32)

# Create a DataLoader instance
dataset = TensorDataset(X_scaled_train, y_scaled_train, X_original_train)
loader = DataLoader(dataset, batch_size=600, shuffle=True)

# PyTorch Scaler implementation
class PyTorchScaler(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean, dtype=torch.float32), requires_grad=False)
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        return (x - self.mean) / self.scale

    def inverse_transform(self, x):
        return x * self.scale + self.mean

# Model definition
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(X_scaled_train.shape[1], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Custom loss function to include physical constraint
def custom_loss(outputs, targets, original_inputs, scaler):
    outputs_rescaled = scaler.inverse_transform(outputs)
    mse_loss = nn.MSELoss()(outputs, targets)
    zero_mask = (original_inputs[:, 3] == 0).unsqueeze(1).float()
    zero_condition_loss = torch.where(zero_mask == 1,
                                      torch.square(outputs_rescaled) * 5e3,
                                      torch.tensor(0.0, device=outputs.device))
    return mse_loss + zero_condition_loss.mean()

# Training loop
def train_model(num_epochs, model, loader, optimizer, scaler):
    model.train()
    loss_history = []  # List to store loss at each epoch
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets, original_inputs in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = custom_loss(outputs, targets, original_inputs, scaler)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(loader)
        loss_history.append(average_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {average_loss}")
    return loss_history

# Scaler initialization
output_scaler = PyTorchScaler(scaler_y.mean_, scaler_y.scale_)

# Model and optimizer
model = RegressionModel()

# # Uncomment this section if re-training the model
# optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# # Train the model
# loss_history = train_model(5000, model, loader, optimizer, output_scaler)

#Load Saved Model
model_path = os.path.join(os.getcwd(), "pinn_state_best.pth")
model.load_state_dict(torch.load(model_path))

X_scaled_test = scaler_X.transform(X_test)  # Ensure you scale the test data first
X_test_tensor = torch.tensor(X_scaled_test, dtype=torch.float32)
y_test_tensor = torch.tensor(scaler_y.transform(y_test.reshape(-1, 1)), dtype=torch.float32)

model.eval() #set model to evaluation mode

#Making prediction on the test set
with torch.no_grad():  # Disables gradient calculation
    predictions = model(X_test_tensor)

# convert predictions and actuals back to original scaling to interpret results
predicted = scaler_y.inverse_transform(predictions.numpy())
actual = scaler_y.inverse_transform(y_test_tensor.numpy())

# Calculate MSE 
mse = np.mean((predicted - actual)**2)
r2 = r2_score(actual, predicted)
print(f'R^2 on testing set: {r2}')
print(f'MSE on testing set: {mse}')

# Make predictions on the training set
with torch.no_grad():  # Ensure gradients are not computed
    X_scaled_train_tensor = torch.tensor(X_scaled_train, dtype=torch.float32)
    y_pred_train_tensor = model(X_scaled_train_tensor)
    y_pred_train = scaler_y.inverse_transform(y_pred_train_tensor.detach().numpy())

# Calculate R-squared and Mean Squared Error
r2_train = r2_score(y_train, y_pred_train)
mse_train = mean_squared_error(y_train, y_pred_train)  # Using sklearn's mean_squared_error function

print('R^2 on training set:', r2_train)
print('MSE on training set:', mse_train)

# Try saving with an absolute path to your home directory, for example
model_state_dict_path = os.path.join(os.getcwd(), "pinn_state_best.pth")
model_path = os.path.join(os.getcwd(), "pinn_model.pth")

torch.save(model.state_dict(), model_state_dict_path)
torch.save(model, model_path)

