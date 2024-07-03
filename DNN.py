import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import  load_model

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

X_scaled_train = scaler_X.fit_transform(X_train)
y_scaled_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

# Function to create model, required for KerasRegressor
def create_model(l2_reg=0.01):
    model = Sequential()
    model.add(Dense(64, input_dim=scaler_X.fit_transform(X_train).shape[1], kernel_regularizer=l2(l2_reg)))
    model.add(LeakyReLU(alpha=0.1))  # Leaky ReLU activation for the first hidden layer
    model.add(Dense(64, kernel_regularizer=l2(l2_reg)))
    model.add(LeakyReLU(alpha=0.1))  # Leaky ReLU activation for the second hidden layer
    model.add(Dense(64, kernel_regularizer=l2(l2_reg)))
    model.add(LeakyReLU(alpha=0.1))  # Leaky ReLU activation for the third hidden layer
    model.add(Dense(1))  # No activation function, or 'linear' by default, for the output layer

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# # Activate to fit a new model
# # Wrap the model with KerasRegressor
# model = KerasRegressor(build_fn=lambda: create_model(l2_reg=.01), epochs=150, batch_size=600, verbose=0)

# # Fit the model and save the history
# history = model.fit(scaler_X.fit_transform(X_train), y_scaled_train, epochs=150, batch_size=600, verbose=0, validation_split=0.3)

# Load pre-fit model
model_path = os.path.join(os.getcwd(), "Best_DNN.h5")
model = load_model(model_path) #Loading the saved and pretrained DNN model

X_scaled_test = scaler_X.transform(X_test)  # Ensure you scale the test data first
y_pred_scaled = model.predict(X_scaled_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
rmse_pred = np.sqrt(sum((y_pred-y_test)**2)/len(y_test))
mse_pred = np.mean((y_pred-y_test)**2)

print('The root mean squared error (RMSE) on the predictions is: ',round(rmse_pred,3), 'm')
print('The mean squared error (MSE) on the predictions is: ',round(mse_pred,3))

# # Activate only to save the DNN model 
# keras_model = model.model   
# keras_model.save('Best_DNN.h5')

