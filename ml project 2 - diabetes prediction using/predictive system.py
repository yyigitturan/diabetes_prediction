import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open(r"C:\Users\yigit_5rkz30x\ML Project\ml project 2 - diabetes prediction using\trained_model.sav", 'rb'))

input_data = (13, 76, 60, 0, 0, 32.8, 0.18, 41)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# standardize the input data

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')
