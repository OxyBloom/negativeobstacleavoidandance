from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

# Create and save a model
model = Sequential([Dense(10, input_shape=(5,), activation='relu')])
model.save('test_model.h5')

# Load the model
loaded_model = load_model('test_model.h5')
