""" General Information 
    Organization:   SYNC Interns
    Track:          Machine Learning
    Role:           Internship
    Author:         Ahmed Hisham Fathy Hassabou
    Task No:        2
    Task Name:      Real Time Face Mask Detection
"""

###################################################################################################

""" Step 1: Import Needed Libraries """

import numpy as np
import keras
import keras.backend as k
from keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, Flatten, Dropout, Dense
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing import image
import cv2
import datetime

###################################################################################################

""" Step 2: Building Model to Classify Between Mask and No-Mask """

# Create a Sequential model
model = Sequential()

# Add the first convolutional layer with 32 filters, a 3x3 kernel, and ReLU activation function
# Input shape is set to (150, 150, 3) for images with height, width, and color channels
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

# Add a max-pooling layer to downsample the feature maps
model.add(MaxPooling2D())

# Add a second convolutional layer with 32 filters and ReLU activation
model.add(Conv2D(32, (3, 3), activation='relu'))

# Add another max-pooling layer
model.add(MaxPooling2D())

# Add a third convolutional layer with 32 filters and ReLU activation
model.add(Conv2D(32, (3, 3), activation='relu'))

# Add another max-pooling layer
model.add(MaxPooling2D())

# Flatten the feature maps into a one-dimensional vector
model.add(Flatten())

# Add a fully connected (dense) layer with 100 units and ReLU activation
model.add(Dense(100, activation='relu'))

# Add the output layer with 1 unit and a sigmoid activation function for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model using the Adam optimizer, binary cross-entropy loss, and accuracy metric
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Import the necessary libraries for image data preprocessing
from keras.preprocessing.image import ImageDataGenerator

# Create an image data generator for training data with various transformations
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Rescale pixel values to the range [0, 1]
    shear_range=0.2,   # Apply shear transformations
    zoom_range=0.2,    # Apply zoom transformations
    horizontal_flip=True  # Perform horizontal flips
)

# Create an image data generator for test data with pixel value rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Load and preprocess the training data from a directory
training_set = train_datagen.flow_from_directory(
    'Train_Dataset',         # Directory containing training images
    target_size=(150, 150),  # Resize images to (150, 150)
    batch_size=16,           # Batch size for training
    class_mode='binary'      # Binary classification (mask or no-mask)
)

# Load and preprocess the test data from a directory
test_set = test_datagen.flow_from_directory(
    'Test_Dataset',          # Directory containing test images
    target_size=(150, 150),  # Resize images to (150, 150)
    batch_size=16,           # Batch size for testing
    class_mode='binary'      # Binary classification (mask or no-mask)
)

# Train the model on the training data for 10 epochs and validate it on the test data
model_saved = model.fit_generator(
    training_set,
    epochs=10,
    validation_data=test_set,
)

# Save the trained model to a file
model.save('FaceMaskDetectionModel.h5', model_saved)

###################################################################################################

""" Step 3: Test Individual Images """

# Load the trained model for testing
mymodel = load_model('FaceMaskDetectionModel.h5')

# Load an image for testing
test_image = image.load_img(
    '/home/ahmedhh/Sync_Machine_Learning/Task_2/FaceMaskDetector/Ahmed_No_Mask.jpg',
    target_size=(150, 150, 3))

# Print the loaded test image file path
print("Loaded Test Image:", '/home/ahmedhh/Sync_Machine_Learning/Task_2/FaceMaskDetector/Ahmed_No_Mask.jpg')

# Convert the test image to a NumPy array
test_image = image.img_to_array(test_image)

# Expand the dimensions to match the model input shape
test_image = np.expand_dims(test_image, axis=0)

# Predict whether the person is wearing a mask or not
prediction = mymodel.predict(test_image)[0][0]

# Check the prediction result and print it
if prediction == 1:
    print("Prediction: NO MASK")
else:
    print("Prediction: MASK")

# Load another image for testing
test_image_1 = image.load_img(
    '/home/ahmedhh/Sync_Machine_Learning/Task_2/FaceMaskDetector/Ahmed_Mask.jpg',
    target_size=(150, 150, 3))

# Print the loaded test image file path
print("Loaded Test Image:", '/home/ahmedhh/Sync_Machine_Learning/Task_2/FaceMaskDetector/Ahmed_Mask.jpg')

# Convert the test image to a NumPy array
test_image_1 = image.img_to_array(test_image_1)

# Expand the dimensions to match the model input shape
test_image_1 = np.expand_dims(test_image_1, axis=0)

# Predict whether the person is wearing a mask or not
prediction = mymodel.predict(test_image_1)[0][0]

# Check the prediction result and print it
if prediction == 1:
    print("Prediction: NO MASK")
else:
    print("Prediction: MASK")

###################################################################################################

""" Step 4: Implementing Live Detection of Face Mask """

# Load the pre-trained model for face mask detection
mymodel = load_model('FaceMaskDetectionModel.h5')

# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while cap.isOpened():
    # Read a frame from the camera
    _, img = cap.read()

    # Detect faces in the frame using the Haar Cascade classifier
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        # Extract the detected face from the frame
        face_img = img[y:y + h, x:x + w]

        # Save the extracted face as a temporary image
        cv2.imwrite('temp.jpg', face_img)

        # Load the temporary image for testing
        test_image = image.load_img('temp.jpg', target_size=(150, 150, 3))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Make a prediction using the pre-trained model
        pred = mymodel.predict(test_image)[0][0]

        if pred == 1:
            # No mask detected
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Red rectangle
            cv2.putText(img, 'NO MASK', ((x + w) // 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            # Mask detected
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Green rectangle
            cv2.putText(img, 'MASK', ((x + w) // 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Display the date and time on the image
        datet = str(datetime.datetime.now())
        cv2.putText(img, datet, (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the result in a window named 'Face Mask Detection'
    cv2.imshow('Face Mask Detection', img)

    # If the 'q' key is pressed, exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
