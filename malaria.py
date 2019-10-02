''' 
Data Source:
https://ceb.nlm.nih.gov/repositories/malaria-datasets/
'''

# Import libraries
import time
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
import cv2
import os
import random
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler

# Define function for capturing run-time
print('\n=== Program Initiated ===')
start_time = time.time()
def timer(start,end):
   hours, rem = divmod(end-start, 3600)
   minutes, seconds = divmod(rem, 60)
   return('{:0>2}:{:0>2}:{:05.2f}'.format(int(hours),int(minutes),seconds))



#%% Import data
   
print('\n[>>> Importing image data...]')
path = 'cell_images//'
image_paths = list(paths.list_images(path))

# Print dataset size and a sample path from image_paths
print('     Total dataset size: {}'.format(len(image_paths)))
print('     Sample path: {}'.format(image_paths[0]))

def show_random_image(image_paths):
    
    idx = np.random.randint(len(image_paths))
    
    image = cv2.imread(image_paths[idx],1) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    label = image_paths[idx].split(os.path.sep)[-2]
    
    print('     Category: ', label)
    plt.imshow(image)
    plt.xticks([]), plt.yticks([]) # hide ticks
    plt.show()
    
print('     Sample image: \n')
show_random_image(image_paths)
print('\n=== Data Import Complete ===')
print('--- Runtime =', timer(start_time, time.time()),'---')
new_time = time.time()



#%% Preprocess images

print('\n[>>> Preprocessing images...]')

random.seed(4)
random.shuffle(image_paths)
test_size = 0.20
random_state = 19

train_paths, test_paths = train_test_split(image_paths, test_size=test_size, 
                                           random_state=random_state)

def preprocess_images(image_paths, verbose=-1):
    
    data, labels =[],[]
    
    for (idx, image_path) in enumerate(image_paths):
        
        image = cv2.imread(image_paths[idx],1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(64,64))
        image = image/255
        
        label = image_path.split(os.path.sep)[-2]

        data.append(image)
        labels.append(label)

        if verbose > 0 and idx > 0 and (idx + 1) % verbose == 0:
            print('    Update: Images processed {}/{}'.format(idx + 1, 
                  len(image_paths)))

    return np.array(data), np.array(labels)

train_images, train_labels = preprocess_images(train_paths, verbose=5000)
test_images, test_labels = preprocess_images(test_paths)

print('\n=== Image Preprocessing Complete ===')
print('--- Runtime =', timer(start_time, time.time()),'---')
new_time = time.time()


#%% Prepare data for model training

print('\n[>>> Preparing model data...]')

# Data prep for Logistic Regression (baseline model)
train_imagesLR = train_images.reshape(len(train_images), -1)
test_imagesLR = test_images.reshape(len(test_images), -1)

train_XLR, val_XLR, train_yLR, val_yLR = train_test_split(train_imagesLR, 
                                                          train_labels, 
                                                          test_size=test_size, 
                                                          random_state=random_state)

# Data prep for SVM
train_imagesSVM = train_images.reshape(len(train_images), -1)
test_imagesSVM = test_images.reshape(len(test_images), -1)
stdscaler = StandardScaler().fit(train_imagesSVM)
train_imagesSVM = stdscaler.transform(train_imagesSVM)
test_imagesSVM = stdscaler.transform(test_imagesSVM)

train_XSVM, val_XSVM, train_ySVM, val_ySVM = train_test_split(train_imagesSVM, 
                                                          train_labels, 
                                                          test_size=test_size, 
                                                          random_state=random_state)

# Data prep for CNN
train_X, val_X, train_y, val_y = train_test_split(train_images, train_labels, 
                                                  test_size=test_size, 
                                                  random_state=random_state)

class_labels = list(np.unique((train_y)))
num_classes = len(class_labels)
print('     Class labels: ', class_labels)
print('     Validation set size: ', str(len(val_y)))
print('     Training set class split: ', Counter(train_y))

# Encode class labels
def encoder(r_list):
    lb = LabelBinarizer()
    return [lb.fit_transform(r) for r in r_list]
encoded = encoder([train_y, val_y])
train_y, val_y = encoded[0], encoded[1]

print('\n=== Data Preparation Complete ===')
print('--- Runtime =', timer(start_time, time.time()),'---')
new_time = time.time()


#%% Build & train baseline, stochastic logistic regression model

print('\n[>>> Training baseline, stochastic logistric regression model...]')

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=1, solver='lbfgs', multi_class='auto')
lr.fit(train_XLR, train_y.ravel())

print('\n=== Training Complete ===')
print('--- Runtime =', timer(start_time, time.time()),'---')
new_time = time.time()


#%% Generate LR model performance metrics

print('\n[>>> Performing model evaluation...]')

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def model_performance(test_y, test_X, regression_type, model_obj, model_name):
    
    # Call model on test set to make predictions
    true_y = encoder([test_y])[0]
    if regression_type:
        predicted_y = model_obj.predict(test_X)
    else: # Sequential() model type
        predicted_y = model_obj.predict_classes(test_X)
    
    # Convert class encodings back to text
    true_y = np.where(true_y == 0, 'Parasitized', 'Uninfected')
    predicted_y = np.where(predicted_y == 0, 'Parasitized', 'Uninfected')
    
    print('\n*** Malaria Classification Performance Metrics Using {} ***\n'.
          format(model_name))
    
    # Print performance metrics
    print('  >> Accuracy: {:.2%}'.format(accuracy_score(predicted_y, true_y)))
    print('  >> Confusion matrix: ')
    confusion_mat = confusion_matrix(predicted_y, true_y)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confusion_mat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            ax.text(x=j, y=i,
                    s=confusion_mat[i, j],
                    va='center', ha='center')
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.show()
    
    print('--- Classification report: \n',classification_report(predicted_y, true_y))

model_performance(test_y=test_labels, test_X=test_imagesLR, regression_type=True,
                  model_obj=lr, model_name='Stochastic Logistic Regression')

print('\n=== Evaluation Complete ===')
print('--- Runtime =', timer(new_time, time.time()),'---')
new_time = time.time()


#%% Build & train support vector machine model

print('\n[>>> Training support vector machine (SVM) model...]')

from sklearn.svm import SVC

svc = SVC(kernel='rbf', gamma=0.001, max_iter=5000, random_state=1)
svc.fit(train_XSVM, train_y.ravel())

print('\n=== Training Complete ===')
print('--- Runtime =', timer(start_time, time.time()),'---')
new_time = time.time()



#%% Generate SVM model performance metrics

print('\n[>>> Performing model evaluation...]')

model_performance(test_y=test_labels, test_X=test_imagesSVM, regression_type=True,
                  model_obj=svc, model_name='SVM')

print('\n=== Evaluation Complete ===')
print('--- Runtime =', timer(new_time, time.time()),'---')
new_time = time.time()

#%% Build & train convolutional neural network (CNN) model

print('\n[>>> Training CNN model...]')

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dropout, Dense
from keras.regularizers import l2

# Build model
model = Sequential()

filters = train_X.shape[1]
kernel_size = len(train_X.shape[1:])
input_shape = train_X.shape[1:]

model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', 
                 input_shape=input_shape, strides=3, kernel_regularizer=l2(0.01)))
model.add(Conv2D(filters=int(filters/2), kernel_size=kernel_size, activation='relu'))
model.add(Conv2D(filters=int(filters/4), kernel_size=kernel_size, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

loss_type = 'binary_crossentropy'
model.compile(loss=loss_type, optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train model
history = model.fit(train_X, train_y, batch_size=64,
          epochs=20, verbose=1, validation_data=(val_X,val_y))

# Plot loss
def plot_loss(model_hist, loss_type):
    fig = plt.figure(figsize=(6,4))
    plt.plot(model_hist.history['loss'])
    plt.plot(model_hist.history['val_loss'], 'g--')
    plt.title('CNN Model Loss')
    plt.ylabel(loss_type)
    plt.xlabel('Epoch')
    plt.legend(['Training Loss', 'Testing Loss'], loc='upper right')
    print('  >> Loss after final iteration: ', history.history['val_loss'][-1])
    print('  >> Accuracy after final iteration: {:.2%}'.
          format(history.history['val_acc'][-1]))
    plt.show()
    
plot_loss(history, loss_type.replace('_',' ').title())

print('\n=== Training Complete ===')
print('--- Runtime =', timer(start_time, time.time()),'---')
new_time = time.time()




#%% Generate CNN model performance metrics

print('\n[>>> Performing model evaluation...]')

model_performance(test_y=test_labels, test_X=np.array(test_images),
                  regression_type=False, model_obj=model, model_name='CNN')

print('\n=== Evaluation Complete ===')
print('--- Runtime =', timer(new_time, time.time()),'---')
print('--- Total Runtime =', timer(start_time, time.time()),'---')