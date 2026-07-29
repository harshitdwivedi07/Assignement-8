# Handwritten Digit Recognition using Artificial Neural Networks (ANN)

**Harshit Dwivedi**

## Objective
Classify handwritten digits (0-9) from the MNIST dataset using an
Artificial Neural Network built with TensorFlow/Keras, to automate
recognition of handwritten postal codes.

## Dataset Link
MNIST Handwritten Digits Dataset (Kaggle):
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

## Libraries Used
- pandas
- numpy
- scikit-learn
- matplotlib
- tensorflow / keras

## Methodology
1. Loaded `mnist_train.csv` (60,000 images, 784 pixel columns + 1 label
   column), displayed the first five records, identified the 784 pixel
   values as input features and `label` as the target variable, checked
   dataset dimensions and info, and displayed one sample digit image with
   Matplotlib.
2. Checked for missing values (none found), separated features (X) and
   target (y), normalized pixel values from the 0-255 range to 0-1, split
   the data 80/20 into training and test sets with stratified sampling, and
   one-hot encoded the target labels into 10 categorical columns.
3. Built an ANN with an input layer, a hidden layer of 128 neurons (ReLU),
   a hidden layer of 64 neurons (ReLU), and a 10-neuron softmax output
   layer. Compiled it with the Adam optimizer, categorical crossentropy
   loss, and accuracy as the tracked metric, then trained it for 10 epochs
   and predicted digit classes on the test set.
4. Evaluated the model using Test Accuracy, a Confusion Matrix, and a
   Classification Report, and plotted Accuracy vs Epoch and Loss vs Epoch
   curves.

## Model Architecture
```
Input Layer:      784 neurons (flattened 28x28 pixel image)
Hidden Layer 1:   128 neurons, ReLU activation
Hidden Layer 2:   64 neurons,  ReLU activation
Output Layer:     10 neurons,  Softmax activation
Optimizer:        Adam
Loss Function:    Categorical Crossentropy
Metric:           Accuracy
Epochs:           10
Total parameters: 109,386
```

## Results
| Metric | Value |
|--------|-------|
| Test Accuracy | 0.9738 (97.38%) |
| Test Loss | 0.0942 |

**Classification Report (test set, 12,000 images):**
| Digit | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| 0 | 0.98 | 0.99 | 0.99 |
| 1 | 0.97 | 0.99 | 0.98 |
| 2 | 0.97 | 0.97 | 0.97 |
| 3 | 0.97 | 0.96 | 0.97 |
| 4 | 0.98 | 0.97 | 0.97 |
| 5 | 0.97 | 0.97 | 0.97 |
| 6 | 0.99 | 0.97 | 0.98 |
| 7 | 0.98 | 0.97 | 0.98 |
| 8 | 0.97 | 0.95 | 0.96 |
| 9 | 0.95 | 0.97 | 0.96 |

**Observations:**
1. The model reached 97.38% test accuracy after just 10 epochs, showing
   that even a relatively simple two-hidden-layer ANN can learn the pixel
   patterns needed to distinguish the 10 digit classes well.
2. Both training and validation accuracy rise steeply in the first few
   epochs and then level off (training accuracy reached 99.25% by epoch
   10, validation 97.50%), while training and validation loss follow the
   mirrored pattern, with validation loss staying reasonably close to
   training loss, indicating the model is not significantly overfitting
   within 10 epochs.
3. The confusion matrix shows most misclassifications occur between
   visually similar digit pairs, such as 4 and 9, or 3 and 5, which is
   consistent with how humans occasionally misread these digits too.
4. Digits with more distinctive shapes, such as 0 and 1, achieve close to
   perfect precision and recall (0.98-0.99), while digit 9 shows the
   lowest precision (0.95), reflecting more stroke variation across
   different writers.

## Conclusion
This project built an Artificial Neural Network with two hidden layers
(128 and 64 neurons, ReLU activation) and a 10-neuron softmax output layer
to classify handwritten digits from the MNIST dataset, achieving a test
accuracy of 97.38% after 10 training epochs. The hidden layers are
essential to the network's performance: they allow the model to learn
increasingly abstract representations of the raw pixel input, transforming
simple edge and stroke patterns detected early on into higher-level shape
combinations that separate the 10 digit classes, something a model with no
hidden layers could not do for this non-linearly separable problem. A key
advantage of Deep Learning over traditional Machine Learning here is that
the ANN learns useful features directly from raw pixel values, without the
manual feature engineering that classical algorithms like Logistic
Regression or Decision Trees would need to achieve comparable accuracy on
image data. One limitation of ANNs is that they require large amounts of
labeled training data and computational resources to train effectively,
and they behave as a black box, making it difficult to interpret exactly
why a particular digit was classified the way it was.
