# CNN Exam & Interview Cheat Sheet: How to Adapt the Code

When you get a specific problem statement or exam question, you **do not** need the entire comprehensive guide. You just need to extract a few core blocks of code and change a few parameters based on the question.

This document explains **exactly what to copy**, **what to change**, and provides a **start-to-finish example** of how to adapt the code.

---

## 🛠️ Step 1: Analyze the Question
Before copying any code, ask yourself these 3 vital questions:
1. **What is the dataset's shape?** Are the images Color (3 channels) or Grayscale (1 channel)? What size should they be?
2. **Is it Binary or Multi-Class?** Are you dividing things into 2 categories (Cat vs Dog) or 3+ categories (Digits 0-9)?
3. **Are there targeted constraints?** Did the question specifically ask for a Regularization technique (like Dropout), a specific Optimizer (like SGD), or an Activation (like LeakyReLU)?

---

## ✂️ Step 2: What to Copy & What to Change

### A. The Setup & Preprocessing
**What to take from the guide:**
```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Normalization step
def preprocess(img):
    img = tf.image.resize(img, (64, 64)) / 255.0  # Normalizes 0-255 to 0-1
    return img
```
**What to change:**
- Adjust the `(64, 64)` to whatever image size the question asks for (e.g., `(28, 28)` or `(128, 128)`).
- Provide the correct dataset loading method (e.g. `tf.keras.datasets.cifar10`).

### B. The Model Architecture
**What to take from the guide (Standard Deep Network):**
```python
model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D(),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid') # <-- THIS IS THE MOST IMPORTANT LINE TO CHANGE
])
```
**What to change:**
1. **`input_shape`**: Change `(64, 64, 3)` to `(size, size, 1)` if the images are grayscale!
2. **The Final `Dense` Layer**:
   - **Binary (2 classes):** Keep `layers.Dense(1, activation='sigmoid')`.
   - **Multi-Class (3+ classes):** Change to `layers.Dense(NUM_CLASSES, activation='softmax')`. Example: `layers.Dense(10, activation='softmax')`.

### C. Compiling the Model
**What to take from the guide:**
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
**What to change:**
1. **Optimizer:** Keep `'adam'` unless the question specifically says "Use SGD" or "Use RMSprop".
2. **Loss Function (Crucial!):**
   - **Binary:** `'binary_crossentropy'`
   - **Multi-Class (Labels are One-Hot, e.g. [0,0,1]):** `'categorical_crossentropy'`
   - **Multi-Class (Labels are Integers, e.g. 0, 1, 2):** `'sparse_categorical_crossentropy'`

---

## 🌟 Concrete Example Scenario 🌟

**The Exam/Interview Question Context:** 
> *"You are given the Fashion MNIST dataset consisting of 10 different categories of clothing. The images are Grayscale 28x28 pixels. Build a Convolutional Neural Network with Dropout to prevent overfitting, use the Adam optimizer, and train it for 5 epochs."*

### How to adapt the Comprehensive Guide to solve this:

**1. Data Loading & Preprocessing:**
*We take the basic setup and reshape our data for Grayscale.*
```python
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load Data (Specific to the question)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# 2. Preprocess: Normalize (division by 255) and reshape for Grayscale (adding the '1' channel dimension)
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
```

**2. The Model Architecture:**
*We take the Deep Network + Dropout block from the guide, but we modify the inputs and outputs.*
```python
model = models.Sequential([
    # MODIFICATION 1: Changed input_shape to (28, 28, 1) for Grayscale
    layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    
    # We kept Dropout from the guide because the question specifically asked to prevent overfitting
    layers.Dropout(0.5), 
    
    # MODIFICATION 2: Changed to 10 nodes and 'softmax' because there are 10 clothing categories!
    layers.Dense(10, activation='softmax') 
])
```

**3. Compile & Train:**
*We change the loss function to handle the 10 integer categories.*
```python
model.compile(
    optimizer='adam', # Question asked for Adam
    # MODIFICATION 3: Switched from binary_crossentropy because we have 10 classes as integers (0-9)
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# Train for 5 epochs as requested
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
```

### Summary of what you copied vs changed:
- **Kept from Guide:** The architectural sequence (`Conv2D` -> `MaxPool` -> `Flatten` -> `Dense`), the `Dropout` layer technique, the normalization logic (`/ 255.0`), and the core training loop concept.
- **Changed for Question:** 
  1. `input_shape` from `(64, 64, 3)` to `(28, 28, 1)` because it's no longer colored Cats vs Dogs.
  2. The Final Layer from `Dense(1, 'sigmoid')` to `Dense(10, 'softmax')` because we went from predicting 2 objects to 10 objects.
  3. The Loss function from `binary_crossentropy` to `sparse_categorical_crossentropy` to mathematically support 10 categories.
