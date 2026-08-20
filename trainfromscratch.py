import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Define directories for the resized images
train_dir = './resized_apples/'  # Parent directory containing resized_badapples and resized_goodapples
batch_size = 32
image_size = (224, 224)  # Image size (should match the size of the resized images)

# Data augmentation and splitting
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=50,  # Increase rotation
    width_shift_range=0.4,  # Increase horizontal shift
    height_shift_range=0.4,  # Increase vertical shift
    shear_range=0.3,  # Add more shear
    zoom_range=0.4,  # Add more zoom
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)
# Load training data
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',  # Assuming binary classification (good vs bad apples)
    subset='training'  # Only use training subset
)

# Load validation data
val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'  # Only use validation subset
)

# Build a simple CNN model (you can modify this as needed)
# model = models.Sequential([
#     layers.Input(shape=(224, 224, 3)),  # Define input shape using Input layer
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(512, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(1, activation='sigmoid')  # Binary classification
# ])

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),  # Define input shape using Input layer
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.00005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=50,
    callbacks=[early_stopping]
)

# Save the trained model
model.save('apple_classifier_model.h5')

print("Training complete and model saved as 'apple_classifier_model.h5'.")
