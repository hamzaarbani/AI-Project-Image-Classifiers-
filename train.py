import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Paths to your subfolders
TRAIN_DIR = 'dataset/seg_train'  
TEST_DIR = 'dataset/seg_test'    
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# Load Datasets using modern utilities
train_generator = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
)
val_generator = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
)

# Rescale values
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_generator = train_generator.map(lambda x, y: (normalization_layer(x), y))
val_generator = val_generator.map(lambda x, y: (normalization_layer(x), y))

# Get class names
import pathlib
data_dir = pathlib.Path(TRAIN_DIR)
extracted_labels = sorted([item.name for item in data_dir.glob('*') if item.is_dir()])
num_classes = len(extracted_labels)

# Build Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False 

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train and Save
model.fit(train_generator, validation_data=val_generator, epochs=3)
model.save('best_model.h5')

with open('labels.txt', 'w') as f:
    for label in extracted_labels:
        f.write(f"{label}\n")
print("Training configuration complete!")