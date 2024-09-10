from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import os
import random
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define constants
DEFAULT_IMAGE_DIR = r'C:\Users\asus\Desktop\Recipe Website\Food_Recognition_System\Fruit_veg_webapp\Fruit_veg_webapp'
DEFAULT_OUTPUT_MODEL = r'C:\Users\asus\Desktop\Recipe Website\Food_Recognition_System\Fruit_veg_webapp\model.h5'
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_IMAGE_SIZE = (299, 299)
DEFAULT_TESTING_PERCENTAGE = 10
DEFAULT_VALIDATION_PERCENTAGE = 10

MODEL_ARCHITECTURES = {
    'inception_v3': {
        'model_func': InceptionV3,
        'weights': 'imagenet',
        'input_shape': (299, 299, 3)
    },
    'mobilenet_v2': {
        'model_func': MobileNetV2,
        'weights': 'imagenet',
        'input_shape': (224, 224, 3)
    }
}

def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """Builds a list of training images from the file system."""
    if not os.path.exists(image_dir):
        print(f"Image directory '{image_dir}' not found.")
        return None

    result = collections.defaultdict(lambda: {'train': [], 'test': [], 'validation': []})
    sub_dirs = [x for x in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, x))]
    
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(image_dir, sub_dir)
        file_list = os.listdir(sub_dir_path)
        if not file_list:
            continue
        
        for file_name in file_list:
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(sub_dir_path, file_name)
                rand = random.random()
                if rand < testing_percentage / 100.0:
                    set_name = 'test'
                elif rand < (testing_percentage + validation_percentage) / 100.0:
                    set_name = 'validation'
                else:
                    set_name = 'train'
                result[sub_dir][set_name].append(file_path)
    
    return result

def build_model(model_name, num_classes):
    """Builds and compiles the model."""
    model_info = MODEL_ARCHITECTURES[model_name]
    base_model = model_info['model_func'](weights=model_info['weights'], include_top=False, input_shape=model_info['input_shape'])
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default=DEFAULT_IMAGE_DIR, help='Path to the base image directory.')
    parser.add_argument('--output_model', type=str, default=DEFAULT_OUTPUT_MODEL, help='Path to save the trained model.')
    parser.add_argument('--model_name', type=str, choices=MODEL_ARCHITECTURES.keys(), default='inception_v3', help='Base model to use.')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of epochs for training.')

    args = parser.parse_args()

    # Prepare data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=20
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(args.image_dir, 'train'),  # Ensure you have a 'train' directory within 'image_dir'
        batch_size=args.batch_size,
        target_size=DEFAULT_IMAGE_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = test_datagen.flow_from_directory(
        os.path.join(args.image_dir, 'validation'),  # Ensure you have a 'validation' directory within 'image_dir'
        batch_size=args.batch_size,
        target_size=DEFAULT_IMAGE_SIZE,
        class_mode='categorical'
    )

    # Debugging print statements
    print(f"Training images found: {train_generator.samples}")
    print(f"Validation images found: {validation_generator.samples}")

    # Build and train the model
    num_classes = len(train_generator.class_indices)
    model = build_model(args.model_name, num_classes)
    
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=args.epochs
    )
    
    # Save the model
    model.save(args.output_model)
    print(f"Model saved to {args.output_model}")

if __name__ == "__main__":
    main()
