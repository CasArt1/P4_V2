"""
CNN Model Architectures for NVDA Trading Strategy
Three different architectures: Simple, Deep, and Custom
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2

def create_simple_cnn(input_shape, num_classes=3):
    """
    Simple CNN Architecture
    - Basic convolutional layers
    - Suitable for capturing local temporal patterns
    
    Args:
        input_shape (tuple): Shape of input data (timesteps, features)
        num_classes (int): Number of output classes (3: Long, Hold, Short)
    
    Returns:
        keras.Model: Compiled model
    """
    model = models.Sequential(name='SimpleCNN')
    
    # Reshape for Conv1D: (batch, timesteps, features)
    model.add(layers.Input(shape=input_shape))
    
    # First Conv Block
    model.add(layers.Conv1D(
        filters=64,
        kernel_size=3,
        activation='relu',
        padding='same',
        name='conv1'
    ))
    model.add(layers.MaxPooling1D(pool_size=2, name='pool1'))
    model.add(layers.Dropout(0.3, name='dropout1'))
    
    # Second Conv Block
    model.add(layers.Conv1D(
        filters=32,
        kernel_size=3,
        activation='relu',
        padding='same',
        name='conv2'
    ))
    model.add(layers.MaxPooling1D(pool_size=2, name='pool2'))
    model.add(layers.Dropout(0.3, name='dropout2'))
    
    # Flatten and Dense Layers
    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dense(64, activation='relu', name='dense1'))
    model.add(layers.Dropout(0.4, name='dropout3'))
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))
    
    return model

def create_deep_cnn(input_shape, num_classes=3):
    """
    Deep CNN Architecture
    - Multiple convolutional blocks
    - Batch normalization for stable training
    - More capacity to learn complex patterns
    
    Args:
        input_shape (tuple): Shape of input data (timesteps, features)
        num_classes (int): Number of output classes
    
    Returns:
        keras.Model: Compiled model
    """
    model = models.Sequential(name='DeepCNN')
    
    model.add(layers.Input(shape=input_shape))
    
    # First Conv Block
    model.add(layers.Conv1D(
        filters=128,
        kernel_size=5,
        activation='relu',
        padding='same',
        name='conv1'
    ))
    model.add(layers.BatchNormalization(name='bn1'))
    model.add(layers.MaxPooling1D(pool_size=2, name='pool1'))
    model.add(layers.Dropout(0.3, name='dropout1'))
    
    # Second Conv Block
    model.add(layers.Conv1D(
        filters=64,
        kernel_size=3,
        activation='relu',
        padding='same',
        name='conv2'
    ))
    model.add(layers.BatchNormalization(name='bn2'))
    model.add(layers.MaxPooling1D(pool_size=2, name='pool2'))
    model.add(layers.Dropout(0.3, name='dropout2'))
    
    # Third Conv Block
    model.add(layers.Conv1D(
        filters=32,
        kernel_size=3,
        activation='relu',
        padding='same',
        name='conv3'
    ))
    model.add(layers.BatchNormalization(name='bn3'))
    model.add(layers.GlobalAveragePooling1D(name='gap'))
    model.add(layers.Dropout(0.4, name='dropout3'))
    
    # Dense Layers
    model.add(layers.Dense(128, activation='relu', name='dense1'))
    model.add(layers.Dropout(0.4, name='dropout4'))
    model.add(layers.Dense(64, activation='relu', name='dense2'))
    model.add(layers.Dropout(0.3, name='dropout5'))
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))
    
    return model

def create_custom_cnn(input_shape, num_classes=3):
    """
    Custom CNN Architecture
    - Multi-scale convolutional approach
    - Different kernel sizes to capture various temporal patterns
    - Regularization to prevent overfitting
    
    Args:
        input_shape (tuple): Shape of input data (timesteps, features)
        num_classes (int): Number of output classes
    
    Returns:
        keras.Model: Compiled model
    """
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Branch 1: Small kernel (short-term patterns)
    branch1 = layers.Conv1D(
        filters=64,
        kernel_size=3,
        activation='relu',
        padding='same',
        name='branch1_conv1'
    )(inputs)
    branch1 = layers.MaxPooling1D(pool_size=2, name='branch1_pool')(branch1)
    
    # Branch 2: Medium kernel (medium-term patterns)
    branch2 = layers.Conv1D(
        filters=64,
        kernel_size=5,
        activation='relu',
        padding='same',
        name='branch2_conv1'
    )(inputs)
    branch2 = layers.MaxPooling1D(pool_size=2, name='branch2_pool')(branch2)
    
    # Branch 3: Large kernel (long-term patterns)
    branch3 = layers.Conv1D(
        filters=64,
        kernel_size=7,
        activation='relu',
        padding='same',
        name='branch3_conv1'
    )(inputs)
    branch3 = layers.MaxPooling1D(pool_size=2, name='branch3_pool')(branch3)
    
    # Concatenate branches
    concatenated = layers.Concatenate(name='concat')([branch1, branch2, branch3])
    
    # Additional processing
    x = layers.Conv1D(
        filters=128,
        kernel_size=3,
        activation='relu',
        padding='same',
        kernel_regularizer=l2(0.001),
        name='conv_combined'
    )(concatenated)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Dropout(0.4, name='dropout1')(x)
    
    x = layers.Conv1D(
        filters=64,
        kernel_size=3,
        activation='relu',
        padding='same',
        kernel_regularizer=l2(0.001),
        name='conv_final'
    )(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    x = layers.Dropout(0.5, name='dropout2')(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001), name='dense1')(x)
    x = layers.Dropout(0.4, name='dropout3')(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001), name='dense2')(x)
    x = layers.Dropout(0.3, name='dropout4')(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='CustomCNN')
    
    return model

def compile_model(model, learning_rate=0.001):
    """
    Compile model with optimizer, loss, and metrics
    
    Args:
        model (keras.Model): Model to compile
        learning_rate (float): Learning rate for optimizer
    
    Returns:
        keras.Model: Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    return model

if __name__ == "__main__":
    """Test model architectures"""
    
    # Example input shape: (timesteps=10, features=28)
    input_shape = (10, 28)
    
    print("="*60)
    print("CNN MODEL ARCHITECTURES")
    print("="*60)
    
    # Simple CNN
    print("\n1. Simple CNN:")
    simple_model = create_simple_cnn(input_shape)
    simple_model = compile_model(simple_model)
    simple_model.summary()
    print(f"Total parameters: {simple_model.count_params():,}")
    
    # Deep CNN
    print("\n" + "="*60)
    print("2. Deep CNN:")
    deep_model = create_deep_cnn(input_shape)
    deep_model = compile_model(deep_model)
    deep_model.summary()
    print(f"Total parameters: {deep_model.count_params():,}")
    
    # Custom CNN
    print("\n" + "="*60)
    print("3. Custom CNN:")
    custom_model = create_custom_cnn(input_shape)
    custom_model = compile_model(custom_model)
    custom_model.summary()
    print(f"Total parameters: {custom_model.count_params():,}")
    
    print("\n" + "="*60)
    print("✅ All models created successfully!")
    print("="*60)
