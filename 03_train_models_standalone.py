"""
Phase 3: CNN Model Training with MLFlow Tracking
All-in-one script with architectures and training
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.keras
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# CNN ARCHITECTURES
# ============================================================

def create_simple_cnn(input_shape, num_classes=3):
    """
    Simple CNN Architecture
    - Basic convolutional layers
    - Suitable for capturing local temporal patterns
    """
    model = models.Sequential(name='SimpleCNN')
    
    model.add(layers.Input(shape=input_shape))
    
    # First Conv Block
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name='conv1'))
    model.add(layers.MaxPooling1D(pool_size=2, name='pool1'))
    model.add(layers.Dropout(0.3, name='dropout1'))
    
    # Second Conv Block
    model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', name='conv2'))
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
    """
    model = models.Sequential(name='DeepCNN')
    
    model.add(layers.Input(shape=input_shape))
    
    # First Conv Block
    model.add(layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same', name='conv1'))
    model.add(layers.BatchNormalization(name='bn1'))
    model.add(layers.MaxPooling1D(pool_size=2, name='pool1'))
    model.add(layers.Dropout(0.3, name='dropout1'))
    
    # Second Conv Block
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name='conv2'))
    model.add(layers.BatchNormalization(name='bn2'))
    model.add(layers.MaxPooling1D(pool_size=2, name='pool2'))
    model.add(layers.Dropout(0.3, name='dropout2'))
    
    # Third Conv Block
    model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', name='conv3'))
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
    """
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Branch 1: Small kernel (short-term patterns)
    branch1 = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name='branch1_conv1')(inputs)
    branch1 = layers.MaxPooling1D(pool_size=2, name='branch1_pool')(branch1)
    
    # Branch 2: Medium kernel (medium-term patterns)
    branch2 = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same', name='branch2_conv1')(inputs)
    branch2 = layers.MaxPooling1D(pool_size=2, name='branch2_pool')(branch2)
    
    # Branch 3: Large kernel (long-term patterns)
    branch3 = layers.Conv1D(filters=64, kernel_size=7, activation='relu', padding='same', name='branch3_conv1')(inputs)
    branch3 = layers.MaxPooling1D(pool_size=2, name='branch3_pool')(branch3)
    
    # Concatenate branches
    concatenated = layers.Concatenate(name='concat')([branch1, branch2, branch3])
    
    # Additional processing
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001), name='conv_combined')(concatenated)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Dropout(0.4, name='dropout1')(x)
    
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001), name='conv_final')(x)
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
    """Compile model with optimizer, loss, and metrics"""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================

def load_data():
    """Load processed train/val/test datasets"""
    print(" Loading processed data...")
    
    train_df = pd.read_csv('data/NVDA_train.csv', index_col=0, parse_dates=True)
    val_df = pd.read_csv('data/NVDA_val.csv', index_col=0, parse_dates=True)
    test_df = pd.read_csv('data/NVDA_test.csv', index_col=0, parse_dates=True)
    
    # Get normalized feature columns
    feature_cols = [col for col in train_df.columns if col.endswith('_norm')]
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    # Extract features and targets
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df['target'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values
    
    # Convert targets from [-1, 0, 1] to [0, 1, 2] for categorical classification
    y_train = y_train + 1
    y_val = y_val + 1
    y_test = y_test + 1
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols

def create_sequences(X, y, sequence_length=10):
    """Create sequences for CNN input"""
    X_seq = []
    y_seq = []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    
    return np.array(X_seq), np.array(y_seq)

def calculate_class_weights(y_train):
    """Calculate class weights to handle imbalanced data"""
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weights = {i: weights[i] for i in range(len(classes))}
    
    print(f"\n  Class Weights:")
    print(f"  Short (0): {class_weights[0]:.3f}")
    print(f"  Hold  (1): {class_weights[1]:.3f}")
    print(f"  Long  (2): {class_weights[2]:.3f}")
    
    return class_weights

# ============================================================
# VISUALIZATION
# ============================================================

def plot_training_history(history, model_name, save_dir='models/plots'):
    """Plot training history"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'{model_name} - Training History', fontsize=16)
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    save_path = f"{save_dir}/{model_name}_history.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Training history saved: {save_path}")
    
    return save_path

def plot_confusion_matrix(y_true, y_pred, model_name, save_dir='models/plots'):
    """Plot confusion matrix"""
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Short', 'Hold', 'Long'],
        yticklabels=['Short', 'Hold', 'Long']
    )
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    save_path = f"{save_dir}/{model_name}_confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 Confusion matrix saved: {save_path}")
    
    return save_path

# ============================================================
# TRAINING AND EVALUATION
# ============================================================

def train_model(model, model_name, X_train_seq, y_train_seq, X_val_seq, y_val_seq, 
                class_weights, epochs=50, batch_size=32):
    """Train a CNN model with MLFlow tracking"""
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Start MLFlow run
    with mlflow.start_run(run_name=model_name):
        
        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("sequence_length", X_train_seq.shape[1])
        mlflow.log_param("num_features", X_train_seq.shape[2])
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("learning_rate", 0.001)
        
        # Log class weights
        for class_id, weight in class_weights.items():
            mlflow.log_param(f"class_weight_{class_id}", weight)
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train model
        print(f"\n Starting training...")
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Log final metrics
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        mlflow.log_metric("final_train_accuracy", final_train_acc)
        mlflow.log_metric("final_val_accuracy", final_val_acc)
        mlflow.log_metric("final_train_loss", final_train_loss)
        mlflow.log_metric("final_val_loss", final_val_loss)
        
        # Plot and log training history
        history_plot = plot_training_history(history, model_name)
        mlflow.log_artifact(history_plot)
        
        # Save model
        model_path = f"models/saved_models/{model_name}.h5"
        os.makedirs("models/saved_models", exist_ok=True)
        model.save(model_path)
        mlflow.log_artifact(model_path)
        
        print(f"\n Training complete!")
        print(f"  Final Train Accuracy: {final_train_acc:.4f}")
        print(f"  Final Val Accuracy: {final_val_acc:.4f}")
        print(f"  Model saved: {model_path}")
    
    return model, history

def evaluate_model(model, model_name, X_test_seq, y_test_seq):
    """Evaluate model on test set"""
    
    print(f"\n Evaluating {model_name} on test set...")
    
    # Predictions
    y_pred_probs = model.predict(X_test_seq, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    test_loss, test_acc = model.evaluate(X_test_seq, y_test_seq, verbose=0)
    
    # Calculate precision, recall, F1 manually
    from sklearn.metrics import precision_score, recall_score
    test_precision = precision_score(y_test_seq, y_pred, average='weighted')
    test_recall = recall_score(y_test_seq, y_pred, average='weighted')
    test_f1 = f1_score(y_test_seq, y_pred, average='weighted')
    
    # Classification report
    print(f"\n{model_name} - Classification Report:")
    print(classification_report(
        y_test_seq, y_pred,
        target_names=['Short', 'Hold', 'Long'],
        digits=4
    ))
    
    # Confusion matrix
    cm_plot = plot_confusion_matrix(y_test_seq, y_pred, model_name)
    
    # Log to MLFlow
    with mlflow.start_run(run_name=f"{model_name}_test"):
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1_score", test_f1)
        mlflow.log_artifact(cm_plot)
    
    metrics = {
        'model_name': model_name,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1_score': test_f1
    }
    
    return metrics

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    """Main training pipeline"""
    
    print("="*60)
    print("PHASE 3: CNN MODEL TRAINING WITH MLFLOW")
    print("="*60)
    
    # Set MLFlow tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("NVDA_Trading_CNN")
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_data()
    
    # Create sequences
    sequence_length = 10
    print(f"\n Creating sequences (length={sequence_length})...")
    
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
    
    print(f"  Train sequences: {X_train_seq.shape}")
    print(f"  Val sequences: {X_val_seq.shape}")
    print(f"  Test sequences: {X_test_seq.shape}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(y_train_seq)
    
    # Input shape for models
    input_shape = (sequence_length, len(feature_cols))
    print(f"\n Input shape: {input_shape}")
    
    # Training parameters
    epochs = 50
    batch_size = 32
    
    # Dictionary to store all results
    all_results = []
    
    # Train Simple CNN
    print(f"\n{'#'*60}")
    print("MODEL 1: SIMPLE CNN")
    print(f"{'#'*60}")
    simple_model = create_simple_cnn(input_shape)
    simple_model = compile_model(simple_model)
    simple_model, simple_history = train_model(
        simple_model, "SimpleCNN", 
        X_train_seq, y_train_seq, X_val_seq, y_val_seq,
        class_weights, epochs, batch_size
    )
    simple_metrics = evaluate_model(simple_model, "SimpleCNN", X_test_seq, y_test_seq)
    all_results.append(simple_metrics)
    
    # Train Deep CNN
    print(f"\n{'#'*60}")
    print("MODEL 2: DEEP CNN")
    print(f"{'#'*60}")
    deep_model = create_deep_cnn(input_shape)
    deep_model = compile_model(deep_model)
    deep_model, deep_history = train_model(
        deep_model, "DeepCNN",
        X_train_seq, y_train_seq, X_val_seq, y_val_seq,
        class_weights, epochs, batch_size
    )
    deep_metrics = evaluate_model(deep_model, "DeepCNN", X_test_seq, y_test_seq)
    all_results.append(deep_metrics)
    
    # Train Custom CNN
    print(f"\n{'#'*60}")
    print("MODEL 3: CUSTOM CNN")
    print(f"{'#'*60}")
    custom_model = create_custom_cnn(input_shape)
    custom_model = compile_model(custom_model)
    custom_model, custom_history = train_model(
        custom_model, "CustomCNN",
        X_train_seq, y_train_seq, X_val_seq, y_val_seq,
        class_weights, epochs, batch_size
    )
    custom_metrics = evaluate_model(custom_model, "CustomCNN", X_test_seq, y_test_seq)
    all_results.append(custom_metrics)
    
    # Model Comparison
    print(f"\n{'='*60}")
    print("MODEL COMPARISON - TEST SET RESULTS")
    print(f"{'='*60}")
    
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('models/model_comparison.csv', index=False)
    print(f"\n Results saved to: models/model_comparison.csv")
    
    # Select best model
    best_model_idx = results_df['test_f1_score'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'model_name']
    best_f1 = results_df.loc[best_model_idx, 'test_f1_score']
    
    print(f"\n BEST MODEL: {best_model_name}")
    print(f"   Test F1-Score: {best_f1:.4f}")
    print(f"   Test Accuracy: {results_df.loc[best_model_idx, 'test_accuracy']:.4f}")
    
    print(f"\n{'='*60}")
    print(" PHASE 3 COMPLETE!")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. Review MLFlow UI: mlflow ui")
    print("2. Check model comparison: models/model_comparison.csv")
    print("3. Move to Phase 4: API Development")
    
    return results_df

if __name__ == "__main__":
    results = main()