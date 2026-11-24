import os
import tensorflow as tf
import helpers
import numpy as np

def load_model(FILENAME):
    if os.path.exists(FILENAME):
        return tf.keras.models.load_model(FILENAME)
    return None

def save_model(FILENAME, model):
    model.save(FILENAME)

def log_model_data(model, LOG_FILENAME, MODEL_FILENAME, train_data_length, test_loss, test_acc):
    helpers.describe_model(model, train_data_length)
    
    # a for append
    with open(LOG_FILENAME, "a") as f:
        f.write(MODEL_FILENAME + "\n")
        f.write(helpers.describe_model(model, train_data_length) + "\n")
        f.write("  Test loss: " + f"{test_loss:.4f}" + "\n")
        f.write("  Test accuracy: " + f"{test_acc:.4f}" + "\n\n")
        
def load_cyrilic_dataset(FILENAME):
    if os.path.exists(FILENAME):
        return np.load(FILENAME)
    
def save_cyrilic_dataset(FILENAME, **arrays):
    np.savez_compressed(FILENAME, arrays)
    