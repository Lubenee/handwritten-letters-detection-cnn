import os
import tensorflow as tf
import helpers

def load_model(FILENAME):
    if os.path.exists(FILENAME):
        return tf.keras.models.load_model(FILENAME)
    return None

def save_model(FILENAME, model):
    model.save(FILENAME)

def log_model_data(model, LOG_FILENAME, MODEL_FILENAME, train_data_length, test_loss, test_acc):
# a for append
    with open(LOG_FILENAME, "a") as f:
        f.write(MODEL_FILENAME + "\n")
        f.write(helpers.describe_model(model, len(train_data_length)) + "\n")
        f.write("  Test loss: " + f"{test_loss:.4f}" + "\n")
        f.write("  Test accuracy: " + f"{test_acc:.4f}" + "\n\n")