import numpy as np
import cv2
import sys
import file_handler
import constants
from cyrilic_letter_dataset import get_train_data

# Builds a text description of the model's settings, read from the model itself.
# This way it's correct even when the model was loaded from a .keras file
# and the code in create_cnn() has been changed since then.
def describe_model(model, n_train_images):
    BATCH_SIZE = 32
    
    lines = ["  Layers:"]
    for layer in model.layers:
        config = layer.get_config() # dict with all settings of this layer
        settings = []
        # only keep the settings we care about (each layer type has different ones)
        for key in ("filters", "kernel_size", "pool_size", "units", "activation", "rate"):
            if key in config:
                settings.append(key + "=" + str(config[key]))
        lines.append("    " + type(layer).__name__ + "(" + ", ".join(settings) + ")")

    lines.append("  Total params: " + f"{model.count_params():,}")
    optimizer = model.optimizer
    lines.append("  Optimizer: " + type(optimizer).__name__ + ", learning_rate=" + f"{float(optimizer.learning_rate):g}")

    # The model doesn't store "epochs", but it does store how many weight updates
    # (steps) it did. One epoch = ceil(n_train_images / BATCH_SIZE) steps.
    steps = int(optimizer.iterations)
    steps_per_epoch = int(np.ceil(n_train_images / BATCH_SIZE))
    lines.append("  Batch size: " + str(BATCH_SIZE)
                 + ", Epochs: " + str(steps // steps_per_epoch)
                 + " (" + str(steps) + " steps)")
    return "\n".join(lines)


# Makes our drawing look like an EMNIST image: EMNIST letters are cut out,
# centered, and scaled to fill the 28x28 frame with a small 2 px border.
# Expects a 1-channel grayscale image, shape (H, W, 1) or (H, W), white ink on black.
def image_to_emnist_format(image):
    gray = image.reshape(image.shape[:2])        # (H, W, 1) -> (H, W)

    # 1. crop to the bounding box of the ink (all non-black pixels)
    x, y, w, h = cv2.boundingRect(gray)          # smallest box that contains every non-zero pixel
    letter = gray[y:y+h, x:x+w]

    # 2. pad to a square, letter in the middle (so resizing doesn't stretch it)
    side = max(h, w)
    square = np.zeros((side, side), np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    square[top:top+h, left:left+w] = letter

    # 3. shrink to 24x24 and put it in the middle of a 28x28 black image (2 px border)
    small = np.zeros((28, 28), np.float32)
    small[2:26, 2:26] = cv2.resize(square, (24, 24), interpolation=cv2.INTER_AREA)  # INTER_AREA averages pixels
    return small / 255.0

def invert_normalize(x):
    x = (255 - x.astype(np.float32)) / 255.0
    return x[..., np.newaxis]                  

def load_or_build_cyrilic_dataset():
    dataset = file_handler.load_cyrilic_dataset(constants.CYRILIC_DATASET_ARCHIVE)
    if dataset is None:
        print("[WARN] Dataset archive not found, building it from the cmnist repo...")
        if not get_train_data():
            sys.exit("[ERROR] cmnist repo not found. ...")
        dataset = file_handler.load_cyrilic_dataset(constants.CYRILIC_DATASET_ARCHIVE)

    return {k: invert_normalize(v) if k.startswith("x_") else v for k, v in dataset.items()}