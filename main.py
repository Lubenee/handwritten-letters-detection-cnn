from latin_letter_dataset import get_train_set, get_test_set
from cnn_model import create_cnn, train_cnn, test_cnn
import file_handler
from drawing_window import Window
import helpers
import constants
import cyrilic_letter_dataset

if __name__ == "__main__":
    cyrilic_model = file_handler.load_model(constants.CYRILIC_MODEL_FILENAME)
    latin_model = file_handler.load_model(constants.LATIN_MODEL_FILENAME)
    
    if cyrilic_model is None:
        print("[WARN] Model " + constants.CYRILIC_MODEL_FILENAME + " not found. Training new model..")
        dataset = cyrilic_letter_dataset.load_or_build_cyrilic_dataset()
        cyrilic_model = create_cnn(len(constants.RUSSIAN_LETTERS))
        x_train, y_train = dataset["x_train"], dataset["y_train"]
        x_test, y_test = dataset["x_test"], dataset["y_test"]
        train_cnn(cyrilic_model, x_train, y_train, x_test, y_test)
        file_handler.save_model(constants.CYRILIC_MODEL_FILENAME, cyrilic_model)
        test_loss, test_acc = test_cnn(cyrilic_model, x_test, y_test)
        file_handler.log_model_data(cyrilic_model, constants.CYRILIC_LOGS_FILE, constants.CYRILIC_MODEL_FILENAME, len(x_train), test_loss, test_acc)

    if latin_model is None:
        print("[WARN] Model " + constants.LATIN_MODEL_FILENAME + " not found. Training new model..")
        latin_model = create_cnn(len(constants.LATIN_LETTERS))
        x_train, y_train = get_train_set()
        x_test, y_test = get_test_set()
        train_cnn(latin_model, x_train, y_train, x_test, y_test)
        file_handler.save_model(constants.LATIN_MODEL_FILENAME, latin_model)
        test_loss, test_acc = test_cnn(latin_model, x_test, y_test)
        file_handler.log_model_data(latin_model, constants.LATIN_LOGS_FILE, constants.LATIN_MODEL_FILENAME, len(x_train), test_loss, test_acc)
        
    drawingWindow = Window(cyrilic_model)
    drawingWindow.run()
