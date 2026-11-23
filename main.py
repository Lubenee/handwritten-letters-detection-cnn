from latin_letter_dataset import get_train_set, get_test_set
from cnn_model import create_cnn, train_cnn, test_cnn
import keras
import file_handler
from drawing_window import Window
from constants import LATIN_MODEL_FILENAME, LETTERS

if __name__ == "__main__":
    print("Keras: ", keras.__version__)
    model = file_handler.load_model(LATIN_MODEL_FILENAME)
    
    if model is None:
        print("Model " + LATIN_MODEL_FILENAME + " not found. Training new model..")
        model = create_cnn(len(LETTERS))
        
        x_train, y_train = get_train_set()
        x_test, y_test = get_test_set()
        train_cnn(model, x_train, y_train, x_test, y_test)
        file_handler.save_model(LATIN_MODEL_FILENAME, model)
        
        test_loss, test_acc = test_cnn(model, x_test, y_test)
        file_handler.log_model_data(model, 'Logs/LATIN_MODEL_LOGS.txt', LATIN_MODEL_FILENAME, len(x_train), test_loss, test_acc)
    
    drawingWindow = Window(model)
    drawingWindow.run()