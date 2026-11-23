from keras import layers
import tensorflow as tf

def create_cnn(output_layer_length):
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28,28,1)), 
        tf.keras.layers.Conv2D(64, (3,3), 1, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2),
        tf.keras.layers.Conv2D(128, (3,3), 1, activation='sigmoid'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='sigmoid'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(output_layer_length, activation='softmax'),
    ])
    
def train_cnn(model, x_train, y_train, x_test, y_test, epochs=10):
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=epochs, validation_data=(x_test, y_test))
    
def test_cnn(model, x_test, y_test): #test_loss, test_acc
    return model.evaluate(x_test, y_test)