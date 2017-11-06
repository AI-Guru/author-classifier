import pickle
import os
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.optimizers import SGD
import numpy as np
import shutil

#model_type = "multi-layer-perceptron"
model_type = "convolutional-neural-network"

training_epochs = 10
training_batch_size = 128

remove_model = True

def main(args=None):
    preprocessed_data_path = "preprocessed"
    model_path = "model"
    train_network(preprocessed_data_path, model_path)

def train_network(preprocessed_data_path, model_path):

    # Make sure that there is a proper folder for analysis.
    if os.path.exists(model_path) and remove_model is True:
        shutil.rmtree(model_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Loading preprocessed data.
    preprocessed_data_name = "preprocessed.pickle"
    preprocessed_data_path = os.path.join(preprocessed_data_path, preprocessed_data_name)
    print("Loading preprocessed data...")
    preprocessed_data = pickle.load(open(preprocessed_data_path, "rb" ) )
    X_train, X_test, y_train, y_test, class_names = preprocessed_data
    print("X_train size: {}".format(len(X_train)))
    print("X_test size: {}".format(len(X_test)))
    print("y_train size: {}".format(len(y_train)))
    print("y_test size: {}".format(len(y_test)))
    print("class_names size: {}".format(len(class_names)))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Create the model.
    model = create_model(model_type, len(class_names))

    # Adapt data if necessary.
    if model_type == "convolutional-neural-network":
        print("Adapting data...")
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)
        print('Text train shape: ', X_train.shape)
        print('Text test shape: ', X_test.shape)

    # Print the model summary.
    model.summary()

    # Compile the model.
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

    # Train the model.
    estimator = model.fit(X_train, y_train,
                          validation_split=0.2,
                          epochs=training_epochs, batch_size=training_batch_size, verbose=1)
    print("Training accuracy: %.2f%% / Validation accuracy: %.2f%%" %
          (100*estimator.history['acc'][-1], 100*estimator.history['val_acc'][-1])) # TODO Rewrite

    #  Save the model.
    model_name = "network.model"
    model_file_path = os.path.join(model_path, model_name)
    print("Writing model to", model_file_path + "...")
    model.save(model_file_path)
    print("Model saved.")

    # Create meta-data.
    model_metadata = model_type, class_names

    # Save model meta-data.
    model_metadata_name = "network.meta"
    model_metadata_file_path = os.path.join(model_path, model_metadata_name)
    print("Writing model-metadata to", model_metadata_file_path + "...")
    with open(model_metadata_file_path, "wb") as output_file:
        pickle.dump(model_metadata, output_file)
        print("Model-metadata saved.")


def create_model(model_type, number_of_classes):
    print("Creating model of type " + model_type)

    if model_type == "multi-layer-perceptron":
        model = Sequential()
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal', input_dim=300))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.2))
        model.add(Dense(number_of_classes, activation='softmax'))

    elif model_type == "convolutional-neural-network":
        inputs = Input(shape=(300,1))
        x = Conv1D(64, 3, strides=1, padding='same', activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(128, 3, strides=1, padding='same', activation='relu')(x)
        x = GlobalMaxPooling1D()(x)
        outputs = Dense(number_of_classes, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs, name='cnn')

    else:
        raise Exception("Unknown model-type " + model_type + ".")

    return model

if __name__ == "__main__":
    main()
