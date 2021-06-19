from typing import List

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from tensorflow.python.keras.models import load_model

if __name__ == '__main__':
    for i in range(5):
        dataset = loadtxt("bank-additional-full.csv", delimiter=",")
        X = dataset[:, 0:19]
        y = dataset[:, 20]

        model = Sequential()
        model.summary()
        model.add(Dense(19, input_dim=19, activation='relu'))
        model.add(Dense(25, activation='sigmoid'))
        model.add(Dense(25, activation='sigmoid'))
        model.add(Dense(25, activation='sigmoid'))
        model.add(Dense(25, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, y, epochs=10, batch_size=50)

        model.save("model.h5")

        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        accuracy = model.evaluate(X, y)
        print("Accuracy: ", (accuracy * 100))

        predictions = model.predict_classes(X)
        num_correct = 0
        num_incorrect = 0
        for i in range(len(dataset)):
            if y[i] == 1:
                # print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
                if predictions[i] == 0:
                    num_incorrect += 1
                else:
                    num_correct += 1
        print("NUM CORRECT: ", num_correct)
        print("NUM INCORRECT: ", num_incorrect)

