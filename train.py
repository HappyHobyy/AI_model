import numpy as np
from tensorflow.keras.utils import to_categorical
from keras import models
from keras import layers
from dataloader import DataLoader
import argparse
import matplotlib.pyplot as plt

def vectorize_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        sequence = list(sequence)
        results[i, sequence] = 1.
    return results

def grid_search(dl, args):
    position_score = [4, 4, 4, 4]
    shape_X = sum(position_score)
    dl.setBias(position_score)

    X_labels = np.arange(dl.getCount())
    scores_with_bias = dl.getDatasetWithBias()

    X_train = vectorize_sequences(scores_with_bias, sum(position_score))
    one_hot_train_labels = to_categorical(X_labels)

    # 그리드 서치할 파라미터 값들
    batch_sizes = [4, 8, 16, 32]
    epochs_values = [100, 150]

    best_accuracy = 0
    best_params = {}

    for batch_size in batch_sizes:
        for epochs in epochs_values:
            model = models.Sequential([
                layers.Dense(40, activation='relu', input_shape=(shape_X,)),
                layers.Dense(20, activation='relu'),
                layers.Dense(dl.getCount(), activation='softmax')
            ])

            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            history = model.fit(X_train,
                                one_hot_train_labels,
                                epochs=epochs,
                                batch_size=batch_size,
                                verbose=0)

            accuracy = model.evaluate(X_train, one_hot_train_labels, verbose=0)[1]

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'batch_size': batch_size, 'epochs': epochs}

    print("Best parameters:", best_params)
    print("Best accuracy:", best_accuracy)

    return best_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/mbti_dataset.json')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=5)

    args = parser.parse_args()

    dl = DataLoader(args.data_path)

    best_params = grid_search(dl, args)

    # 데이터 전처리
    position_score = [4, 4, 4, 4]
    shape_X = sum(position_score)
    dl.setBias(position_score)

    X_labels = np.arange(dl.getCount())
    scores_with_bias = dl.getDatasetWithBias()

    X_train = vectorize_sequences(scores_with_bias, sum(position_score))
    one_hot_train_labels = to_categorical(X_labels)

    # 새 모델로 시작
    model = models.Sequential([
        layers.Dense(40, activation='relu', input_shape=(shape_X,)),
        layers.Dense(20, activation='relu'),
        layers.Dense(dl.getCount(), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train
    history = model.fit(X_train,
                        one_hot_train_labels,
                        epochs=best_params['epochs'],
                        batch_size=best_params['batch_size'],
                        verbose=1)

    # training and validation loss
    loss = history.history['loss']

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'b-', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Save the model
    model.save('./hobby_model.keras')
