from keras.datasets import mnist
import numpy as np
from six.moves import cPickle as pickle
from mycode import generateNewMNIST

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

new_train_dataset, new_train_labels, train_sequences = generateNewDataset(X_train, y_train, 5000000, 5, 1, True)
new_test_dataset, new_test_labels, test_sequences = generateNewDataset(X_test, y_test, 30000, 5, 1, True)

print("Generating sequence data complete.")
print(new_train_dataset.shape, new_train_labels.shape, train_sequences.shape)
print(new_test_dataset.shape, new_test_labels.shape, test_sequences.shape)

print(np.sum(new_train_labels, axis=0)) # Check distribution of classes
print("\n")
print(np.sum(train_sequences, axis=0)) # Check distribution of sequence lengths

pickle_file = 'MNIST-1.pickle'
try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': new_train_dataset,
    'train_labels': new_train_labels,
    'train_sequences': train_sequences,
    'test_dataset': new_test_dataset,
    'test_labels': new_test_labels,
    'test_sequences': test_sequences,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
    
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
