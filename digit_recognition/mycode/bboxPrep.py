import os
from six.moves import cPickle as pickle
import genBBox

pickle_file = 'SVHN-dictionaries.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dictionary = save['train_dictionary']
  test_dictionary = save['test_dictionary']
  del save  # hint to help gc free up memory

print("Train dictionary:", len(train_dictionary))
print("Test dictionary:", len(test_dictionary))

rain_dataset, train_labels, train_sequences, train_bboxes = genBBox.generateData('train', train_dictionary)
test_dataset, test_labels, test_sequences, test_bboxes = genBBox.generateData('test', test_dictionary)
print("Done.")

pickle_file = 'SVHN-BB-1.pickle'
try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'train_sequences': train_sequences,
        'train_bboxes': train_bboxes,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
        'test_sequences': test_sequences,
        'test_bboxes': test_bboxes,
        # 'extra_dataset': extra_dataset,
        # 'extra_labels': extra_labels,
        # 'extra_sequences': extra_sequences,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
    
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
