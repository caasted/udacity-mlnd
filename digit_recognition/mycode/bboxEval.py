from keras.models import Model, load_model
from six.moves import cPickle as pickle
import modelEval
import genNew

pickle_file = 'SVHN-BB-1.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  train_sequences = save['train_sequences']
  train_bboxes = save['train_bboxes']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  test_sequences = save['test_sequences']
  test_bboxes = save['test_bboxes']
  del save  # hint to help gc free up memory

clf = load_model('SVHN-BB-1.h5')
evaluate = clf.evaluate(test_dataset, [test_sequences, test_labels[:,:,0], test_labels[:,:,1], 
                            test_labels[:,:,2], test_labels[:,:,3], test_labels[:,:,4], test_bboxes])
print("\n", evaluate)
print("\nAgainst test dataset")
print("\n", modelEval.accuracy(clf, test_dataset, test_sequences, test_labels))

photos_dataset, photos_labels, photos_sequences = genNew.generateData('photos')

print("\nAgainst photos dataset")
print("\n", modelEval.accuracy(clf, photos_dataset, photos_sequences, photos_labels))