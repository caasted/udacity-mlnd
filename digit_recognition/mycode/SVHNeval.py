from keras.models import Model, load_model
from six.moves import cPickle as pickle
import modelEval

pickle_file = 'SVHN-1.pickle'
with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	train_dataset = save['train_dataset']
	train_labels = save['train_labels']
	train_sequences = save['train_sequences']
	test_dataset = save['test_dataset']
	test_labels = save['test_labels']
	test_sequences = save['test_sequences']
	del save  # hint to help gc free up memory

clf = load_model('SVHN-1.h5')
evaluate = clf.evaluate(test_dataset, [test_sequences, test_labels[:,:,0], test_labels[:,:,1], 
                            test_labels[:,:,2], test_labels[:,:,3], test_labels[:,:,4]])
print("\n", evaluate)
print("\n", modelEval.getAccuracy(clf, test_dataset, test_sequences, test_labels))
