
# cnn model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
import numpy as np

from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

def load_dataset():
	X_train = read_csv('Data/X_train.csv')
	y_train = read_csv('Data/y_train.csv')

	X_data = X_train.iloc[:,3:].values
	X_data = StandardScaler().fit_transform(X_data)# standarlize the data
	X_data_2D = np.reshape(X_data, (int(487680/128),128,10))

	y_data = y_train.surface.astype('category')
	# integer encode
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(y_data)
	# binary encode
	onehot_encoder = OneHotEncoder(sparse=False)
	integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
	y_onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

	X_train, X_test, y_train, y_test = train_test_split(X_data_2D, y_onehot_encoded, test_size=0.1, random_state=42)
	print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)
	return(X_train, y_train, X_test, y_test)


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
	verbose, epochs, batch_size = 1, 50, 100
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	#model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(LSTM(100, return_sequences=True, input_shape=(n_timesteps,n_features)))

	model.add(LSTM(100, return_sequences=False))

	#model.add(Flatten())
	model.add(Dense(n_outputs, activation='softmax'))
	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)
	mc = ModelCheckpoint('best_model_lstm_only.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
	# fit model
	history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[es, mc])

	pyplot.plot(history.history['loss'], label='train')
	pyplot.plot(history.history['val_loss'], label='test')
	pyplot.legend()
	pyplot.show()
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=10):
	# load data
	trainX, trainy, testX, testy = load_dataset()
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)

# run the experiment
run_experiment()
