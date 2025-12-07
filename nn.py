import random
import data_gen
import numpy as np
import pandas as pd
import tensorflow as tf
from random import randint

if __name__ == '__main__':
	profile_keys = data_gen.PROFILES.keys()
	data_gen.args.noise = False
	rng = random.Random(data_gen.args.seed)
	inputs = []
	outputs = []

	for profile_name in profile_keys:
		for _ in range(data_gen.args.samples):
			i, o = data_gen.generate_sample(rng, profile_name = profile_name)
			if data_gen.args.noise:
				i.inject_noise(noise_type = data_gen.args.noise_type, level = data_gen.args.noise_level, rng = rng)
			inputs.append(i.to_list())
			outputs.append(o.to_list())

	input_column_names = ['region', 'session_id', 'packets', 'avg_size', 'avg_time', 'min_time', 'max_time', 'upload_batch', 'upload_single']
	output_column_names = ['age', 'gender', 'condition', 'device_count']

	dataframe = pd.DataFrame(np.c_[inputs, outputs], columns = input_column_names + output_column_names)
	train_dataframe = dataframe.sample(frac = 0.75, random_state = randint(0, 2**32 - 1))
	validate_dataframe = dataframe.drop(train_dataframe.index)

	input_train_data = train_dataframe.drop(output_column_names, axis = 1)
	output_train_data = train_dataframe.drop(input_column_names, axis = 1)
	input_validate_data = validate_dataframe.drop(output_column_names, axis = 1)
	output_validate_data = validate_dataframe.drop(input_column_names, axis = 1)

	combined_inputs = pd.concat([input_train_data, input_validate_data])
	min_vals = combined_inputs.min(axis = 0)
	max_vals = combined_inputs.max(axis = 0)
	ranges = max_vals - min_vals
	input_train_data = (input_train_data - min_vals) / ranges
	input_validate_data = (input_validate_data - min_vals) / ranges

	input_shape = [input_train_data.shape[1]]
	output_shape = output_train_data.shape[1]

	model = tf.keras.Sequential \
	([
		tf.keras.layers.Dense \
		(
			units = 64,
			activation = 'relu',
			input_shape = input_shape
		),
		tf.keras.layers.Dense \
		(
			units = 64,
			activation = 'relu'
		),
		tf.keras.layers.Dense(units = output_shape)
	])

	model.compile(optimizer = 'adam', loss = 'mae')
	model.fit(input_train_data, output_train_data, validation_data = (input_validate_data, output_validate_data), batch_size = input_train_data.shape[0], epochs = 100)

	predictions = model.predict(input_validate_data)
	print(predictions.shape)
	print('Sample results:')
	for i in range(5):
		prediction = predictions[i]
		actual = output_validate_data.iloc[i].to_numpy()
		print('Prediction:', prediction, '| Actual:', actual)
