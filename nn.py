import random
import data_gen
import numpy as np
import pandas as pd
import tensorflow as tf
from random import randint

def print_bar():
	print('====================')

def extract_dataframe_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
	return df.drop(df.columns.difference([column_name]), axis = 1)

def compute_mae(a: pd.DataFrame, b: pd.DataFrame) -> float:
	return (a - b).abs().mean().mean()

def compute_accuracy(a: pd.DataFrame, b: pd.DataFrame) -> float:
	return (a == b).mean().mean() * 100

if __name__ == '__main__':
	# Whether noise is being injected into the data or not
	data_gen.args.noise = False
	# RNG seed to pass as a parameter to data_gen functions
	rng = random.Random(data_gen.args.seed)
	# Data inputs and outputs
	inputs = []
	outputs = []

	# Create batch of data for each profile type in data_gen
	profile_keys = data_gen.PROFILES.keys()
	for profile_name in profile_keys:
		# Creates rows of data using a certain profile
		for _ in range(data_gen.args.samples):
			# Get 1 row of data (inputs and outputs)
			i, o = data_gen.generate_sample(rng, profile_name = profile_name)
			# Inject noise if desired
			if data_gen.args.noise:
				i.inject_noise(noise_type = data_gen.args.noise_type, level = data_gen.args.noise_level, rng = rng)
			# Add row of data to input and output lists
			inputs.append(i.to_list())
			outputs.append(o.to_list())

	# Column names for data
	input_column_names = ['region', 'session_id', 'packets', 'avg_size', 'avg_time']
	output_column_names = ['age', 'gender', 'condition', 'device_count']

	# Split data into train and validation data (training for training the nn, validation for testing it to see how well it works)
	dataframe = pd.DataFrame(np.c_[inputs, outputs], columns = input_column_names + output_column_names)
	train_dataframe = dataframe.sample(frac = 0.75, random_state = randint(0, 2**32 - 1))
	validate_dataframe = dataframe.drop(train_dataframe.index)

	# Split the data back into inputs and outputs
	input_train_data = train_dataframe.drop(output_column_names, axis = 1)
	output_train_data = train_dataframe.drop(input_column_names, axis = 1)
	input_validate_data = validate_dataframe.drop(output_column_names, axis = 1)
	output_validate_data = validate_dataframe.drop(input_column_names, axis = 1)

	# Normalize the inputs (make sure they're all between 0 and 1 for math reasons)
	combined_inputs = pd.concat([input_train_data, input_validate_data])
	min_vals = combined_inputs.min(axis = 0)
	max_vals = combined_inputs.max(axis = 0)
	ranges = max_vals - min_vals
	input_train_data = (input_train_data - min_vals) / ranges
	input_validate_data = (input_validate_data - min_vals) / ranges

	# Makes sure the neural network has the right number of neurons for the input and output layers for the data to be fed in
	input_shape = [input_train_data.shape[1]]
	output_shape = output_train_data.shape[1]

	# Create the model layers (3 regular dense perceptron layers)
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
	# Create the model optimizers (the algorithm that improves it after each training cycle)
	# Larger learning rate means it improves faster, but less precisely (takes fewer epochs (less time) to get better, but won't be as good when it's done)
	# Smaller learning rate means it improves slower, but more precisely (takes more epochs (more time) to get better, but will be more accurate when it's done)
	# Default learning rate for Adam optimizer is 0.001
	# Train a bunch of times at the start with a fast learning rate
	optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1)
	# Loss function determines how good the functions is doing / how off it was from predicting something
	# This loss function is Mean Absolute Error (MAE)
	loss_function = 'mae'
	# Make the model ready to train and predict
	model.compile(optimizer = optimizer, loss = loss_function)

	# Train the model on the data
	# Give it validation data as well to see its loss score (how well it performs) after each epoch (training cycle)
	# Change the epochs to have it do more or less training cycles
	model.fit(input_train_data, output_train_data, validation_data = (input_validate_data, output_validate_data), batch_size = input_train_data.shape[0], epochs = 300)

	# Train more with a lower learning rate to get more precise
	optimizer = tf.keras.optimizers.Adam(learning_rate = 0.05)
	model.compile(optimizer = optimizer, loss = loss_function)
	model.fit(input_train_data, output_train_data, validation_data = (input_validate_data, output_validate_data), batch_size = input_train_data.shape[0], epochs = 300)

	# Train more with an even lower learning rate to get even more precise
	optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
	model.compile(optimizer = optimizer, loss = loss_function)
	model.fit(input_train_data, output_train_data, validation_data = (input_validate_data, output_validate_data), batch_size = input_train_data.shape[0], epochs = 400)

	
	overall_loss = model.evaluate(input_validate_data, output_validate_data, batch_size = input_validate_data.shape[0])
	predictions = np.round(model.predict(input_validate_data))

	# Ensure predictions are the correct shape, the same shape as the actual outputs
	if (predictions.shape != output_validate_data.shape):
		print(f'ERROR: Predictions & actual values aren\'t the same shape. Predictions: {predictions.shape}, Actual: {output_validate_data.shape}')
		exit(1)
	
	# Show 3 example outputs from the model from the validation data juxtaposed by the actual value
	print('Sample results:')
	print_bar()
	for row in range(3):
		prediction = predictions[row]
		actual = output_validate_data.iloc[row].to_numpy()
		print(f'Prediction:\t{prediction}')
		print(f'Actual:\t\t{actual}')
		print_bar()
	# Show the mode's overall loss performance
	print(f'Overall model loss performance: {overall_loss}')

	predictions_dataframe = pd.DataFrame(predictions, columns = output_column_names)
	actual_values = output_validate_data.reset_index(drop = True)

	age_predictions = extract_dataframe_column(predictions_dataframe, 'age')
	gender_predictions = extract_dataframe_column(predictions_dataframe, 'gender')
	condition_predictions = extract_dataframe_column(predictions_dataframe, 'condition')
	device_count_predictions = extract_dataframe_column(predictions_dataframe, 'device_count')

	actual_ages = extract_dataframe_column(actual_values, 'age')
	actual_genders = extract_dataframe_column(actual_values, 'gender')
	actual_conditions = extract_dataframe_column(actual_values, 'condition')
	actual_device_counts = extract_dataframe_column(actual_values, 'device_count')

	age_mae_loss = compute_mae(age_predictions, actual_ages)
	gender_accuracy = compute_accuracy(gender_predictions, actual_genders)
	condition_accuracy = compute_accuracy(condition_predictions, actual_conditions)
	device_count_mae_loss = compute_mae(device_count_predictions, actual_device_counts)

	print(f'Age loss:\t\t{age_mae_loss}')
	print(f'Gender accuracy:\t{gender_accuracy}%')
	print(f'Condition accuracy:\t{condition_accuracy}%')
	print(f'Device count loss:\t{device_count_mae_loss}')
