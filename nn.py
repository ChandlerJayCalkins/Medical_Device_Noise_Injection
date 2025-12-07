import random
import data_gen
import numpy as np
import pandas as pd
import tensorflow as tf
from random import randint
from typing import Tuple

def print_bar():
	print('====================')

# Returns a dataframe of a single column from another dataframe
def extract_dataframe_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
	return df.drop(df.columns.difference([column_name]), axis = 1)

# Returns the mean average error loss between 2 dataframes
def compute_mae(a: pd.DataFrame, b: pd.DataFrame) -> float:
	return (a - b).abs().mean().mean()

# Returns the accuracy between 2 dataframes (percentage of values that are the same)
def compute_accuracy(a: pd.DataFrame, b: pd.DataFrame) -> float:
	return (a == b).mean().mean() * 100

# Trains a neural network and evaluates it
# Takes parameters for how to inject noise into data neural network trains on
# noise determines whether or not to even inject noise into the data
# noise_type determines the kind of noise (either "gaussian" or "uniform")
# noise level determines the relative noise level as a percentage between 0 and 1 (0.1 = ~10%)
# Returns:
# Overall model loss, age loss, gender accuracy (0% - 100%), condition accuracy (0% - 100%), and device count loss
def test_nn(noise: bool, noise_type: str, noise_level: float, verbose: bool = True) -> Tuple[float, float, float, float, float]:

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
			if noise:
				i.inject_noise(noise_type = noise_type, level = noise_level, rng = rng)
			# Add row of data to input and output lists
			inputs.append(i.to_list())
			outputs.append(o.to_list())

	# Column names for data
	input_column_names = ['region', 'session_id', 'packets', 'avg_size', 'avg_time']
	output_column_names = ['age', 'gender', 'condition', 'device_count']

	# Split data into train and validation data (training for training the nn, validation for testing it to see how well it works)
	dataframe = pd.DataFrame(np.c_[inputs, outputs], columns = input_column_names + output_column_names)
	train_dataframe = dataframe.sample(frac = 0.75)
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
	# Define learning rate schedule (how quickly the model learns)
	# Larger learning rate means it improves faster, but less precisely (takes fewer epochs (less time) to get better, but won't be as good when it's done)
	# Smaller learning rate means it improves slower, but more precisely (takes more epochs (more time) to get better, but will be more accurate when it's done)
	# Start with a large learning rate to approach optimal performance quicker at the start, and decrease it over time to get more precise
	learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.05, decay_steps = 10000, decay_rate = 0.9)
	# Create the model optimizers (the algorithm that improves it after each training cycle)
	# Uses the learning rate schedule to scale the optimizations
	optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_schedule)
	# Loss function determines how good the functions is doing / how off it was from predicting something
	# This loss function is Mean Absolute Error (MAE)
	loss_function = 'mae'
	# Make the model ready to train and predict
	model.compile(optimizer = optimizer, loss = loss_function)

	# Train the model
	# Change the epochs value to change the number of training cycles the model goes through
	model.fit(input_train_data, output_train_data, validation_data = (input_validate_data, output_validate_data), batch_size = input_train_data.shape[0], epochs = 1000)
	
	# Evaluates the average loss score for all the model's outputs (not super useful but gives a general idea of how accurate the model is)
	overall_loss = model.evaluate(input_validate_data, output_validate_data, batch_size = input_validate_data.shape[0])
	# Get predictions from the model for the validation data to see how well it did
	predictions = np.round(model.predict(input_validate_data))

	# Ensure predictions are the correct shape, the same shape as the actual outputs
	if (predictions.shape != output_validate_data.shape):
		print(f'ERROR: Predictions & actual values aren\'t the same shape. Predictions: {predictions.shape}, Actual: {output_validate_data.shape}')
		exit(1)
	
	if verbose:
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

	# Turn predictions into dataframe for getting evaluation metrics for each individual column
	predictions_dataframe = pd.DataFrame(predictions, columns = output_column_names)
	# Reset indexes in actual outputs so they're the same as the predictions dataframe, allowing them to do comparisons with each other in compute_accuracy()
	actual_values = output_validate_data.reset_index(drop = True)

	# Get columns of each of the predicted values
	age_predictions = extract_dataframe_column(predictions_dataframe, 'age')
	gender_predictions = extract_dataframe_column(predictions_dataframe, 'gender')
	condition_predictions = extract_dataframe_column(predictions_dataframe, 'condition')
	device_count_predictions = extract_dataframe_column(predictions_dataframe, 'device_count')

	# Get columns of each of the actual values
	actual_ages = extract_dataframe_column(actual_values, 'age')
	actual_genders = extract_dataframe_column(actual_values, 'gender')
	actual_conditions = extract_dataframe_column(actual_values, 'condition')
	actual_device_counts = extract_dataframe_column(actual_values, 'device_count')

	# Compute evaluation metrics for each column
	age_mae_loss = compute_mae(age_predictions, actual_ages)
	gender_accuracy = compute_accuracy(gender_predictions, actual_genders)
	condition_accuracy = compute_accuracy(condition_predictions, actual_conditions)
	device_count_mae_loss = compute_mae(device_count_predictions, actual_device_counts)

	if verbose:
		# Show evaluation metrics for each column
		print(f'Age loss:\t\t{age_mae_loss}')
		print(f'Gender accuracy:\t{gender_accuracy}%')
		print(f'Condition accuracy:\t{condition_accuracy}%')
		print(f'Device count loss:\t{device_count_mae_loss}')

	return overall_loss, age_mae_loss, gender_accuracy, condition_accuracy, device_count_mae_loss

# Calls test_nn() test_cycles times and returns the average of the results from all those tests
# test_cycles is the number of times to train and test the neural network
# noise, noise_type, and noise_level determine the parameters for noise injection in the data the network trains on
# Returns averages of:
# Overall model loss, age loss, gender accuracy (0% - 100%), condition accuracy (0% - 100%), and device count loss
def trial(test_cycles: int, noise: bool, noise_type: str, noise_level: float) -> Tuple[float, float, float, float, float]:
	# Results to return
	average_overall_loss = 0
	average_age_mae_loss = 0
	average_gender_accuracy = 0
	average_condition_accuracy = 0
	average_device_count_mae_loss = 0

	# Runs the neural network test_cycles times and adds the results to the total each time
	for _ in range(test_cycles):
		overall_loss, age_mae_loss, gender_accuracy, condition_accuracy, device_count_mae_loss = test_nn(noise, noise_type, noise_level)
		average_overall_loss += overall_loss
		average_age_mae_loss += age_mae_loss
		average_gender_accuracy += gender_accuracy
		average_condition_accuracy += condition_accuracy
		average_device_count_mae_loss += device_count_mae_loss
	
	# Divide the total by the number of times it was run to get the average
	average_overall_loss /= test_cycles
	average_age_mae_loss /= test_cycles
	average_gender_accuracy /= test_cycles
	average_condition_accuracy /= test_cycles
	average_device_count_mae_loss /= test_cycles

	return average_overall_loss, average_age_mae_loss, average_gender_accuracy, average_condition_accuracy, average_device_count_mae_loss

# Writes the a single accuracy metric from the experiment to a file
def write_result(file, label: str, result: float):
	file.write(f'{label}{result}\n')

# Write the results of an entire trial to a file
def write_results(file, average_overall_loss: float, average_age_mae_loss: float, average_gender_accuracy: float, average_condition_accuracy: float, average_device_count_mae_loss: float):
	write_result(file, 'Average overall loss:\t\t\t', average_overall_loss)
	write_result(file, 'Average age loss (mae):\t\t\t', average_age_mae_loss)
	write_result(file, 'Average gender accuracy:\t\t', average_gender_accuracy)
	write_result(file, 'Average condition accuracy:\t\t', average_condition_accuracy)
	write_result(file, 'Average device count loss (mae):\t', average_device_count_mae_loss)

# Run the control trial (no noise injection) and write the results to a file
# result_file is the file to write the results to
# test_cycles is the number of times to run the control trial
def control_trial(result_file: str, test_cycles: int):
	average_overall_loss, average_age_mae_loss, average_gender_accuracy, average_condition_accuracy, average_device_count_mae_loss = trial(test_cycles, False, '', 0.0)
	with open(result_file, 'w') as file:
		file.write(f'Control Trial Results ({test_cycles} training cycles):\n\n')
		write_results(file, average_overall_loss, average_age_mae_loss, average_gender_accuracy, average_condition_accuracy, average_device_count_mae_loss)

# Run a single experimental trial with certain noise parameters and write the results to a file
# result_file is the file to write the results to
# test_cycles is the number of times to run the control trial
# noise_type is the type of noise to inject (gaussian or uniform)
# noise level is the amount of noise to inject (percentage value on scale from 0 to 1 being 0% to 100%)
def experimental_trial(result_file: str, test_cycles: int, noise_type: str, noise_level: float):
	average_overall_loss, average_age_mae_loss, average_gender_accuracy, average_condition_accuracy, average_device_count_mae_loss = trial(test_cycles, True, noise_type, noise_level)
	with open(result_file, 'a') as file:
		file.write('==========\n')
		file.write(f'Noise type: {noise_type} | Noise level: {noise_level}\n\n')
		write_results(file, average_overall_loss, average_age_mae_loss, average_gender_accuracy, average_condition_accuracy, average_device_count_mae_loss)

# Run all experimental trials (noise injection with various parameters) and write the results to a file
# result_file is the file to write the results to
# test_cycles is the number of times to run the control trial
def experimental_trials(result_file: str, test_cycles: int):
	with open(result_file, 'w') as file:
		file.write('Experimental Trial Results:\n\n')
	noise_types = ['gaussian', 'uniform']
	for noise_type in noise_types:
		for i in range(1, 11):
			noise_level = i / 10
			experimental_trial(result_file, test_cycles, noise_type, noise_level)

if __name__ == '__main__':
	# Warning: Takes ~7 hours to run on a beefy PC!
	# On my pc, it takes about 1 minute to do 1 training cycle
	# That means it takes about test_cycles * 21 minutes to do the entire experiment
	# Or (minutes / 60) hours
	test_cycles = 20
	control_trial('control_trial_results.txt', test_cycles)
	experimental_trials('experimental_trial_results.txt', test_cycles)
