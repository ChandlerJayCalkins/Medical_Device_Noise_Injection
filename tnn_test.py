from tnn import *
import logging
import time
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow_text

MAX_TOKENS=128
BUFFER_SIZE = 20000
BATCH_SIZE = 64

def prepare_batch(pt, en):
		pt = tokenizers.pt.tokenize(pt)	# Output is ragged.
		pt = pt[:, :MAX_TOKENS]			# Trim to MAX_TOKENS.
		pt = pt.to_tensor()				# Convert to 0-padded dense Tensor

		en = tokenizers.en.tokenize(en)
		en = en[:, :(MAX_TOKENS+1)]
		en_inputs = en[:, :-1].to_tensor()	# Drop the [END] tokens
		en_labels = en[:, 1:].to_tensor()	 # Drop the [START] tokens

		return (pt, en_inputs), en_labels

def make_batches(ds):
	return (
		ds
		.shuffle(BUFFER_SIZE)
		.batch(BATCH_SIZE)
		.map(prepare_batch, tf.data.AUTOTUNE)
		.prefetch(buffer_size=tf.data.AUTOTUNE))


def masked_loss(label, pred):
	mask = label != 0
	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
	loss = loss_object(label, pred)

	mask = tf.cast(mask, dtype=loss.dtype)
	loss *= mask

	loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
	return loss


def masked_accuracy(label, pred):
	pred = tf.argmax(pred, axis=2)
	label = tf.cast(label, pred.dtype)
	match = label == pred

	mask = label != 0

	match = match & mask

	match = tf.cast(match, dtype=tf.float32)
	mask = tf.cast(mask, dtype=tf.float32)
	return tf.reduce_sum(match)/tf.reduce_sum(mask)

if __name__ == '__main__':
	num_layers = 4
	d_model = 128
	dff = 512
	num_heads = 8
	dropout_rate = 0.1
	input_vocab_size = 50
	target_vocab_size = 50

	examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
	train_examples, val_examples = examples['train'], examples['validation']

	for pt_examples, en_examples in train_examples.batch(3).take(1):
		print('> Examples in Portuguese:')
		for pt in pt_examples.numpy():
			print(pt.decode('utf-8'))
		print()

		print('> Examples in English:')
		for en in en_examples.numpy():
			print(en.decode('utf-8'))

	model_name = 'ted_hrlr_translate_pt_en_converter'
	# Used this to download test data set
	# tf.keras.utils.get_file(
	# 	f'{model_name}.zip',
	# 	f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
	# 	cache_dir='.', cache_subdir='', extract=True
	# )
	tokenizers = tf.saved_model.load(model_name)

	# Create training and validation set batches.
	train_batches = make_batches(train_examples)
	val_batches = make_batches(val_examples)

	learning_rate = CustomSchedule(d_model)
	optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

	transformer = Transformer(
		num_layers=num_layers,
		d_model=d_model,
		num_heads=num_heads,
		dff=dff,
		input_vocab_size=input_vocab_size,
		target_vocab_size=target_vocab_size,
		dropout_rate=dropout_rate)

	transformer.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])
	transformer.fit(train_batches, epochs=20, validation_data=val_batches)