if __name__ == '__main__':
	from old.tnn import *

	num_layers = 4
	d_model = 128
	num_heads = 8
	dff = 512
	input_vocab_size = 8_500
	target_vocab_size = 8_000
	maximum_position_encoding = 10_000
	dropout_rate = 0.1

	batch_size = 128
	num_epochs = 10

	transformer = Transformer \
	(
		num_layers,
		d_model,
		num_heads,
		dff,
		input_vocab_size,
		target_vocab_size,
		maximum_position_encoding,
		dropout_rate
	)
	transformer.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

	# Create input and target sequences
	inputs = tf.random.uniform((64, 50), dtype=tf.int32, minval=0, maxval=input_vocab_size)
	targets = tf.random.uniform((64, 50), dtype=tf.int32, minval=0, maxval=target_vocab_size)

	# Prepare shifted targets for teacher forcing
	# tar_inp is fed to the decoder, tar_real is the prediction values
	tar_inp = targets[:, :-1]
	tar_real = targets[:, 1:]

	look_ahead_mask = None
	padding_mask = None

	transformer.fit(x=(inputs, tar_inp), y=tar_real, epochs=num_epochs)

	output_logits = transformer((inputs, tar_inp), training=False, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
	output = tf.argmax(output_logits, axis=-1)

	print('Targets (shifted):')
	print(tar_real)
	print('==========')
	print('Outputs:')
	print(output)
