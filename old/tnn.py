# Source Citation: Used transformer neural network example from GeeksforGeeks
# https://www.geeksforgeeks.org/deep-learning/transformer-model-from-scratch-using-tensorflow/
# Made modifications to it to fit the noise injection experiment

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, LayerNormalization
from tensorflow.keras.models import Model
import numpy as np

# Gives an encoding value for a position / index value
def positional_encoding(position, d_model):
	angle_radians = np.arange(position)[:, np.newaxis] / np.power(10_000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))
	angle_radians[:, 0::2] = np.sin(angle_radians[:, 0::2])
	angle_radians[:, 1::2] = np.cos(angle_radians[:, 1::2])
	return tf.cast(angle_radians[np.newaxis, ...], dtype=tf.float32)

# First part of a transformer block
# Calculates the amount of attention to pay to a certain part of the input
# Does this by calculating a context vector based on inputs
class MultiHeadAttention(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads):
		super(MultiHeadAttention, self).__init__()
		self.num_heads = num_heads
		self.d_model = d_model
		assert d_model % num_heads == 0
		self.depth = d_model // num_heads
		# Weight values for queries, keys, and values
		self.wq = Dense(d_model)
		self.wk = Dense(d_model)
		self.wv = Dense(d_model)
		# Last layer of this part of the model (basic perceptron layer)
		self.dense = Dense(d_model)
	
	# Splits inputs into multiple heads
	# Output tensor will have shape (batch_size, num_heads, seq_len, depth)
	def split_heads(self, x, batch_size):
		x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
		return tf.transpose(x, perm=[0, 2, 1, 3])
	
	# Calculates the attention values
	def scaled_dot_product_attention(self, q, k, v, mask):
		matmul_qk = tf.matmul(q, k, transpose_b=True)
		dk = tf.cast(tf.shape(k)[-1], tf.float32)
		scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

		if mask is not None:
			scaled_attention_logits += (mask * -1e9)
		
		attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
		output = tf.matmul(attention_weights, v)
		return output, attention_weights

	# Takes input tensors (v, k, and q), feeds them through this layer, and gives this layer's output
	def call(self, v, k, q, mask):
		batch_size = tf.shape(q)[0]
		q = self.wq(q)
		k = self.wk(k)
		v = self.wv(v)
		q = self.split_heads(q, batch_size)
		k = self.split_heads(k, batch_size)
		v = self.split_heads(v, batch_size)

		attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
		attention = tf.transpose(attention, perm=[0, 2, 1, 3])
		attention = tf.reshape(attention, (batch_size, -1, self.d_model))
		output = self.dense(attention)
		return output

# Second part of transformer block
# Basic neural network layers with position data from encoder
class PositionwiseFeedforward(tf.keras.layers.Layer):
	def __init__(self, d_model, dff):
		super(PositionwiseFeedforward, self).__init__()
		self.d_model = d_model
		self.dff = dff
		self.dense1 = Dense(dff, activation='relu')
		self.dense2 = Dense(d_model)
	
	# Takes input tensor, feeds it through this layer, and gives this layer's output
	def call(self, x):
		x = self.dense1(x)
		x = self.dense2(x)
		return x

# Made up of a Multi-headed attention layer, and a position-wise feed forward layer
# Basic building unit of a transformer model, used to build encoders and decoders
class TransformerBlock(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
		super(TransformerBlock, self).__init__()
		self.att = MultiHeadAttention(d_model, num_heads)
		self.ffn = PositionwiseFeedforward(d_model, dff)
		self.layernorm1 = LayerNormalization(epsilon=1e-6)
		self.layernorm2 = LayerNormalization(epsilon=1e-6)
		self.dropout1 = Dropout(dropout_rate)
		self.dropout2 = Dropout(dropout_rate)
	
	# Takes input tensor, feeds it through this layer, and gives this layer's output
	def call(self, x, training, mask):
		# Masked self-attention (look-ahead)
		attn_output = self.att(x, x, x, mask)
		attn_output = self.dropout1(attn_output, training=training)
		out1 = self.layernorm1(x + attn_output)
		# Feedforward
		ffn_output = self.ffn(out1)
		ffn_output = self.dropout2(ffn_output, training=training)
		out2 = self.layernorm2(out1 + ffn_output)
		return out2


# Decoder block with masked self-attention followed by encoder-decoder cross-attention
class DecoderBlock(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
		super(DecoderBlock, self).__init__()
		self.att1 = MultiHeadAttention(d_model, num_heads)  # masked self-attention
		self.att2 = MultiHeadAttention(d_model, num_heads)  # encoder-decoder attention
		self.ffn = PositionwiseFeedforward(d_model, dff)

		self.layernorm1 = LayerNormalization(epsilon=1e-6)
		self.layernorm2 = LayerNormalization(epsilon=1e-6)
		self.layernorm3 = LayerNormalization(epsilon=1e-6)

		self.dropout1 = Dropout(dropout_rate)
		self.dropout2 = Dropout(dropout_rate)
		self.dropout3 = Dropout(dropout_rate)

	# Takes input tensor, feeds it through this layer, and gives this layer's output
	def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
		# Masked self-attention (look-ahead)
		attn1 = self.att1(x, x, x, look_ahead_mask)
		attn1 = self.dropout1(attn1, training=training)
		out1 = self.layernorm1(x + attn1)
		# Cross-attention with encoder output (keys/values from encoder, queries from decoder)
		attn2 = self.att2(enc_output, enc_output, out1, padding_mask)
		attn2 = self.dropout2(attn2, training=training)
		out2 = self.layernorm2(out1 + attn2)
		# Feedforward
		ffn_output = self.ffn(out2)
		ffn_output = self.dropout3(ffn_output, training=training)
		out3 = self.layernorm3(out2 + ffn_output)

		return out3

# Converts inputs sequence into a set of embeddings with positional information
# Made of embedding layer, dropout layer, and a list of transformer blocks
class Encoder(tf.keras.layers.Layer):
	def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, dropout_rate=0.1):
		super(Encoder, self).__init__()
		self.d_model = d_model
		self.num_layers = num_layers
		self.embedding = Embedding(input_vocab_size, d_model)
		self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
		self.dropout = Dropout(dropout_rate)
		self.enc_layers = [TransformerBlock(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
	
	# Takes input tensor, feeds it through this layer, and gives this layer's output
	def call(self, x, training, mask):
		seq_len = tf.shape(x)[1]
		x = self.embedding(x)
		x += self.pos_encoding[:, :seq_len, :]
		x = self.dropout(x, training=training)
		for i in range(self.num_layers):
			x = self.enc_layers[i](x, training=training, mask=mask)
		return x

# Generates output sequence from encoder outputs and previously generated output sequences
# Made of embedding layer, dropout layer, and a list of transformer blocks
class Decoder(tf.keras.layers.Layer):
	def __init__(self, num_layers, d_model, num_heads, dff, taret_vocab_size, maximum_position_encoding, dropout_rate=0.1):
		super(Decoder, self).__init__()
		self.d_model = d_model
		self.num_layers = num_layers
		self.embedding = Embedding(taret_vocab_size, d_model)
		self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
		self.dropout = Dropout(dropout_rate)
		# Use DecoderBlock (masked self-attn + cross-attn)
		self.dec_layers = [DecoderBlock(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
	
	# Takes input tensor, feeds it through this layer, and gives this layer's output
	def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
		seq_len = tf.shape(x)[1]
		x = self.embedding(x)
		x += self.pos_encoding[:, :seq_len, :]
		x = self.dropout(x, training=training)
		for i in range(self.num_layers):
			x = self.dec_layers[i](x, enc_output, training=training, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)

		return x

# An encoder, decoder, and a final basic perceptron layer
class Transformer(tf.keras.Model):
	def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, maximum_position_encoding, dropout_rate=0.1):
		super(Transformer, self).__init__()
		self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, dropout_rate)
		self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, dropout_rate)
		self.final_layer = Dense(target_vocab_size)
	
	def call(self, inputs, training=False, look_ahead_mask=None, padding_mask=None):
		inp, tar = inputs
		enc_output = self.encoder(inp, training=training, mask=padding_mask)
		dec_output= self.decoder(tar, enc_output, training=training, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
		final_output = self.final_layer(dec_output)

		return final_output
