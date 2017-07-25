import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers.python.layers import initializers

class GRUCell(RNNCell):
	"""custom GRU cell class that implements tensorflow functionality, with additional parameters that may be tuned by the model
	http://arxiv.org/abs/1406.1078

	Args:
	num_units - integer, the hidden size of the GRU
	init_reset_update_bias - the initial bias of the gru reset update gates. 1.0 means no reset, no update
	_init_candidate_bias - the bias applied to the linear transformation fully connected layer calculation of inputs dot reset*state
	initializer - the initialization function for the weights of the gru. defaults to xavier
	activation - tensorflow activation function. NOT the activation for the reset and update gates, which are always sigmoid

	Returns:
	 A tuple of new_hidden_state, new_hidden_state
	"""
	def __init__(self,
				num_units,
				init_reset_update_bias=1.0, #1.0 = no reset, no update
				init_candiate_bias=0.0,
				initializer=initializers.xavier_initializer(),
				activation=tanh):

		self._num_units = num_units
		self._init_reset_update_bias = init_reset_update_bias
		self._init_candidate_bias = init_candiate_bias
		self._initializer = initializer
		self._activation = activation

	@property
	def state_size(self):
		return self._num_units

	@property
	def output_size(self):
		return self._num_units

	def __call__(self, inputs, state, scope=None):

		with vs.variable_scope(scope or "gru_cell"):
			with vs.variable_scope("gates"):

				#we pass this off to a fully connected layer and specify our initializers and bias for the gate parameters. 
				res, upd = array_ops.split(
						val = fully_connected([inputs, state],
												2 * self._num_units,
												activation_fn=None,
												weights_initializer=self._initializer,
												biases_initializer=init_ops.constant_initializer(self._init_reset_update_bias),
												scope=scope)
						num_or_size_splits=2,
						axis=1)

				res = sigmoid(res)
				upd = sigmoid(upd)

			with vs.variable_scope("candidate"):
				cand = fully_connected([inputs, res * state],
										self._num_units,
										activation_fn=self._activation,
										weights_initializer=self._initializer,
										biases_initializer=init_ops.constant_initializer(self._init_candiate_bias), #could use another bias parameter here
										scope=scope)

			new_hidden_state = upd * state + (1 - upd) * cand

		return new_hidden_state, new_hidden_state