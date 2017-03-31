import tensorflow as tf

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 16,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("from_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("to_vocab_size", 40000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/home/rob/WMT", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/home/rob/WMT", "Training directory.")
tf.app.flags.DEFINE_string("from_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("from_dev_data", None, "Validation data.")
tf.app.flags.DEFINE_string("to_dev_data", None, "Validation data.")
tf.app.flags.DEFINE_integer("max_train_data_size", 100000,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 300,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("loss_increases_per_decay", 2,
                            "The learning rate will decay if the loss is greater than the max of the last (this many) checkpoint losses.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")




#Embedding Flags for Encoder and Decoder
#Was 1024
tf.app.flags.DEFINE_integer("encoder_embedding_size", 512,
                            "Number of units in the embedding size of the encoder inputs. This will be used in a wrapper to the first layer")
#Was 1024
tf.app.flags.DEFINE_integer("decoder_embedding_size", 512,
                            "Number of units in the embedding size of the encoder inputs. This will be used in a wrapper to the first layer")



#Was 1024
#LSTM Flags for Encoder and Decoder
tf.app.flags.DEFINE_integer("encoder_hidden_size", 512,
                            "Number of units in the hidden size of encoder LSTM layers. Bidirectional layers will therefore output 2*this")
#Was 1024
tf.app.flags.DEFINE_integer("decoder_hidden_size", 512,
                            "Number of units in the hidden size of decoder LSTM layers. Bidirectional layers will therefore output 2*this")

tf.app.flags.DEFINE_boolean("encoder_use_peepholes", False,
                            "Whether or not to use peephole (diagonal) connections on LSTMs in the encoder")
tf.app.flags.DEFINE_boolean("decoder_use_peepholes", False,
                            "Whether or not to use peephole (diagonal) connections on LSTMs in the decoder")

tf.app.flags.DEFINE_float("encoder_init_forget_bias", 0.0,
                            "The initial forget bias (0-1) to use for LSTMs in the encoder")
tf.app.flags.DEFINE_float("decoder_init_forget_bias", 0.0,
                            "The initial forget bias (0-1) to use for LSTMs in the decoder")

tf.app.flags.DEFINE_float("encoder_dropout_keep_probability", 1.0,
                            "The keep probability to use in the dropout wrapper for LSTM's in the encoder. To disable dropout, just use 1.0")
tf.app.flags.DEFINE_float("decoder_dropout_keep_probability", 1.0,
                            "The keep probability to use in the dropout wrapper for LSTM's in the decoder. To disable dropout, just use 1.0")


def flag_test():
    pass