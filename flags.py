import tensorflow as tf
import os
import sys


##===================Gradients and learning rate===================================
#TODO - make this trainable and saveable so it doesn't reload at 0.5
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")

tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")

tf.app.flags.DEFINE_float("minimum_learning_rate", 0.005, "Minimum learning rate")

tf.app.flags.DEFINE_float("max_clipped_gradient", 5.0,
                          "Clip gradients to a maximum of this this norm.")
#==========================================================================================





#this should probably be a larger number than 16
tf.app.flags.DEFINE_integer("batch_size", 16,
                            "Batch size to use during training.")

tf.app.flags.DEFINE_integer("from_vocab_size", 40000, "Source language vocabulary size.")

tf.app.flags.DEFINE_integer("to_vocab_size", 40000, "Target language vocabulary size.")

tf.app.flags.DEFINE_string("data_dir", "/home/rob/WMT", "Directory where we will store the data as well as model checkpoint")

tf.app.flags.DEFINE_string("from_train_data", None, "Training data.")

tf.app.flags.DEFINE_string("to_train_data", None, "Training data.")

tf.app.flags.DEFINE_string("from_dev_data", None, "Validation data.")

tf.app.flags.DEFINE_string("to_dev_data", None, "Validation data.")

tf.app.flags.DEFINE_integer("loss_increases_per_decay", 3,
                            "The learning rate will decay if the loss is greater than the max of the last (this many) checkpoint losses.")

tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")

tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")

tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")


#Checkpoint Flags
tf.app.flags.DEFINE_string("checkpoint_name", "translate.ckpt", "Name of the Tensorflow checkpoint file")

tf.app.flags.DEFINE_integer("steps_per_checkpoint", 300,
                            "How many training steps to do per checkpoint.")


#Bucket Sizes
tf.app.flags.DEFINE_boolean("use_default_buckets", True, "Use the default bucket sizes defined in bucket_utils. Otherwise, will try candidate bucket sizes defined in bucket_utils")
tf.app.flags.DEFINE_float("minimum_data_ratio_per_bucket", 0.1, "Each one of the buckets must have at least this ratio of the data in it.")
tf.app.flags.DEFINE_integer("bucket_inference_sample_size", 1000000, "Only check this many data samples when trying to find the best distribution of the data for buckets")
tf.app.flags.DEFINE_integer("num_buckets", 3, "Will use this number to look up default bucket sizes, otherwise use this number to find good division of data with this many buckets")



#Dataset Flags
tf.app.flags.DEFINE_boolean("load_train_set_in_memory", True,
                            "If True, loads training set into memory. Otherwise, reads batches by opening files and reading appropriate lines.")

tf.app.flags.DEFINE_integer("max_train_data_size", 200000,
                            "Limit on the size of training data (0: no limit).")

tf.app.flags.DEFINE_integer("train_offset", 0,
                            "ignore the first train_offset lines of the training file when loading the training set or getting randomly")
#==========================================================================================



#Embedding Flags for Encoder and Decoder
#Was 1024, 512


tf.app.flags.DEFINE_string("embedding_algorithm", "glove",
                            "glove, fasttext, word2vec, or network. The first three are unsupervised trainers implemented by other programs. the latter is a network layer trained only by backprop")

tf.app.flags.DEFINE_boolean("train_embeddings", True,
                            "Whether or not to continue training the glove embeddings from backpropagation or to leave them be")







tf.app.flags.DEFINE_integer("encoder_embedding_size", 200,
                            "Number of units in the embedding size of the encoder inputs. This will be used in a wrapper to the first layer")
#Was 1024, 512
tf.app.flags.DEFINE_integer("decoder_embedding_size", 512,
                            "Number of units in the embedding size of the encoder inputs. This will be used in a wrapper to the first layer")

tf.app.flags.DEFINE_string("glove_embedding_file", "GloVe/build/rob_vectors_50it.txt",
                            "The output file for Glove-trained word embeddings on the dataset.")


#==========================================================================================


#Was 1024, 512
#LSTM Flags for Encoder and Decoder
tf.app.flags.DEFINE_integer("encoder_hidden_size", 512,
                            "Number of units in the hidden size of encoder LSTM layers. Bidirectional layers will therefore output 2*this")
#Was 1024
tf.app.flags.DEFINE_integer("decoder_hidden_size", 512,
                            "Number of units in the hidden size of decoder LSTM layers. Bidirectional layers will therefore output 2*this")


#decoder_state_initializer's mechanism will depend on the implementation. there are a few different ways of doing it
# essentially, you need to figure out what to do to the decoder state given the last encoder state, either just the top layer of it
# or all layers of it. there are a few options

# 1. "mirror" - take the final state of every layer in the encoder and apply it as the first state of the decoder
#             - THIS ONLY WORKS if you have the exact same number of layers, bidirectional layers, and layer sizes, between encoder and decoder
#
# 2. "top_layer_mirror" - take the final state of the top layer of the encoder and use it for the state of the decoder
#                         at each layer. THIS ONLY WORKS if you have the exact same layer size on the decoder as the top layer
#
# 3. "bahdanu" - take the final state of the top layer of the encoder. if bidirectional, use the BACKWARD STATE ONLY
#                this top state is then multiplied by a trained weight vector with size equal to the sum of the hidden
#                size of each layer of the decoder. finally, this is passed through a hyperbolic tangent.
#                this way, we can create a list of initial states to use in the decoder.
#                this approach will add a trainable weight vector to the parameters.
#
# 4. "nematus" - similar to bahdanu, except using a mean annotation of the hidden states across all time steps of the
#                encoder, not just the final one, multiplied by the trained weight parameter. This does not use the backward direction only, but both
#                concatenated in the event that the top layer of the encoder is bidirectional.
#                this is passed through a hyperbolic tangent as well. 
#
tf.app.flags.DEFINE_string("decoder_state_initializer", "top_layer_mirror",
                           "The metric used to calculate initial state values for the decoder from the encoder")




#This is where we store the JSON of the encoder and decoder architecture
#See the file encoder_decoder_architecture_help.txt for more information
tf.app.flags.DEFINE_string("encoder_decoder_architecture_json", 'encoder_decoder_architecture.json',
                            "The file name which stores the JSON architecture of the encoder and decoder")


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

#TODO - Flesh this out
def flag_test():
    f = tf.app.flags.FLAGS
    #build flag tests for the rest of the flags for input checking. like this...
    assert f.encoder_dropout_keep_probability <= 1.0 and f.encoder_dropout_keep_probability >= 0.0, "Encoder dropout keep probability must be between 0 and 1"
    assert f.decoder_dropout_keep_probability <= 1.0 and f.decoder_dropout_keep_probability >= 0.0, "Decoder dropout keep probability must be between 0 and 1"


    assert f.num_buckets in [3], "Only 3 buckets are supported, for now, but you can easily change this. Just pass in bucket sizes."
    assert f.num_buckets * f.minimum_data_ratio_per_bucket < 1, "Product of the number of buckets (%d) and the data ratio per bucket (%f) must be less than 1" % (f.num_buckets, f.minimum_data_ratio_per_bucket)

    assert os.path.isfile(f.encoder_decoder_architecture_json), "Invalid JSON file location passed for encoder architecture. Could not find %s" % f.encoder_architecture_json
    #TODO - add decoder as well