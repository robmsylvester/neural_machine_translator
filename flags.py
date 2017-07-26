import tensorflow as tf
import os
import sys

#==========================Regularization===============================================
#tf.app.flags.DEFINE_boolean("l2_loss", False,
#                            "Not currently implemented. Adds L2 loss term to parameter weights in encoder and attention decoder")
#tf.app.flags.DEFINE_float("l2_loss_lambda", 0.00,
#                           "Not currently implemented. Adds L2 loss term to parameter weights in encoder and attention decoder")



#==========================Basic Execution Options======================================
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")


#=================================Learning Rate==============================================
#TODO - abstract this away into a learning rate schedule
tf.app.flags.DEFINE_integer("loss_increases_per_decay", 3,
                            "The learning rate will decay if the loss is greater than the max of the last (this many) checkpoint losses.")

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")

tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much upon seeing loss_increases_per_decay consecutive higher average perplexities over the last steps_per_checkpoint training iterations")

tf.app.flags.DEFINE_float("minimum_learning_rate", 0.005, "Minimum learning rate")




#=======================Gradient Clipping====================================================
tf.app.flags.DEFINE_float("max_clipped_gradient", 5.0,
                          "Clip gradients to a maximum of this this norm.")




#=======================Vocabulary File Locations and Save File Locations=======================
tf.app.flags.DEFINE_integer("from_vocab_size", 40000, "Source language vocabulary size.") #20-50k is good

tf.app.flags.DEFINE_integer("to_vocab_size", 40000, "Target language vocabulary size.")

tf.app.flags.DEFINE_string("data_dir", "/home/rob/WMT", "Directory where we will store the data as well as model checkpoint")



#===================================Checkpoint Flags====================================
tf.app.flags.DEFINE_string("checkpoint_name", "translate.ckpt", "Name of the Tensorflow checkpoint file")

tf.app.flags.DEFINE_integer("steps_per_checkpoint", 300, #change me to 300
                            "How many training steps to do per checkpoint.")




#Dataset Flags

#TODO - this right now is useless until i dynamically load datasets.
tf.app.flags.DEFINE_boolean("load_train_set_in_memory", True,
                            "If True, loads training set into memory. Otherwise, reads batches by opening files and reading appropriate lines.")

tf.app.flags.DEFINE_integer("max_train_data_size", 200000,
                            "Limit on the size of training data (0: no limit).")

tf.app.flags.DEFINE_integer("train_offset", 0,
                            "ignore the first train_offset lines of the training file when loading the training set or getting randomly")
#==========================================================================================





#==========================Data Preprocessing Flags===================================
tf.app.flags.DEFINE_integer("max_source_sentence_length", 35, #I like long sentences and tough datasets, so I use 50 and 60 here usually
                            "the maximum number of tokens in the source sentence training example in order for the sentence pair to be able to be used in the dataset")

tf.app.flags.DEFINE_integer("max_target_sentence_length", 45,
                            "the maximum number of tokens in the target sentence training example in order for the sentence pair to be able to be used in the dataset")




#Embedding Flags for Encoder and Decoder
#Was 1024, 512
#===========================Word Embeddings=====================================
tf.app.flags.DEFINE_string("embedding_algorithm", "network",
                            "glove, or network. The first three are unsupervised trainers implemented by other programs. the latter is a network layer trained only by backprop")

tf.app.flags.DEFINE_boolean("train_embeddings", True,
                            "Whether or not to continue training the glove embeddings from backpropagation or to leave them be")

tf.app.flags.DEFINE_integer("encoder_embedding_size", 512,
                            "Number of units in the embedding size of the encoder inputs. This will be used in a wrapper to the first layer")
#Was 1024, 512
tf.app.flags.DEFINE_integer("decoder_embedding_size", 512,
                            "Number of units in the embedding size of the encoder inputs. This will be used in a wrapper to the first layer")

tf.app.flags.DEFINE_string("glove_encoder_embedding_file", "../translator/GloVe/build/rob_vectors_50it_200vec_source.txt",
                            "The output file for Glove-trained word embeddings on the dataset.")

tf.app.flags.DEFINE_string("glove_decoder_embedding_file", "../translator/GloVe/build/rob_vectors_25it_200vec_target.txt",
                            "The output file for Glove-trained word embeddings on the dataset.")




#===========================Parameters==========================================
tf.app.flags.DEFINE_integer("batch_size", 32, #64 would be good, 128 is better.
                            "Batch size to use during training.")


#TODO - add to json, implement and test
tf.app.flags.DEFINE_integer("num_attention_heads", 1,
                            "The number of heads to use in the attention mechanism")


tf.app.flags.DEFINE_integer("sampled_softmax_size", 512, #64 would be good, 128 is better.
                            "Sampled Softmax will use this many logits out of the vocab size for the probability estimate of the true word")


#TODO - decoder vocab boosting is currently not implemented.
tf.app.flags.DEFINE_boolean("decoder_vocab_boosting", False,
                            "adaboost decoder prediction weights in the loss function based on perplexities of sentences that contain that word")


tf.app.flags.DEFINE_integer("vocab_boost_occurrence_memory", 100,
                            "When calculating the perpelxity of sentences that contain a certain word, only count the last (this many) sentences with that word")



#==========================================================================================


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
                           "The strategy used to calculate initial state values for the decoder from the encoder")


#This is where we store the JSON of the encoder and decoder architecture
#See the file encoder_decoder_architecture_help.txt for more information
tf.app.flags.DEFINE_string("encoder_decoder_architecture_json", 'encoder_decoder_architecture.json',
                            "The file name which stores the JSON architecture of the encoder and decoder")


#Dynamic vs Static Encoder and Decoders.
#Dynamic will use calls to the dynamic api and pass sequence length
#Static will use max_time on all network runs, using _PAD symbol, and weighted the padded logits at 0.
#The Decoder intercepts outputs at every time step and passes to attention and other stuff, so we just intercept them anyway (meaning it's always static with time=1)
tf.app.flags.DEFINE_string("encoder_rnn_api", "static",
                            "must be static or dynamic. if static, uses tensorflow static rnn calls and PAD symbols. if dynamic, uses tensorflow dynamic rnn calls and sequence lengths.")

#TODO - Flesh this out when flags by migrating a few of the tests over from the other code that are common mistakes.
# Alternatively, do absolutely all that we can right here with the flag testing and try to remove a few more of them from the model.
def flag_test():

    def validate_learning_rate_flags(flags):
        assert flags.learning_rate > 0. and flags.learning_rate <= 1., "Your initial learning rate must be in the range (0,1]"
        assert flags.learning_rate_decay_factor > 0. and flags.learning_rate_decay_factor <= 1., "Your learning rate decay must be in the range (0,1]"
        assert flags.minimum_learning_rate > 0. and flags.minimum_learning_rate <=1., "Your minimum learning rate must be in the range (0,1]"

    def validate_gradient_flags(flags):
        assert flags.max_clipped_gradient > 0., "The max clipped gradient norm must be a positive real number"

    def validate_file_locations(flags):
        assert os.path.isdir(flags.data_dir), "%s is not a directory in this filesystem." % flags.data_dir
        assert flags.checkpoint_name.endswith(".ckpt"), "The checkpoint file name %s needs to ends with '.ckpt' " % flags.checkpoint_name
        assert os.path.isfile(os.path.join(os.getcwd(), flags.encoder_decoder_architecture_json)), "%s is not a directory in this filesystem." % os.path.join(flags.data_dir, flags.encoder_decoder_architecture_json)

    def validate_encoder_api(flags):
        permitted = ['static', 'dynamic']
        assert flags.encoder_rnn_api in permitted, "Encoder api %s is invalid" % flags.encoder_rnn_api

    def validate_decoder_state_initializer(flags):
        permitted = ['nematus', 'mirror', 'top_layer_mirror', 'bahdanu']
        assert flags.decoder_state_initializer in permitted, "Decoder state initializer %s is invalid" % flags.decoder_state_initializer

    def validate_softmax_sample_size(flags):
        assert flags.sampled_softmax_size <= flags.to_vocab_size, "Sampled softmax must not use more labels than there are target vocabulary words."

    def validate_embedding_algorithm(flags):
        permitted = ['network', 'glove']
        assert flags.embedding_algorithm in permitted, "Embedding algorithm %s is not supported" % flags.embedding_algorithm

        if flags.embedding_algorithm == 'glove':
            assert os.path.isfile(os.path.join(os.getcwd(), flags.glove_encoder_embedding_file)), "Glove embedding file %s does not exist in the file system" % os.path.join(os.getcwd(), flags.glove_encoder_embedding_file)
            assert os.path.isfile(os.path.join(os.getcwd(), flags.glove_decoder_embedding_file)), "Glove embedding file %s does not exist in the file system" % os.path.join(os.getcwd(), flags.glove_decoder_embedding_file)
    f = tf.app.flags.FLAGS

    validate_learning_rate_flags(f)
    validate_gradient_flags(f)
    validate_file_locations(f)
    validate_encoder_api(f)
    validate_decoder_state_initializer(f)
    validate_softmax_sample_size(f)
    validate_embedding_algorithm(f)
    print("Flag inputs are valid.")