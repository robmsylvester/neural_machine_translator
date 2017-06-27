# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to the data directory.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import vocabulary_utils
import bucket_utils
import download_utils
import seq2seqEDA


FLAGS = tf.app.flags.FLAGS

def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in tensorflow session.
  Args:
    session - Tensorflow session created with tf.Session()    
    forward_only - Boolean, amounts to whether or not we are training. If decoding or running validation set,
                   we don't care about updating the parameters via backpropagation, just doing a prediction.
                   If training, need backprop. Amounts to a control op on whether or not to run those gradient
                   updates in the session
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32 #TODO - deprecated support for tf.float16, kill it.

  model = seq2seqEDA.seq2seqEDA(
      FLAGS.from_vocab_size,
      FLAGS.to_vocab_size,
      FLAGS.max_clipped_gradient,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      FLAGS.minimum_learning_rate,
      FLAGS.encoder_decoder_architecture_json,
      FLAGS.max_source_sentence_length,
      FLAGS.max_target_sentence_length,
      forward_only=forward_only,
      dtype=dtype)
  ckpt = tf.train.get_checkpoint_state(FLAGS.data_dir)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model


def beam_search_decoder():
  #outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
  pass



def train():
  #Load data from file, preprocess it and tokenize it, integerize it, all according to different flags.
  from_train, to_train, from_dev, to_dev, _, _ = vocabulary_utils.prepare_wmt_data(FLAGS.data_dir,
                                                                                  FLAGS.from_vocab_size,
                                                                                  FLAGS.to_vocab_size)

  with tf.Session() as sess:
    # Create model.
    print("Session initialized. Creating Model...")

    if FLAGS.load_train_set_in_memory:
      print("Training set will be loaded into memory.")
    else:
      print("Training set will be read from training file on each batch instance.")

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)

    #The training set will be loaded into memory only if the user specifies in a flag
    if FLAGS.load_train_set_in_memory:
      train_set, _ = vocabulary_utils.load_dataset_in_memory(from_train,
                                                            to_train,
                                                            FLAGS.max_source_sentence_length,
                                                            FLAGS.max_target_sentence_length,
                                                            ignore_lines=FLAGS.train_offset,
                                                            max_size=FLAGS.max_train_data_size)
    else:
      raise NotImplementedError("No support yet for loading training data from files.")

    #Load the validation set in memory always, because its relatively small
    dev_set, _ = vocabulary_utils.load_dataset_in_memory(from_dev,
                                                         to_dev,
                                                         FLAGS.max_source_sentence_length,
                                                         FLAGS.max_target_sentence_length)

    model = create_model(sess, False)

    total_train_size = len(train_set)

    # This is the training loop.
    step_time = 0.0
    loss = 0.0
    current_step = 0
    previous_losses = []

    #Start a best-model estimator
    lowest_loss = float("inf")

    while True:

      #Start a timer for training
      start_time = time.time()

      #Get a batch from the model by choosing source-target pairs.
      #Target weights are necessary because they will allow us to weight how much we care about missing
      #each of the logits. a naive implementation of this, and probably not a bad way to do it at all,
      #is to just weight the PAD tokens at 0 and every other word as 1
      encoder_inputs, decoder_inputs, encoder_input_lengths, decoder_input_lengths, target_weights = model.get_batch(train_set,
                                                                                                                    load_from_memory=FLAGS.load_train_set_in_memory,
                                                                                                                    use_all_rows=False)
      
      #Run a step of the model. 
      _, step_loss, _ = model.step(sess,
                                   encoder_inputs,
                                   decoder_inputs,
                                   encoder_input_lengths,
                                   decoder_input_lengths,
                                   target_weights,
                                   forward_only=False)

      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:

        # Print statistics for the previous epoch.
        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
        print ("global step %d report:\n\tlearning rate %.6f\n\taverage step-time %.2f\n\taverage last %d batch perplexity "
               "%.4f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, FLAGS.steps_per_checkpoint, perplexity))

        # Decrease learning rate if no improvement was seen over last x times.
        if len(previous_losses) > 1 and loss > max(previous_losses[-1*FLAGS.loss_increases_per_decay:]):
          sess.run(model.learning_rate_decay_op)
          print("Decayed learning rate after seeing %d consecutive increases in perplexity" % FLAGS.loss_increases_per_decay)
        previous_losses.append(loss)

        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.data_dir, FLAGS.checkpoint_name)
        
        #Only save the model if the loss is better than any loss we've yet seen.
        if loss < lowest_loss:
          lowest_loss = loss
          print("new lowest loss. saving model")
          #TODO - wrap this in some sort of flag
          model.saver.save(sess, checkpoint_path, global_step=model.global_step)

        #Prepare for validation set evaluation
        step_time = 0.0
        loss = 0.0

        print("Running Validation set...")

        encoder_inputs, decoder_inputs, encoder_input_lengths, decoder_input_lengths, target_weights = model.get_batch(dev_set,
                                                                                                                      load_from_memory=True,
                                                                                                                      use_all_rows=False)

        _, eval_loss, _ = model.step(sess,
                                      encoder_inputs,
                                      decoder_inputs,
                                      encoder_input_lengths,
                                      decoder_input_lengths,
                                      target_weights,
                                      forward_only=True)

        eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")

        print("\teval on random batch of validation sentences: perplexity %.4f" % eval_ppx)
        sys.stdout.flush()


def decode():
  with tf.Session() as sess:

    #TODO - remove inference for best buckets if you're using a decoder. 
    #_buckets = bucket_utils.get_default_bucket_sizes(FLAGS.num_buckets)

    # Create model and load parameters.
    model = create_model(sess, forward_only=True)

    model.batch_size = 1  # We decode one sentence at a time, regardless of batch_size flag.

    # Load vocabularies.
    en_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocabulary_%d.from" % FLAGS.from_vocab_size)
    fr_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocabulary_%d.to" % FLAGS.to_vocab_size)

    en_vocab, _ = vocabulary_utils.initialize_vocabulary(en_vocab_path)
    _, rev_fr_vocab = vocabulary_utils.initialize_vocabulary(fr_vocab_path)


    # Decode from standard input.
    sys.stdout.write(">> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:

      #TODO - CALL CLEANER HERE
      # Get token-ids for the input sentence.
      sentence = tf.compat.as_bytes(sentence)
      clean_sentence = vocabulary_utils.clean_sentence(sentence, language="en")
      str_tokens = vocabulary_utils.vanilla_ft_tokenizer(clean_sentence)
      print("\nYour sentence will be tokenized as follows:\n\t%s" % str(str_tokens))

      token_ids = [en_vocab.get(word, vocabulary_utils.UNK_ID) for word in str_tokens]
      print("\nYour sentence will be integerized as follows:\n\t%s" % str(token_ids))

      if len(token_ids) > FLAGS.max_source_sentence_length:
        logging.warning("Sentence truncated: %s", sentence)

      # Get a 1-element batch to feed the sentence to the model.
      # List has one element as input, a tuple
      # The source token id's are the token's we just processed
      # the target token id's are an empty list.
      encoder_inputs, decoder_inputs, encoder_input_lengths, decoder_input_lengths, target_weights = model.get_batch(
                                                                                                      [(token_ids, [])],
                                                                                                      load_from_memory=True)

      #TODO - explicitly set decoder input lengths to some large number?

      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess,
                                      encoder_inputs,
                                      decoder_inputs,
                                      encoder_input_lengths,
                                      decoder_input_lengths,
                                      target_weights,
                                      forward_only=True)

      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

      # TODO - BEAM SEARCH HERE

      # If there is an EOS symbol in outputs, cut them at that point.
      if vocabulary_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(vocabulary_utils.EOS_ID)]
      # Print out French sentence corresponding to outputs.
      print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()




#TODO - remove this or change it
def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.global_variables_initializer())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id, load_from_memory=True)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)