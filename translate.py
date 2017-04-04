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
and then start training a model saving checkpoints to --train_dir.

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
import download_utils
import seq2seq_model


FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
#_buckets2 = [(5, 10), (10, 15), (20, 25), (40, 50)]

#THESE WILL NEED TO CHANGE
_buckets = [(8,12), (16, 24), (24,40)]

def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.from_vocab_size,
      FLAGS.to_vocab_size,
      _buckets,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      forward_only=forward_only,
      dtype=dtype)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model


def train():

  from_train, to_train, from_dev, to_dev, _, _ = vocabulary_utils.prepare_wmt_data(FLAGS.data_dir,
                                                                                  FLAGS.from_vocab_size,
                                                                                  FLAGS.to_vocab_size)

  with tf.Session() as sess:
    # Create model.
    print("Session initialized. Creating Model...")

    if FLAGS.load_train_set_in_memory:
      print("Training set will be loaded into memory")
    else:
      print("Training set batches will be dynamically placed into buckets at each training step by reading from file")

    model = create_model(sess, False)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)

    #Load the validation set in memory always, because its relatively small
    dev_set = vocabulary_utils.load_dataset_in_memory(from_dev, to_dev, _buckets)

    #The training set will be loaded into memory only if the user specifies in a flag, because
    #with large numbers of buckets, big models, or huge datasets, you can run out of memory easily
    if FLAGS.load_train_set_in_memory:
      train_set = vocabulary_utils.load_dataset_in_memory(from_train,
                                                    to_train,
                                                    _buckets,
                                                    ignore_lines=FLAGS.train_offset,
                                                    max_size=FLAGS.max_train_data_size)
    else:
      raise NotImplementedError("Rob, write this method")

    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time = 0.0
    loss = 0.0
    current_step = 0
    previous_losses = []

    while True:

      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.

      #we will choose a random bucket
      r_n = np.random.rand()

      for bucket_idx, bucket_scale in enumerate(train_buckets_scale):
        if bucket_scale > r_n:
          bucket_id = bucket_idx
          break

      #Start a timer for training
      start_time = time.time()

      #Get a batch from the model by choosing source-target bucket pairs that match the bucket id chosen.
      #Target weights are necessary because they will allow us to weight how much we care about missing
      #each of the logits. a naive implementation of this, and probably not a bad way to do it at all,
      #is to just weight the PAD tokens at 0 and every other word as 1
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set,
                                                                        bucket_id,
                                                                        load_from_memory=FLAGS.load_train_set_in_memory)
      
      #Run a step of the model. 
      _, step_loss, _ = model.step(sess,
                                   encoder_inputs,
                                   decoder_inputs,
                                   target_weights,
                                   bucket_id,
                                   False)

      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      lowest_loss = float("inf")

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:

        # Print statistics for the previous epoch.
        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
        print ("global step %d learning rate %.6f step-time %.2f perplexity "
               "%.4f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))

        # Decrease learning rate if no improvement was seen over last x times.
        if len(previous_losses) > 1 and loss > max(previous_losses[-1*FLAGS.loss_increases_per_decay:]):
          sess.run(model.learning_rate_decay_op)
          print("Decayed learning rate after seeing two consecutive increases in perplexity")
        previous_losses.append(loss)

        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        
        if loss < lowest_loss:
          lowest_loss = loss
          print("new lowest loss. saving model")
          #TODO - wrap this in some sort of flag
          model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        else:
          print("not saving model")

        step_time = 0.0
        loss = 0.0
        # Run evals on development set and print their perplexity.
        for bucket_id in xrange(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("\teval on validation set: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set,
                                                                          bucket_id,
                                                                          load_from_memory=True)

          _, eval_loss, _ = model.step(sess,
                                        encoder_inputs,
                                        decoder_inputs,
                                        target_weights,
                                        bucket_id,
                                        True)

          eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")

          print("\teval on validation set: bucket %d perplexity %.4f" % (bucket_id, eval_ppx))
        sys.stdout.flush()


def decode():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    
    en_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocabulary_%d.from" % FLAGS.from_vocab_size)
    fr_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocabulary_%d.to" % FLAGS.to_vocab_size)
    en_vocab, _ = vocabulary_utils.initialize_vocabulary(en_vocab_path)
    _, rev_fr_vocab = vocabulary_utils.initialize_vocabulary(fr_vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      # Get token-ids for the input sentence.
      token_ids = vocabulary_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
      # Which bucket does it belong to?
      bucket_id = len(_buckets) - 1
      for i, bucket in enumerate(_buckets):
        if bucket[0] >= len(token_ids):
          bucket_id = i
          break
      else:
        logging.warning("Sentence truncated: %s", sentence)

      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id, load_from_memory=True)
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if vocabulary_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(vocabulary_utils.EOS_ID)]
      # Print out French sentence corresponding to outputs.
      print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()


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