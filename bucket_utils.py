#This file is responsible for handling all the helper code for buckets
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def get_default_bucket_sizes(num_buckets):
	if num_buckets == 3:
		return [(8,12), (16, 24), (24,40)]
	#TODO - finish this up

def determine_ideal_bucket_sizes(sentence_lengths, num_buckets):
	max_source_size_to_check = FLAGS.max_source_bucket_size_to_check
	max_target_size_to_check = FLAGS.max_target_bucket_size_to_check

	ideal_num_sentences_per_bucket = len(sentence_lengths) / float(num_buckets) #1 million data points / 5 buckets = 200,000 per bucket

	return None

