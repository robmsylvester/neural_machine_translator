#This file is responsible for handling all the helper code for buckets
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def get_default_bucket_sizes(num_buckets):
	if num_buckets == 3:
		return [(8,12), (16, 24), (24,40)]
	#TODO - finish this up

def get_candidate_bucket_sizes(num_buckets):
	if num_buckets == 3:
		return [
			[(8,12), (16,24), (24,40)],
			[(10,15), (20,30), (30,40)],
			[(10,15), (15,20), (25,40)]
		]

def get_bucket_score(bucket_sizes, dataset, minimum_bucket_data_ratio=0.0):
	num_buckets = len(dataset)
	print("get bucket score sees %d buckets" % num_buckets)

	total_dataset_size = 0
	for i in range(len(dataset)):
		total_dataset_size += len(dataset[i])

	print("total dataset size is %d" % total_dataset_size)

	for i in range(len(dataset)):
		if float(len(dataset[i])) / total_dataset_size < minimum_bucket_data_ratio:
			print("Bucket index %d does not have the required %f ratio of the total data. It instead has %.4f" % (i, minimum_bucket_data_ratio, float(len(dataset[i])) / total_dataset_size ) )
			return float("inf")

	total_source_pads = 0
	total_target_pads = 0

	for i in range(len(dataset)):
		for sentence_pair in dataset[i]:
			total_source_pads += ( bucket_sizes[i][0] - len(sentence_pair[0]) )
			total_target_pads += ( bucket_sizes[i][1] - len(sentence_pair[1]) )

	print("found a total of %d source pads and %d target pads" % (total_source_pads, total_target_pads))

	return total_target_pads + total_source_pads





