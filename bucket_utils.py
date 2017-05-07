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
			[(8,12), (16,24), (30,50)],
			[(10,15), (20,30), (32,50)],
			[(10,20), (20,35), (36,56)]
		]

def get_bucket_score(bucket_sizes, dataset, unbucketed_dataset_ratio, minimum_bucket_data_ratio=0.0):
	num_buckets = len(dataset)

	total_dataset_size = 0
	for i in range(len(dataset)):
		total_dataset_size += len(dataset[i])

	print("evaluating bucket scores for buckets " + str(bucket_sizes))
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

	print("found a total of %d source pads and %d target pads. Unbucketed data ratio is %.4f" % (total_source_pads, total_target_pads, unbucketed_dataset_ratio))

	#this is a naive hack, but the idea is that we can penalize for unbucketed data examples.
	#this is an open ended problem here, what we do to penalize and target.
	#low numbers are good!
	return (total_target_pads + total_source_pads) * max(0.05, unbucketed_dataset_ratio)





