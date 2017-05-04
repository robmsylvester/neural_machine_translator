# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import tarfile

from six.moves import urllib
from tensorflow.python.platform import gfile
import download_utils
import bucket_utils
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.

#The pad symbol is what will be appended to the end of sentences until they hit a fixed length.
#Because this translator trains with buckets, the number of pad symbols will be based on sentence length.
#If the bucket has a max of 10 words, and your sentence has 7 words, then three _PAD symbols are added.
_PAD = b"_PAD"

#The decoder works by feeding the previous decoder output into the attention mechanism decoder and
#using it to subtract away the old context and poop out the next word in the sequence. This means
#we can prepend target language sentences with this _GO symbol and it will work as sort of a base case
#to get things going for the target sentence.
_GO = b"_GO"

#This special marker for the end of a sentence is used by the predictive model to know when to stop
#outputting words. Output continues until this symbol is hit. Happily for us, FastText has a word
#embedding for this, though it is called </s> so this special case will be taken care of throughout this
#project
_EOS = b"_EOS"

#Words that the model does not recognize will be replace with this special symbol. If we are using FastText
#then we have a few options. We can not use this at all, or we can look up a word and find it's closest
#neighbor for which we have a vector. ("canine" gets replaced with "dog", for example). Or, even better,
#we can use a weighted combination of possible word candidates. Or, we can use a weighted combination
#of word candidates and the unknown word embedding, all based upon their ratios and the strength of the
#fastText prediction as evidenced by its cosine distance between the unknown word and the embedding prediction.
_UNK = b"_UNK"

_INITIAL_VOCABULARY = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


def vanilla_ft_tokenizer(text):
  #This function splits sentences into a list of tokens in a naive way targeted for two things:
  # A - Using FastText word embeddings that expect lowercase words, no hyphens, no punctuation.

  pattern = re.compile(b"([-.,\!?\"':;_)(])/")
  tokens = []

  #split by spaces
  for split_by_space in text.strip().split():

    #then by tokens
    tokens.extend(pattern.split(split_by_space))

  return [i for i in tokens if i]




def clean_sentence(sentence, language="en"):

  #always lowercase for everything, but that doesn't capture all the characters in french
  sentence = sentence.lower()

  #this awkward quotation mark makes an appearance sometimes in the WMT dataset. He can fuck right off.
  sentence = re.sub("’", "'", sentence)

  #FastText does this with numbers. So we will do this too to use their embeddings. Besides, it's
  #not a bad way of dealing with numbers really.
  if language == "en":
    sentence = re.sub('0', ' zero ', sentence)
    sentence = re.sub('1', ' one ', sentence)
    sentence = re.sub('2', ' two ', sentence)
    sentence = re.sub('3', ' three ', sentence)
    sentence = re.sub('4', ' four ', sentence)
    sentence = re.sub('5', ' five ', sentence)
    sentence = re.sub('6', ' six ', sentence)
    sentence = re.sub('7', ' seven ', sentence)
    sentence = re.sub('8', ' eight ', sentence)
    sentence = re.sub('9', ' nine ', sentence)
  elif language == "fr":
    sentence = re.sub('É', 'é', sentence)
    sentence = re.sub('È', 'è', sentence)
    sentence = re.sub('Ë', 'ë', sentence)
    sentence = re.sub("Ê", "ê", sentence)
    sentence = re.sub('À', 'à', sentence)
    sentence = re.sub('Á', 'á', sentence)
    sentence = re.sub('Â', 'â', sentence)
    sentence = re.sub('Î', 'î', sentence)
    sentence = re.sub('Ï', 'ï', sentence)
    sentence = re.sub('Ö', 'ö', sentence)
    sentence = re.sub('Ô', 'ô', sentence)
    sentence = re.sub('Ó', 'ó', sentence)
    sentence = re.sub('Ò', 'ò', sentence)
    sentence = re.sub('Û', 'û', sentence)
    sentence = re.sub('Ü', 'ü', sentence)
    sentence = re.sub('Ù', 'ù', sentence)
    sentence = re.sub('Ç', 'ç', sentence)
    sentence = re.sub('0', ' zéro ', sentence)
    sentence = re.sub('1', ' un ', sentence)
    sentence = re.sub('2', ' duex ', sentence)
    sentence = re.sub('3', ' trois ', sentence)
    sentence = re.sub('4', ' quatre ', sentence)
    sentence = re.sub('5', ' cinq ', sentence)
    sentence = re.sub('6', ' six ', sentence)
    sentence = re.sub('7', ' sept ', sentence)
    sentence = re.sub('8', ' huit ', sentence)
    sentence = re.sub('9', ' neuf ', sentence)
  else:
    raise ValueError("clean_setnence() only has rules implemented for English and French. This isn't a horrible error. Basically,\
      write out commands for what to do with numbers. If using a FastText embedding, this means explicitly finding string representations\
       for the numbers in the language. Write out any explicit commands for dealing with lowercase accented characters that Python's lower()\
       will not take care of.")

  return sentence


def clean_enfr_wmt_data(output_file,
                    input_file,
                    language="en",
                    report_frequency=500000):

  """
  This will go through and lowercase everything, make every number written out (ie, 45 becomes four five, not forty-five) 
  We also explicitly lowercase silly french letters with silly little baguettes over them.
  """
  if gfile.Exists(output_file):
    print("Cleaned dataset file %s detected. Skipping cleaning" % output_file)
    return output_file

  assert gfile.Exists(input_file), "Could not find dataset file %s to create cleaned dataset file" % input_file

  read_counter = 0

  with gfile.GFile(input_file,mode='rb') as f:
    with gfile.GFile(output_file,mode='w') as out:
      print("Cleaning dataset file %s using %s as a source language." % (input_file, language))
      for line in f:
        read_counter += 1
        line = tf.compat.as_bytes(line)
        #print line

        line = clean_sentence(line, language=language)

        #Our word vector tokenizer in fasttext is admittedly shitty. he looks for whitespace, and thats it.
        #So we will abstract this away by a tokenizer ourselves and writing into the clean file a separation
        #of tokens by a single white space.
        tokens = vanilla_ft_tokenizer(line)
        to_write = ''
        for token in tokens:
          to_write += token + ' '

        #We do NOT need to add an _EOS tag, because fast text will do this for us. We can just reference it in the trainer.
        #Fasttext uses a </s> symbol for the end of sentences.
        to_write += '\n'
        
        #We should explicitly specify utf8 because of all the bytecode character above.
        #to_write = to_write.decode('utf-8')
        out.write(to_write)

        if read_counter % 500000 == 0:
          print("read %d lines" % read_counter)
  print("Done.\nClean output dataset file created at %s" % output_file)
  return output_file





def create_vocabulary(output_vocabulary_path, input_data_path, max_vocabulary_size,
                      tokenizer=None, report_frequency=1000000):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
  """

  if gfile.Exists(output_vocabulary_path):
    print("Vocabulary file %s already exists. Skipping this step..." % output_vocabulary_path)
  else:
    assert gfile.Exists(input_data_path), "Cannot find input data file at %s\nNo vocabulary file will be created" % input_data_path

    vocabulary = {}
    total_tokens = 0
    vocab_tokens = 0

    with gfile.GFile(input_data_path, mode='rb') as f:
      print("Creating vocabulary file %s for the top %d words in corpus\nThis may take a few minutes. Go eat a sandwich." % (output_vocabulary_path, max_vocabulary_size))
      line_count = 0
      for line in f:

        #read the sentence and tokenize it using whatever tokenizer we defined
        line = tf.compat.as_bytes(line)
        line_count += 1
        tokens = vanilla_ft_tokenizer(line)
        total_tokens += len(tokens)
        
        #go token through token and increment each word counter every time we see it
        for t in tokens:
          if t in vocabulary:
            vocabulary[t] += 1
          else:
            vocabulary[t] = 1
        
        if line_count % report_frequency == 0:
          print("Processing line %d..." % line_count)

      #append the special symbols to our vocabulary
      full_vocabulary = _INITIAL_VOCABULARY + sorted(vocabulary, key=vocabulary.get, reverse=True)
      top_vocabulary = full_vocabulary[:max_vocabulary_size]

      with gfile.GFile(output_vocabulary_path, mode='wb') as out:
        for word in top_vocabulary:
          if word not in _INITIAL_VOCABULARY:
            vocab_tokens += vocabulary[word]
          out.write(word)
          out.write(b"\n")

    print("Created vocabulary file with %d words.\nThe rate of unknown words for this vocabulary was %.4f" % (min(len(top_vocabulary),max_vocabulary_size), 1 - vocab_tokens / float(total_tokens)))




def get_sentence_length_distribution(input_file, max_length, report_frequency=500000):
  #read sentence lengths from the file. does not take tokenized sentences, but a raw file.
  #max_length is the max possible length a sentence can have. one more bin will be made above
  #this in the list to hold values more than max_length. holds a value for 0-length as well
  #returns the list of lengths. 
  #
  # example with max_length=4
  # returns [ 0, 100,130,127,65,44 ] if 44 sentences have length >5, 65 have length 4, 127 have length 3, 130 length 2, 100 length 1, and 0 length 0
  # 
  lengths = [0] * (max_length+2) #+2 to include a spot for 0 and >max_length

  with gfile.GFile(input_file,'rb') as f:
    line_count = 0
    for line in f:
      tokens = vanilla_ft_tokenizer(line)
      length = len(tokens)
      if length > max_length:
        lengths[max_length+1] += 1
      else:
        lengths[length] += 1 
      line_count += 1
      if line_count % report_frequency == 0:
        print("Processed line %d" % line_count)
  return lengths



def even_bucket_distribution(sentence_lengths, num_buckets):
  #this is just a naive implementation, since bucket lengths don't need to be exact, but we might as well get something in the ballpark
  #to get the buckets to some sane values by looking at some averages.
  running_sum = 0
  bucket_indexes = []
  bucket_sums = []
  target_per_bucket = sum(sentence_lengths) / num_buckets
  for length_idx, num_lengths in enumerate(sentence_lengths):
    running_sum += num_lengths
    if running_sum > target_per_bucket or length_idx == len(sentence_lengths) - 1:
      bucket_indexes.append(length_idx)
      bucket_sums.append(running_sum)
      running_sum = 0
  return bucket_indexes



def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """

  assert gfile.Exists(vocabulary_path), "Vocabulary path %s not found" % vocabulary_path
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)






def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = vanilla_ft_tokenizer(sentence)
  
  return [vocabulary.get(w, UNK_ID) for w in words]





def integerize_sentence(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True, report_frequency=100000):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Integerizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % report_frequency == 0:
            print("Processed line %d" % counter)
          token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab,
                                            tokenizer, normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_wmt_data(data_dir, en_vocabulary_size, fr_vocabulary_size, tokenizer=None):
  """Get WMT data into data_dir, create vocabularies and tokenize data.

  Args:
    data_dir: directory in which the data sets will be stored.
    en_vocabulary_size: size of the English vocabulary to create and use.
    fr_vocabulary_size: size of the French vocabulary to create and use.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.

  Returns:
    A tuple of 6 elements:
      (1) path to the token-ids for English training data-set,
      (2) path to the token-ids for French training data-set,
      (3) path to the token-ids for English development data-set,
      (4) path to the token-ids for French development data-set,
      (5) path to the English vocabulary file,
      (6) path to the French vocabulary file.
  """
  # Get wmt data to the specified directory.
  train_path = download_utils.get_wmt_enfr_train_set(data_dir)
  dev_path = download_utils.get_wmt_enfr_dev_set(data_dir)

  from_train_path = train_path + ".en"
  to_train_path = train_path + ".fr"
  from_dev_path = dev_path + ".en"
  to_dev_path = dev_path + ".fr"
  return prepare_data(data_dir, from_train_path, to_train_path, from_dev_path, to_dev_path, en_vocabulary_size,
                      fr_vocabulary_size, tokenizer)






def prepare_data(data_dir, from_train_path, to_train_path, from_dev_path, to_dev_path, from_vocabulary_size,
                 to_vocabulary_size, tokenizer=None):
  """Preapre all necessary files that are required for the training.

    Args:
      data_dir: directory in which the data sets will be stored.
      from_train_path: path to the file that includes "from" training samples.
      to_train_path: path to the file that includes "to" training samples.
      from_dev_path: path to the file that includes "from" dev samples.
      to_dev_path: path to the file that includes "to" dev samples.
      from_vocabulary_size: size of the "from language" vocabulary to create and use.
      to_vocabulary_size: size of the "to language" vocabulary to create and use.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.

    Returns:
      A tuple of 6 elements:
        (1) path to the token-ids for "from language" training data-set,
        (2) path to the token-ids for "to language" training data-set,
        (3) path to the token-ids for "from language" development data-set,
        (4) path to the token-ids for "to language" development data-set,
        (5) path to the "from language" vocabulary file,
        (6) path to the "to language" vocabulary file.
    """

  # Clean the sentences by using lowercase letters and changing numbers into written words.



  # Clean the data files by dealing with lowercases, and numbers
  # This will run only if the .clean file doesn't exist
  to_clean_train_path = clean_enfr_wmt_data(to_train_path+ ".clean", to_train_path, language="fr")
  from_clean_train_path = clean_enfr_wmt_data(from_train_path+ ".clean", from_train_path, language="en")
  to_clean_dev_path = clean_enfr_wmt_data(to_dev_path+ ".clean", to_dev_path, language="fr")
  from_clean_dev_path = clean_enfr_wmt_data(from_dev_path+ ".clean", from_dev_path, language="en")


  # Create vocabularies based on the cleaned dataset files and the vocabulary paths
  # This will run only if the vocabulry file doesn't exist
  to_vocab_path = os.path.join(data_dir, "vocabulary_%d.to" % to_vocabulary_size)
  from_vocab_path = os.path.join(data_dir, "vocabulary_%d.from" % from_vocabulary_size)
  create_vocabulary(to_vocab_path, to_clean_train_path , to_vocabulary_size, tokenizer)
  create_vocabulary(from_vocab_path, from_clean_train_path , from_vocabulary_size, tokenizer)


  # Integerize the training data by replacing words with their vocabulary representations (integers)
  # This will run only if the integerized version of the training set doesn't already exist
  to_train_ids_path = to_clean_train_path + (".ids_%d" % to_vocabulary_size)
  from_train_ids_path = from_clean_train_path + (".ids_%d" % from_vocabulary_size)
  integerize_sentence(to_clean_train_path, to_train_ids_path, to_vocab_path, tokenizer)
  integerize_sentence(from_clean_train_path, from_train_ids_path, from_vocab_path, tokenizer)


  # Create token ids for the development data.
  # This will run only if the integerized version of the dev set doesn't already exiwst
  to_dev_ids_path = to_dev_path + (".ids_%d" % to_vocabulary_size)
  from_dev_ids_path = from_dev_path + (".ids_%d" % from_vocabulary_size)
  integerize_sentence(to_clean_dev_path, to_dev_ids_path, to_vocab_path, tokenizer)
  integerize_sentence(from_clean_dev_path, from_dev_ids_path, from_vocab_path, tokenizer)

  return (from_train_ids_path, to_train_ids_path,
          from_dev_ids_path, to_dev_ids_path,
          from_vocab_path, to_vocab_path)



def load_dataset_in_memory(source_path, target_path, buckets, max_size=None, ignore_lines=0, report_frequency=200000,
                            auto_build_buckets=False, num_auto_buckets=3):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with integer-like token-ids for the source language.
                 integer-like means they are still strings, obviously, but after being
                 processed through the integerizer
    target_path: path to the file with integer-like token-ids for the target language.
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    buckets - list of tuples (source max length, target max length) for
            - various sentence length buckets with which to append the data.
            - or, if auto_build_buckets is True, this can be None
    ignore_lines - integer, how many lines to ignore at the beginning of the file.
                  at times, it may be easier to train on a few million at a time.
                  then just stop the model and train on a different part of the data.
                  this will allow you to load it all in memory
    max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, all lines will be read.
    report_frequency: integer to specify to console how often to report progress in processing file
    auto_build_buckets: if True, will ignore buckets and instead create num_auto_buckets different buckets
                  such that the number of data sentences will be split evenly between the buckets
    num_auto_buckets: how many buckets to use in the auto build

  Returns:
    data_set: a list of length len(buckets); 
      data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < buckets[n][0] and
      len(target) < buckets[n][1];
      source and target are integer lists of token-ids.
    buckets: the buckets to be used in the neural network.
  """

  if auto_build_buckets:
    temp_data_set = [] #we don't have the bucket sizes, so we will get them later and write them all in one big bucket
  else:
    data_set = [ [] for _ in buckets] #we have the bucket sizes, so write straight to memory

  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      counter = 0
      source = source_file.readline() 
      target = target_file.readline()

      while source and target:

        #ignore the first x lines of the file
        if ignore_lines > 0:
          print("Will skip the first %d lines of the dataset" % ignore_lines)
        for k in range(ignore_lines+1):
          source = source_file.readline() 
          target = target_file.readline()         

        counter += 1

        if counter % report_frequency == 0:
          print("reading data line %d" % counter)

        #What we read is not quite an integer yet, so convert the string representations to actual integers.
        source_ids = [ int(x) for x in source.split()]
        target_ids = [ int(x) for x in target.split()]

        #We always add the end of sentence to the target sentence and do it here so that it doesn't
        #go above the max bucket size ever. 
        target_ids.append(EOS_ID)

        #Add to appropriate bucket based upon the sentence length
        #If the sentence goes over the bucket length, it doesnt go in
        if not auto_build_buckets:
          for bucket_id, (source_size, target_size) in enumerate(buckets):
            if len(source_ids) < source_size and len(target_ids) < target_size:
              data_set[bucket_id].append([source_ids, target_ids])
              break
        else:
          temp_data_set.append([source_ids, target_ids])

        if max_size and counter >= max_size:
          break

        #Prepare next loop iteration
        source = source_file.readline()
        target = target_file.readline()


  if auto_build_buckets:
    print("Building data set lengths to prepare for determining ideal buckets...")
    data_set_lengths = [ [len(sentence_pair[0]), len(sentence_pair[1])] for sentence_pair in temp_data_set]

    buckets = bucket_utils.determine_ideal_bucket_sizes(data_set_lengths, num_auto_buckets)

    data_set = [ [] for _ in buckets]
    for bucket_id, (source_size, target_size) in enumerate(buckets):
      if len(source_ids) < source_size and len(target_ids) < target_size:
        data_set[bucket_id].append([source_ids, target_ids])

  return data_set, buckets



def get_dataset_sentence_lengths(source_path, target_path, max_size=None, ignore_lines=0, report_frequency=200000):
  """Read data from source and target files and get their lengths so that they can be used to auto-build buckets

  Args:
    source_path: path to the files with integer-like token-ids for the source language.
                 integer-like means they are still strings, obviously, but after being
                 processed through the integerizer
    target_path: path to the file with integer-like token-ids for the target language.
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    ignore_lines - integer, how many lines to ignore at the beginning of the file.
                  at times, it may be easier to train on a few million at a time.
                  then just stop the model and train on a different part of the data.
                  this will allow you to load it all in memory
    max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, all lines will be read.
    report_frequency: integer to specify to console how often to report progress in processing file

  Returns:
    data_set: a list of (x_i,y_i) tuples where x_i=truncated source sentence length at line i in source file, y_i=
              truncated target sentence length at line i in target file
  """
  print("Reading dataset line lengths to determine ideal bucket lengths. ")
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      counter = 0
      source = source_file.readline() 
      target = target_file.readline()

      while source and target:

        #ignore the first x lines of the file
        if ignore_lines > 0:
        for k in range(ignore_lines+1):
          source = source_file.readline() 
          target = target_file.readline()         

        counter += 1

        if counter % report_frequency == 0:
          print("reading data line %d" % counter)

        #What we read is not quite an integer yet, so convert the string representations to actual integers.
        source_ids = [ int(x) for x in source.split()]
        target_ids = [ int(x) for x in target.split()]

        #We always add the end of sentence to the target sentence and do it here so that it doesn't
        #go above the max bucket size ever. 
        target_ids.append(EOS_ID)

        #Add to appropriate bucket based upon the sentence length
        #If the sentence goes over the bucket length, it doesnt go in
        for bucket_id, (source_size, target_size) in enumerate(buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break

        if max_size and counter >= max_size:
          break

        #Prepare next loop iteration
        source = source_file.readline()
        target = target_file.readline()

  return data_set, buckets