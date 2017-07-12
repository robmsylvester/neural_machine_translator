# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import tarfile

from six.moves import urllib
from tensorflow.python.platform import gfile
from collections import OrderedDict
import download_utils
import tensorflow as tf
import numpy as np

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


#Our vanilla tokenizer doesnt even use regex, because dealing with all the french characters
# in the regex ended up being slower than just writing out a list of seperators and looping through them.
# We already have text inputs that contain only lowercase words, so we simply split out on any and all
# punctuation that appears in the dataset.
def vanilla_ft_tokenizer(text):
    tokens = []
    seperators = ("(",
                  ")",
                  "?",
                  "!",
                  "@",
                  "#",
                  "$",
                  "%",
                  "^",
                  "&",
                  "*",
                  "[",
                  "]",
                  "'",
                  '"',
                  '~',
                  ".",
                  ",",
                  ":",
                  ";",
                  "-",
                  "_",
                  "=",
                  "+",
                  "{",
                  "}",
                  "<",
                  ">",
                  "/",
                  "\\",
                  "|",
                  "°")
    
    def split(txt, seps):

        for sep in seps:
            txt = txt.replace(sep, " "+sep+" ")
        return [i.strip() for i in txt.split(" ")]

    #split by spaces
    for split_by_space in text.strip().split():
        tokens.extend(split(split_by_space,seperators))
        
    return [i for i in tokens if i]




def clean_sentence(sentence, language="en"):

  #always lowercase for everything, but that doesn't capture all the characters in french, so we'll need to finish this up later explicitly.
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
    sentence = re.sub('Œ', 'œ', sentence)
    sentence = re.sub('0', ' zéro ', sentence)
    sentence = re.sub('1', ' un ', sentence)
    sentence = re.sub('2', ' deux ', sentence)
    sentence = re.sub('3', ' trois ', sentence)
    sentence = re.sub('4', ' quatre ', sentence)
    sentence = re.sub('5', ' cinq ', sentence)
    sentence = re.sub('6', ' six ', sentence)
    sentence = re.sub('7', ' sept ', sentence)
    sentence = re.sub('8', ' huit ', sentence)
    sentence = re.sub('9', ' neuf ', sentence)
  else:
    raise ValueError("clean_sentence() only has rules implemented for English and French. This isn't a horrible error. Basically,\
      write out commands for what to do with numbers. If using a FastText embedding, this means explicitly finding string representations\
       for the numbers in the language (ie, for english, '9' means ' nine '. Write out any explicit commands for dealing with lowercase accented characters that Python's lower()\
       will not take care of.")

  return sentence

#returns a python set with all characters appearing in a file. good for knowing how to tokenize
def get_all_chars_in_data_file(file_name, progress=1000000):
    charset = set()
    read = 0
    with open(file_name, 'rb') as f:
        for line in f:
            for ch in line:
                charset.add(ch)
            read += 1
            if read % progress == 0:
                print("read %d lines" % read)
    #print len(charset)
    #print("***************************")
    #for char in charset:
    #    print char
    return charset



def clean_enfr_wmt_data(output_file,
                    input_file,
                    language="en",
                    report_frequency=500000):

  """
  This will go through and lowercase everything, make every number written out (ie, 45 becomes four five, not forty-five) 
  We also explicitly have to lowercase those silly french letters with silly little baguettes over them.
  """
  if gfile.Exists(output_file):
    print("Cleaned dataset file %s detected. Skipping cleaning" % output_file)
    return output_file

  assert gfile.Exists(input_file), "Could not find dataset file %s to create cleaned dataset file" % input_file

  read_counter = 0

  with gfile.GFile(input_file,mode='rb') as f:
    with gfile.GFile(output_file,mode='w') as out:
      print("Cleaning dataset file %s to conform to translator conventions (ie, lowercase, proper tokenization, etc) using %s as the detected language.\nFor a large dataset like WMT, this might take an hour-plus. Go eat a sandwich." % (input_file, language))
      for line in f:
        read_counter += 1
        line = tf.compat.as_bytes(line)
        #print line

        #we need to take care of casing, particularly with foreign characters, among other things.
        line = clean_sentence(line, language=language)

        #Our word vector tokenizer in fasttext is admittedly shitty...looks for whitespace, and thats it.
        #So we will abstract this away by a tokenizer of our own and create an entirely new file
        #of tokens by a single white space.
        tokens = vanilla_ft_tokenizer(line)
        
        to_write = ''
        for token in tokens:
          to_write += token + ' ' #append a space after each word

        #We do NOT need to add an _EOS tag. We can just reference it in the trainer.
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
  #read sentence lengths from the file. does not take tokenized sentences, but a raw vocab file.
  #max_length is the max possible length a sentence can have. one more bin will be made above
  #this in the list to hold values more than max_length. holds a value for 0-length as well
  #returns the list of total lengths. 
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










def sentence_to_token_ids(sentence,
                          vocabulary,
                          unknown_words=None):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    unknown_words: a function to use to decide what to do with unknown words,
      perhaps it is truly best to represent them 

  Returns:
    a list of integers, the token-ids for the sentence.
  """

  words = sentence.split()
  return [vocabulary.get(w, UNK_ID) for w in words]








def integerize_sentences(data_path, target_path, vocabulary_path,
                        report_frequency=500000):
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
          token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")




def get_word_frequency_ratio(vocabulary, integerized_dataset_file_path, target_word, report_progress=2000000):
  try:
    target_index = vocabulary[target_word]
  except KeyError, e:
    raise("Could not find word %s in vocabulary dictionary" % target_word)

  print("Get word frequency will search for target word %s, which is integer %d" % (target_word, target_index))
  found = 0
  total = 0

  with open(integerized_dataset_file_path, "rb") as f:
    lines_read = 0
    for line in f:
      words = line.split()
      total += len(words)
      for w in words:
        if int(w) == target_index:
          found += 1
      lines_read += 1
      if lines_read % report_progress==0:
        print("...processed line %d, found %d occurrences of %s so far" % (lines_read, found, target_word))
  return (found, float(found) / total)






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
                 to_vocabulary_size, tokenizer=None, glove=False, word2vec=False, fasttext=False):
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
      A tuple of 9 elements:
        (1) path to the token-ids for "from language" training data-set,
        (2) path to the token-ids for "to language" training data-set,
        (3) path to the token-ids for "from language" development data-set,
        (4) path to the token-ids for "to language" development data-set,
        (5) path to the "from language" vocabulary file,
        (6) path to the "to language" vocabulary file.
        (7) path to the GloVe word embeddings created from the tokenized data, with size from_vocabulary_size and to_vocabulary_size
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

  #Now, we have a valid vocabulary that has been properly tokenized, so we need to run
  # some unsupervised learning algorithms

  # Integerize the training data by replacing words with their vocabulary representations (integers)
  # This will run only if the integerized version of the training set doesn't already exist
  to_train_ids_path = to_clean_train_path + (".ids_%d" % to_vocabulary_size)
  from_train_ids_path = from_clean_train_path + (".ids_%d" % from_vocabulary_size)
  integerize_sentences(to_clean_train_path, to_train_ids_path, to_vocab_path)
  integerize_sentences(from_clean_train_path, from_train_ids_path, from_vocab_path)


  # Create token ids for the development data.
  # This will run only if the integerized version of the dev set doesn't already exist
  to_dev_ids_path = to_dev_path + (".ids_%d" % to_vocabulary_size)
  from_dev_ids_path = from_dev_path + (".ids_%d" % from_vocabulary_size)
  integerize_sentences(to_clean_dev_path, to_dev_ids_path, to_vocab_path)
  integerize_sentences(from_clean_dev_path, from_dev_ids_path, from_vocab_path)

  # Stats - using 40,000 english and french words and default vanilla tokenizer gives about 1.4% english unknown and 1.7% french unknown words.
  # For reference, "the" occurs at about a 5% hit rate for the english dataset.
  return (from_train_ids_path, to_train_ids_path,
          from_dev_ids_path, to_dev_ids_path,
          from_vocab_path, to_vocab_path)




def load_dataset_in_memory(source_path,
                          target_path,
                          max_source_sentence_length,
                          max_target_sentence_length,
                          max_size=None,
                          ignore_lines=0,
                          report_frequency=200000):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with integer-like token-ids for the source language.
                 integer-like means they are still strings, obviously, but after being
                 processed through the integerizer
    target_path: path to the file with integer-like token-ids for the target language.
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_source_sentence_length: int - the max length of characters possible for a source sentence, others not used
    max_target_sentence_length: int - the max length of characters possible for a target sentence, others not used
    ignore_lines - integer, how many lines to ignore at the beginning of the file.
                  at times, it may be easier to train on a few million at a time.
                  then just stop the model and train on a different part of the data.
                  this will allow you to load it all in memory
    max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, all lines will be read.
    report_frequency: integer to specify to console how often to report progress in processing file

  Returns:
    data_set contains a list of
      (source, target) pairs read from the provided data files, but only storing up to max_sentence_length for the
      source and target. final symbol will always be _EOS
  """

  data_set = []
  bucketed_data_ratio = 0.

  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      
      counter = 0
      used_sentence_pairs = 0
      unused_sentence_pairs = 0 #too big to fit in max_size constraints

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
          print("\tloaded up through line %d in memory" % counter)

        #What we read is not quite an integer yet, so convert the string representations to actual integers.
        source_ids = [ int(x) for x in source.split()]
        target_ids = [ int(x) for x in target.split()]

        #We always add the end of sentence to the target sentence and do it here so that it doesn't
        #go above the max target size ever. Do this BEFORE checking the sizes.
        target_ids.append(EOS_ID)

        #Notice here that we use <= to source sentence, because we don't mess with it at all, and can feed these words into the neural network
        #However, target sentence uses <, and this is because we append a _GO symbol to it. This means the max target length could be too large
        if len(source_ids) <= max_source_sentence_length and len(target_ids) < max_target_sentence_length:
          data_set.append([source_ids, target_ids])
          used_sentence_pairs += 1
        else:
          unused_sentence_pairs += 1

        if max_size and counter >= max_size:
          break

        #Prepare next loop iteration
        source = source_file.readline()
        target = target_file.readline()

  bucketed_data_ratio = float(used_sentence_pairs) / (used_sentence_pairs+unused_sentence_pairs)
  return data_set, bucketed_data_ratio



def initialize_glove_embeddings_tensor(num_enc_symbols, embed_size, embedding_file, vocab_verification_file, dtype=None):

  if not gfile.Exists(embedding_file):
    raise IOError("Embedding file location %s not found" % embedding_file)
  if not gfile.Exists(vocab_verification_file):
    raise IOError("Vocab file location %s not found" % vocab_verification_file)

  embeddings = np.zeros(shape=(num_enc_symbols, embed_size), dtype=np.float32) #TODO - fix this dtype

  #PAD can remain all zeros, so let's just not deal with it.

  #TODO, for now, randomize the _GO symbols. alternatively could use glove with them and skip this part
  embeddings[1] = np.random.uniform(low=-1.0, high=1.0, size=embed_size)

  #randomize the _EOS symbols
  embeddings[2] = np.random.uniform(low=-1.0, high=1.0, size=embed_size)

  #randomize the _UNK symbols
  embeddings[3] = np.random.uniform(low=-1.0, high=1.0, size=embed_size)

  #load the pre-trained vectors
  limit = num_enc_symbols

  embed_dict = OrderedDict()
  with gfile.GFile(embedding_file, mode="rb") as ef:
    for line in ef:
      emb = line.split()
      vector = emb[1:] #0 is the word, 1: are the word vectors, of which there are 200-300 or 512 or something
      vector = [float(i) for i in vector]
      embed_dict[str(emb[0])] = vector

  with gfile.GFile(vocab_verification_file, mode="rb") as vf:

      print("Building and verifying glove embedding tensors against vocabulary files.")

      #get a word in the vocab file
      words_read=0
      for line in vf:
        word = line.split()[0]

        #skip the first four words
        if word in _INITIAL_VOCABULARY:
          words_read += 1
          assert words_read <= len(_INITIAL_VOCABULARY), "all initial vocabulary words are expected to come first in a vocab file."
          continue
        found=False

        #place the word vector in the numpy array at the right index
        try:
          embeddings[words_read] = embed_dict[word]
        except KeyError, err:
          print("The word %s did not occur in the embedding glove file" % word)
          print(err)
          raise
        
        words_read += 1

        #and quit if we hit our limit in the event that we are testing
        if limit == words_read:
          print("Cutting off vocabulary file at line limit %d. Size of dictionary is %d" % (limit, len(embeddings)))
          break

  return tf.convert_to_tensor(embeddings, dtype=dtype)