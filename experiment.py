import tensorflow as tf
import translate
import flags

def main(_):

  flags.flag_test()

  if tf.app.flags.FLAGS.decode:
    translate.decode()
  else:
    translate.train()

if __name__ == "__main__":
  tf.app.run() #initialize flags
