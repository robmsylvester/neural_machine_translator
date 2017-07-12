<h3>Neural Machine Translation Suite</h3>

<p>This project is aimed at combining many of the strategies in different academic papers for
implemeting Neural Machine Translators in Tensorflow. The project is written with Tensorflow 1.0,
but is aiming toward support for the newer API as well. It began by working with the original
seq2seq Tensorflow tutorial, but slowly has evolved into its own beast with many more features. There are still some legacy API calls that are taken from the original seq2seq tutorials, and some of the original code, so I have left the tensorflow copyright in some of the files.</p>

<p>Current Status: Unstable for Production, Stable for Development.
This means you can download it and change a few directories and run it, with maybe a hiccup or two</p>

<p>Features</p>
<ul>
<li>Support for Arbitrary Encoder and Decoder RNN Geometries</li>
<li>Support for Arbitrary Residual Connections</li>
<li>Support for multiple Attention Mechanism implementations</li>
<li>Network and Unsupervised (Glove) Embedding Support</li>
<li>Samples Softmax Loss and Cross-Entropy Support </li>
<li>Lots more! See the Flags file</li>
</ul>

<p>Features Coming Soon:</p>
<ul>
<li>Beam Search Decoding</li>
<li>Boosted Vocabulary Perplexity Analysis</li>
<li>FastText and Word2Vec Unsupervised Embedding Support</li>
<li>More State Value Initializer Functions (Nematus, etc) </li>
</ul>

<p>How does it work?</p>
<p>Running experiment.py will instantiate the flags file and check the json architecture defined in
the encoder_decoder architecture flag file location. The program will search for the dataset, WMT15
by default, with the data directory and filenames provided. Afterward, it will tokenize and integerize the data according to several modifiable parameters. The model will then begin to train, at which point it will save its parameters every time it reaches a lowest validation set score. The model can be restarted in training from this checkpoint file, or alternatively can
be executed with a live decoding option</p>

<Libraries>
<ul>
<li>tensorflow = 1.0</li>
<li>numpy >= 1.12.0
