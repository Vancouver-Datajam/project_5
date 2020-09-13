# Serve model as a flask application
from flask import Flask, request, render_template, redirect, url_for
import urllib.request
from urllib.error import HTTPError
import json 
import subprocess
import os
import tensorflow as tf
import numpy as np

app = Flask(__name__)

char_to_index = {'\n': 0,
 ' ': 1,
 '!': 2,
 '"': 3,
 '#': 4,
 '$': 5,
 '%': 6,
 '&': 7,
 "'": 8,
 '(': 9,
 ')': 10,
 '*': 11,
 '+': 12,
 ',': 13,
 '-': 14,
 '.': 15,
 '/': 16,
 '0': 17,
 '1': 18,
 '2': 19,
 '3': 20,
 '4': 21,
 '5': 22,
 '6': 23,
 '7': 24,
 '8': 25,
 '9': 26,
 ':': 27,
 ';': 28,
 '=': 29,
 '?': 30,
 '@': 31,
 'A': 32,
 'B': 33,
 'C': 34,
 'D': 35,
 'E': 36,
 'F': 37,
 'G': 38,
 'H': 39,
 'I': 40,
 'J': 41,
 'K': 42,
 'L': 43,
 'M': 44,
 'N': 45,
 'O': 46,
 'P': 47,
 'Q': 48,
 'R': 49,
 'S': 50,
 'T': 51,
 'U': 52,
 'V': 53,
 'W': 54,
 'X': 55,
 'Y': 56,
 'Z': 57,
 '[': 58,
 ']': 59,
 '^': 60,
 '_': 61,
 'a': 62,
 'b': 63,
 'c': 64,
 'd': 65,
 'e': 66,
 'f': 67,
 'g': 68,
 'h': 69,
 'i': 70,
 'j': 71,
 'k': 72,
 'l': 73,
 'm': 74,
 'n': 75,
 'o': 76,
 'p': 77,
 'q': 78,
 'r': 79,
 's': 80,
 't': 81,
 'u': 82,
 'v': 83,
 'w': 84,
 'x': 85,
 'y': 86,
 'z': 87,
 '{': 88,
 '|': 89,
 '}': 90,
 '~': 91}


index_to_char = np.array(['\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+',
       ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
       '9', ':', ';', '=', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
       'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
       'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '^', '_', 'a', 'b', 'c',
       'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
       'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}',
       '~'])

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    """Define the model: character embedding -> GRU -> fully connected """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


def generate_text(model, start_string, num_generate=280):
    """Generate text, given a trained model and a starting string"""
    input_eval = [char_to_index[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(index_to_char[predicted_id])

    return start_string + ''.join(text_generated)

@app.route('/')
def home():
    return render_template('index.html')

# Pretty print JSON content
# Receives a URL for JSON and returns a pretty print version
@app.route('/tweet', methods=['POST'])
def tweet_result():
    if request.method == 'POST':
        word = request.form['start_word']

        checkpoint_dir = './training_checkpoints'
        vocab_size=92
        embedding_dim = 64
        rnn_units = 512

        model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        model.build(tf.TensorShape([1, None]))
        gen_tweet = generate_text(model, start_string=word, num_generate=280)

        return render_template('tweet_viewer.html', gen_tweet = gen_tweet)

@app.route('/<others>')
def other(others):
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0', debug=True, port=80)