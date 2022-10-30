# Predictive Text Generation

# Import Libraries

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def tokenize_corpus(corpus):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    # pad sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # create predictors and label
    xs, labels = input_sequences[:, :-1], input_sequences[:, -1]

    ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

    return xs, ys, total_words, max_sequence_len, tokenizer


def model_generator(total_words, xs, ys, max_sequence_len, epoch):
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(150)))
    model.add(Dense(total_words, activation='softmax'))
    adam = Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    model.fit(xs, ys, epochs=epoch, verbose=1)
    return model


def to_text(tokenizer, model, max_sequence_len, seed_text, next_words=100):


    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=1), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


def to_corpus(text):
    corpus = text
    corpus = corpus.split(". ")
    corpus = [i.lower() for i in corpus]
    return corpus


def text_generator(data, text_question, groups, epoch, seed_text, next_words):
    text_predictions = []

    for group in groups:

        temp = data[text_question].loc[data['targetgroup'] == group]

        if (temp.empty):
            continue
        else:
            temp = temp.str.cat(sep=' ')
            corpus = to_corpus(temp)
            x, y, total_words, max_seq_len, tokenizer = tokenize_corpus(corpus)
            model = model_generator(total_words, x, y, max_seq_len, epoch)
            predicted_text = to_text(tokenizer, model, max_seq_len, seed_text, next_words)
            # print(group, '\n', predicted_text)
            text_predictions.append([group, text_question, seed_text, predicted_text])

    return text_predictions


def varied_seed_txt(data, textqs, groups, seed_text, epoch, next_words):
    predictions = pd.DataFrame()

    for i in range(len(textqs)):
        # print(textqs[i])
        pred_text = pd.DataFrame(text_generator(data, textqs[i], groups, epoch, seed_text[i], next_words))
        predictions = pd.concat([predictions, pred_text], axis=0)

    predictions = predictions.rename(columns={0: "Group", 1: "Question", 2: "Seed Text", 3: "Predicted Text"})

    return predictions


if __name__ == "__main__":

    filepath = Path('Research/cleaneddata1.csv')
    df = pd.read_csv(filepath)

    textqs = ['cw_maincause', 'cw_slaverysince','cw_causessince', 'cr_progress', 'cr_achieved', 'rr_legacy', 'rr_nolegacy']
    groups = ['White Conservative', 'White Liberal', 'Black Conservative', 'Black Non-Conservative']
    seed_text = ['the main cause of the civil war was', 'since the civil war slavery has been dealt with by', 'since the civil war the causes ofslavery have been dealt with by',
                'progress towards the main goals of the civil rights movement have been prevented by',
                'the goals of the civil rights movement have been achieved because',
                'the legacy of slavery and discrimination still affects people because',
                'the legacy of slavery and discrimination has little or no effect on people today because']
    predicted_text = varied_seed_txt(df, textqs, groups, seed_text, 100, 100)
    filepath = Path('varied_seed_predict.csv')
    predicted_text.to_csv(filepath, index=False)

