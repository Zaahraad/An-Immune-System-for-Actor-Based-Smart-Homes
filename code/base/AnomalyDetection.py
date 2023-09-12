import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam

class AnomalyDetection:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.MinMaxScaler = MinMaxScaler()

    
    def prepare4Seq2Seq(self, data):
        # Flatten the data and create vocabulary
        all_words = [word for sublist in data for word in sublist]
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_words)
        vocab_size = len(tokenizer.word_index) + 1

        # Prepare encoder input sequences
        encoder_sequences = tokenizer.texts_to_sequences(data)
        max_sequence_length = max(len(seq) for seq in encoder_sequences)
        encoder_input_data = pad_sequences(encoder_sequences, maxlen=max_sequence_length, padding='post')

        # Prepare decoder input and target sequences
        decoder_input_data = [seq[:-1] for seq in data]  # Remove the last element from each sequence
        decoder_target_data = [seq[1:] + [''] for seq in data]  # Shift decoder target by one step
        decoder_input_data = tokenizer.texts_to_sequences(decoder_input_data)
        decoder_target_data = tokenizer.texts_to_sequences(decoder_target_data)
        decoder_input_data = pad_sequences(decoder_input_data, maxlen=max_sequence_length - 1, padding='post')
        decoder_target_data = pad_sequences(decoder_target_data, maxlen=max_sequence_length - 1, padding='post')

        # Convert data to NumPy arrays
        encoder_input_data = np.array(encoder_input_data)
        decoder_input_data = np.array(decoder_input_data)
        decoder_target_data = np.array(decoder_target_data)
        
        return (max_sequence_length, vocab_size, encoder_input_data, decoder_input_data, decoder_target_data)
    
    def seq2seq(self, max_sequence_length, vocab_size, encoder_input_data, decoder_input_data, 
                decoder_target_data, embedding_dim, hidden_units, activation_function, 
                loss_function, metrics, epochs, batch_size, validation_split):
        
        
        encoder_input = Input(shape=(max_sequence_length,))
        encoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(encoder_input)
        encoder_lstm = LSTM(hidden_units, return_state=True)
        encoder_output, state_h, state_c = encoder_lstm(encoder_embedding)

        decoder_input = Input(shape=(max_sequence_length - 1,))
        decoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(decoder_input)
        decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
        decoder_output, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])
        decoder_dense = Dense(vocab_size, activation=activation_function)
        output = decoder_dense(decoder_output)

        model = Model(inputs=[encoder_input, decoder_input], outputs=output)

        # Compile the model
        model.compile(optimizer=Adam(), loss=loss_function, metrics=[metrics])

        # Train the model
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=[early_stopping])

        return (history, model)

    def predict_miss_sensor(self, data, model, max_sequence_length):
        # Flatten the data and create vocabulary
        all_words = [word for sublist in data for word in sublist]

        # Assuming you already have a tokenizer for training data
        keras_tokenizer = Tokenizer()
        keras_tokenizer.fit_on_texts(all_words)
        vocab_size = len(keras_tokenizer.word_index) + 1

        # Prepare test input sequences
        tokenized_test_input = [keras_tokenizer.texts_to_sequences([' '.join(sentence)])[0] for sentence in data]
        # max_sequence_length = max(len(seq) for seq in tokenized_test_input)
        input_sequences_padded_test = pad_sequences(tokenized_test_input, maxlen=max_sequence_length, padding='post')

        # Prepare test decoder input sequences
        decoder_input_sequences_test = []
        for seq in tokenized_test_input:
            seq_indices = [keras_tokenizer.word_index[word] if word in keras_tokenizer.word_index else keras_tokenizer.word_index['null'] for word in seq]
            decoder_input_sequences_test.append(seq_indices)

        # Pad or truncate test decoder input sequences
        decoder_input_sequences_padded_test = pad_sequences(decoder_input_sequences_test, maxlen=max_sequence_length - 1, padding='post')

        # Use the trained model to predict
        predictions_test = model.predict([input_sequences_padded_test, decoder_input_sequences_padded_test])

        # Decode the predictions
        predicted_tokens_test = np.argmax(predictions_test, axis=-1)
        predicted_words_test = []

        for token_list in predicted_tokens_test:
            word_list = []
            for token_idx in token_list:
                if token_idx in keras_tokenizer.index_word:
                    word_list.append(keras_tokenizer.index_word[token_idx])
            predicted_words_test.append(word_list)

        print(predicted_words_test)
        
        return predicted_words_test
              
    def prepare4autoencoder(self, data, group_by, command_column, command_1, command_2):
        
        group_df= data.groupby(by=group_by).agg({command_column: lambda x: " ".join(x)})
        group_df[command_column] = group_df[command_column].replace("(\w*\:)", '', regex=True)
        group_df[command_column] = group_df[command_column].str.split()
        
        def convert_entry(entry):
            if isinstance(entry, list):
                numeric_entry = []
                
                for item in entry:
                    if isinstance(item, (int, float)):
                        numeric_entry.append(item)
                    elif isinstance(item, str):
                        numeric_entry.append(1 if item == command_1 else (0 if item == command_2 else item))
                
                return numeric_entry
            elif isinstance(entry, str):
                return self.convert_entry(eval(entry))
            else:
                return []
        
        group_df[command_column] = group_df[command_column].apply(convert_entry)

        df = group_df.copy()
        converted_type_df = [[float(item) if isinstance(item, str) and item.replace('.', '', 1).isdigit() else item for item in commad] for commad in df[command_column]]
        df[command_column] = converted_type_df
        
        numerical_data = df[command_column].apply(lambda x: [item for item in x if isinstance(item, (int, float))])
        normalized_numerical= self.scaler.fit_transform(numerical_data.apply(pd.Series))
        numeric_data_imputed = self.imputer.fit_transform(normalized_numerical)
        
        return (df, numeric_data_imputed)
        
        
    def autoencoders4anomaly(self, data, encoding_dim, l2_regularization, encode_activation, decode_activation, optimizer, loss,
                             num_epochs, batch_size):
        # Build the autoencoder model without additional hidden layers but with regularization
        input_dim = data.shape[1]
        l2_regularization = l2_regularization

        # Encoder
        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(encoding_dim, activation=encode_activation)(input_layer)

        # Decoder
        decoded = layers.Dense(input_dim, activation=decode_activation)(encoded)

        autoencoder = models.Model(input_layer, decoded)
        autoencoder.compile(optimizer=optimizer, loss=loss)

        # Early stopping
        early_stopping = callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        history = autoencoder.fit(data, data, epochs=num_epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping])
        
        return (history, autoencoder)
    
    def detect_anoamly(self, reconstruction_errors_data, threshold, data):
        # Perform anomaly detection
        anomalies = []
        for i, reconstruction_error in enumerate(reconstruction_errors_data):
            if reconstruction_error > threshold :
                anomalies.append((i, data.iloc[i]['Command_v1']))

        print(len(anomalies))
        # Print detected anomalies
        if len(anomalies) > 0:
            print("Detected anomalies:")
            for i, anomaly in anomalies:
                print(f"Index: {i}, Data: {anomaly}")
                
        else:
            print("No anomalies detected.")
        
        return anomalies
    
    
        