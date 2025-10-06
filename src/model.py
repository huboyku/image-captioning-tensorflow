def buildModel(num_of_features, max_length, vocab_size, learning_rate,):
  ### ENCODER MODEL ###

  #image input 
  input_image = Input(shape= (num_of_features,)) # model.summary for Check output of i
  cnn_layer1 = tf.keras.layers.Dropout(.4)(input_image)
  cnn_layer2 = tf.keras.layers.Dense(256, activation='relu')(cnn_layer1)

  #sequence input 
  input_seq = tf.keras.layers.Input(shape=(MAX_LEN,))
  embedding_layer = tf.keras.layers.Embedding(vocab_size, 256, mask_zero=True)(input_seq) #dimension of  the seq
  drop_layer = tf.keras.layers.Dropout(.4)(embedding_layer)
  lstm_layer = tf.keras.layers.LSTM(256,activation='tanh')(drop_layer)


  ### DECODER MODEL ###

  decoder_merge = tf.keras.layers.add([cnn_layer2,lstm_layer])
  decoder_dense = tf.keras.layers.Dense(256, activation='relu')(decoder_merge)
  output = tf.keras.layers.Dense(vocab_size, activation='softmax')(decoder_dense)

  model = tf.keras.models.Model(inputs=[input_image, input_seq], outputs=output)
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  model.compile(loss='categorical_crossentropy', optimizer='adam')

  return model
