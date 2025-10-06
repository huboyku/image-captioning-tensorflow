import numpy as np
import tensorflow as tf
import random

def data_generator(
    data_keys,
    image_to_captions_mapping,
    features,
    vectorizer,
    max_caption_length,
    vocab_size,
    batch_size
):
    X1_batch, X2_batch, y_batch = [], [], []
    last_image_id = None  # To keep track of the most recent image ID
    count = 0 
    n=0
    while True:
        count = 0
        shuffled_keys = random.sample(data_keys, len(data_keys))
        print("Shuffled")
        #Dprint(len(data_keys))
        for image_id in shuffled_keys:
            if image_id not in features:
                print(f"Warning: image_id {image_id} not found in features. Skipping!")
                continue  # skip this image
            
            last_image_id = image_id
            captions = image_to_captions_mapping[image_id]
            count = count +1
            for caption_seq in captions:
                for i in range(1, len(caption_seq)):
                    in_seq, out_seq = caption_seq[:i], caption_seq[i]
                    in_seq = tf.keras.preprocessing.sequence.pad_sequences(
                        [in_seq], maxlen=max_caption_length, padding='post')[0]
                    out_seq = tf.keras.utils.to_categorical(
                        out_seq, num_classes=vocab_size)
                    X1_batch.append(features[image_id])
                    X2_batch.append(in_seq)
                    y_batch.append(out_seq)
                    if len(X1_batch) == batch_size:
                        n = n+1 
                        #if n== 1000:
                            #print(f"count: {count}")
                        #print("Yielding batch shapes:",
                        #      np.array(X1_batch).shape,
                        #      np.array(X2_batch).shape,
                        #      np.array(y_batch).shape)
                        # Yield batch
                        yield (
                            {"input_1": np.array(X1_batch, dtype=np.float32),
                             "input_2": np.array(X2_batch, dtype=np.int32)},
                            np.array(y_batch, dtype=np.float32)
                        )
                        X1_batch, X2_batch, y_batch = [], [], []
        
        # After finishing all data_keys, yield remaining (possibly partial) batch
        if len(X1_batch) > 0:
            print("Yielding batch with shape:", np.array(X1_batch).shape)
            yield (
                {"input_1": np.array(X1_batch, dtype=np.float32),
                 "input_2": np.array(X2_batch, dtype=np.int32)},
                np.array(y_batch, dtype=np.float32)
            )
            X1_batch, X2_batch, y_batch = [], [], []
        else:
            print(f"Did not yield: X1_batch is empty at end of data_keys loop! Last image_id processed: {last_image_id}")
        
    
    
def data_generator2(
    data_keys,
    image_to_captions_mapping,
    features,
    vectorizer,
    max_caption_length,
    vocab_size,
    batch_size
):
    X1_batch, X2_batch, y_batch = [], [], []
    last_image_id = None  # To keep track of the most recent image ID
    count = 0 
    while True:
        count = 0
        shuffled_keys = random.sample(data_keys, len(data_keys))
        #print(len(data_keys))
        #print("Shuffled")
        for image_id in shuffled_keys:
            if image_id not in features:
                print(f"Warning: image_id {image_id} not found in features. Skipping!")
                continue  # skip this image

            last_image_id = image_id
            captions = image_to_captions_mapping[image_id]
            count = count +1
            for caption_seq in captions:
                for i in range(1, len(caption_seq)):
                    in_seq, out_seq = caption_seq[:i], caption_seq[i]
                    in_seq = tf.keras.preprocessing.sequence.pad_sequences(
                        [in_seq], maxlen=max_caption_length, padding='post')[0]
                    out_seq = tf.keras.utils.to_categorical(
                        out_seq, num_classes=vocab_size)
                    X1_batch.append(features[image_id])
                    X2_batch.append(in_seq)
                    y_batch.append(out_seq)
                    if len(X1_batch) == batch_size:
                        #print(f"countval: {count}")
                        #print("Yielding batch shapes:",
                        #      np.array(X1_batch).shape,
                        #      np.array(X2_batch).shape,
                        #      np.array(y_batch).shape)
                        # Yield batch
                        yield (
                            {"input_1": np.array(X1_batch, dtype=np.float32),
                             "input_2": np.array(X2_batch, dtype=np.int32)},
                            np.array(y_batch, dtype=np.float32)
                        )
                        X1_batch, X2_batch, y_batch = [], [], []

        # After finishing all data_keys, yield remaining (possibly partial) batch
        if len(X1_batch) > 0:
            print("Yielding batch with shape:", np.array(X1_batch).shape)
            yield (
                {"input_1": np.array(X1_batch, dtype=np.float32),
                 "input_2": np.array(X2_batch, dtype=np.int32)},
                np.array(y_batch, dtype=np.float32)
            )
            X1_batch, X2_batch, y_batch = [], [], []
        else:
            print(f"Did not yield: X1_batch is empty at end of data_keys loop! Last image_id processed: {last_image_id}")