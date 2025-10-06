from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
import os
def extract_features(img_folder, image_ids, target_size=(299, 299)):
    # Load pre-trained InceptionV3 model
    model = InceptionV3(weights='imagenet')
    #restructure model (remove top classification layer)
    model = tf.keras.Model(model.input, model.layers[-2].output)  # Global Average Pooling
    print(model.summary())

    #extract features from image
    features_dict = {}
    #for img_name in tqdm(os.listdir(img_folder)):
    for img_id in tqdm(image_ids, desc="Extracting features"):
        filename = os.path.join(img_folder, img_id)

        try:
            #load image from file
            img = image.load_img(filename, target_size=target_size)
            #convert image pixels to numpy array
            img_array = image.img_to_array(img)
            #reshape data for model
            #img_array = img_array.reshape((1,img_array.shape[0], img_array.shape[1], img_array.shape(2)))
            img_array = tf.expand_dims(img_array, axis=0)
            #prepocess for the model
            img_array = preprocess_input(img_array)
            # extract and store features
            features = model.predict(img_array, verbose=0)
            features_dict[img_id] = tf.squeeze(features)

        except Exception as e:
            print(f"Error processing {img_id}: {e}")
            continue

    return features_dict