import tensorflow as tf


def parse_captions(path):
    #read file, split into dict
    imageCapDict = {}
    with open(path,"r") as f:
        for i, line in enumerate(f):
            tokens =line.split("\t")
            imageId = tokens[0].split("#")[0]
            caption = tokens[1].strip()
            if imageId not in imageCapDict:
                imageCapDict[imageId] = [caption]
            else:
                imageCapDict[imageId].append(caption)
    return imageCapDict
    
    
    import tensorflow as tf

def train_val_test_datasets(
    dataset, 
    train_split=0.8, 
    val_split=0.1, 
    test_split=0.1, 
    batch_size=None, 
    seed=42
):
    """
    Splits a tf.data.Dataset into train, val, test.
    Returns unbatched datasets if batch_size=None.
    Otherwise, returns batched datasets.
    """
    assert abs((train_split + val_split + test_split) - 1.0) < 1e-6, "Splits must sum to 1"

    total_size = tf.data.experimental.cardinality(dataset).numpy()
    assert total_size > 0, "Dataset must not be empty"

    # Shuffle for reproducibility
    dataset = dataset.shuffle(buffer_size=total_size, seed=seed)

    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size)

    if batch_size is not None:
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset
