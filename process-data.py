import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm

trainX, trainy = [], []
testX, testy = [], []

try:
    trainX = np.load('trainX.npy')
    trainy = np.load('trainy.npy')
    testX = np.load('testX.npy')
    testy = np.load('testy.npy')
    print("Preprocessed data loaded from disk.")
except FileNotFoundError: 
    print("Preprocessing data, this may take some time...")
    print("If this is the first run after a code change, or if errors persist,")
    print("you might need to delete the existing dataset directory:")
    print("rm -rf /home/user/tensorflow_datasets/emnist/byclass/3.1.0") 

    builder = tfds.builder('emnist/byclass')

    builder.download_and_prepare(file_format='array_record')

    ds_train_source = builder.as_data_source(split='train')
    ds_test_source = builder.as_data_source(split='test')

    features = builder.info.features
    image_spec = tf.TensorSpec(shape=features['image'].shape, dtype=features['image'].dtype)
    label_spec = tf.TensorSpec(shape=(), dtype=features['label'].dtype)

    ds_train = tf.data.Dataset.from_generator(
        lambda: ds_train_source,
        output_signature={
            'image': image_spec,
            'label': label_spec
        }
    )

    ds_test = tf.data.Dataset.from_generator(
        lambda: ds_test_source,
        output_signature={
            'image': image_spec,
            'label': label_spec
        }
    )

    ds_train = ds_train.map(lambda x: {
        'image': tf.transpose(x['image'], perm=[1, 0, 2]),
        'label': x['label']
    })

    ds_test = ds_test.map(lambda x: {
        'image': tf.transpose(x['image'], perm=[1, 0, 2]),
        'label': x['label']
    })

    for example in tqdm(ds_train, desc="Processing training data", total=builder.info.splits['train'].num_examples): 
        trainX.append(example['image'])
        trainy.append(example['label'])

    trainX = np.array(trainX)
    trainy = np.array(trainy)

    for example in tqdm(ds_test, desc="Processing test data", total=builder.info.splits['test'].num_examples): 
        testX.append(example['image'])
        testy.append(example['label'])

    testX = np.array(testX)
    testy = np.array(testy)

    np.save('trainX.npy', trainX)
    np.save('trainy.npy', trainy)
    np.save('testX.npy', testX)
    np.save('testy.npy', testy)
    print("Preprocessed data saved to disk.")