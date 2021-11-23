# Life is incomplete without this statement!
import tensorflow as tf

# And this as well!
import numpy as np

# To visualize results
import matplotlib.pyplot as plt

import os
import datetime

print(tf.__version__)

# Image size for our model.
MODEL_INPUT_IMAGE_SIZE = [ 200 , 200 ]

# Fraction of the dataset to be used for testing.
TRAIN_TEST_SPLIT = 0.3

# Number of samples to take from dataset
N = 20000

# This method will be mapped for each filename in `list_ds`. 
def parse_image( filename ):

    # Read the image from the filename and resize it.
    image_raw = tf.io.read_file( filename )
    image = tf.image.decode_jpeg( image_raw , channels=3 ) 
    image = tf.image.resize( image , MODEL_INPUT_IMAGE_SIZE ) / 255

    # Split the filename to get the age and the gender. Convert the age ( str ) and the gender ( str ) to dtype float32.
    parts = tf.strings.split( tf.strings.split( filename , '/' )[ 2 ] , '_' )

    # Normalize
    age = tf.strings.to_number( parts[ 0 ] )
    lower = ( age - 5 ) / 116
    upper = ( age + 5 ) / 116

    return image , [ lower , upper ]
    

# List all the image files in the given directory.
list_ds = tf.data.Dataset.list_files( './utkface_23k/*' , shuffle=True )

# Map `parse_image` method to all filenames.
dataset = list_ds.map( parse_image , num_parallel_calls=tf.data.AUTOTUNE )
dataset = dataset.take( N )


# Create train and test splits of the dataset.
num_examples_in_test_ds = int( dataset.cardinality().numpy() * TRAIN_TEST_SPLIT )

test_ds = dataset.take( num_examples_in_test_ds )
train_ds = dataset.skip( num_examples_in_test_ds )

print( 'Num examples in train ds {}'.format( train_ds.cardinality() ) )
print( 'Num examples in test ds {}'.format( test_ds.cardinality() ) )

# for image in dataset.batch( 1 ).take( 1 ):
#     image=image.
#     plt.imshow(image)


batch_size = 128

# Batch and repeat `train_ds` and `test_ds`.
# train_ds = train_ds.batch( batch_size )
# test_ds = test_ds.batch( batch_size )


model = tf.keras.models.load_model( 'my_model.h5' )
p = model.evaluate( test_ds.batch( batch_size ) )
print( p )
