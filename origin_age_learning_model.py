
# Life is incomplete without this statement!
import tensorflow as tf
# And this as well!
import numpy as np
# To visualize results
#import matplotlib.pyplot as plt
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

for x , y in dataset.batch( 5 ).take( 2 ):
    print( y )


# Negative slope coefficient for LeakyReLU.
relu_alpha = 0.2

lite_model = True

# Define the conv block.
def conv( x , num_filters , kernel_size=( 3 , 3 ) , strides=1 ):
    if lite_model:
        x = tf.keras.layers.SeparableConv2D( num_filters ,
                                            kernel_size=kernel_size ,
                                            strides=strides, 
                                            use_bias=False ,
                                            kernel_initializer=tf.keras.initializers.HeNormal() ,
                                            kernel_regularizer=tf.keras.regularizers.L2( 1e-5 )
                                             )( x )
    else:
        x = tf.keras.layers.Conv2D( num_filters ,
                                   kernel_size=kernel_size ,
                                   strides=strides ,
                                   use_bias=False ,
                                   kernel_initializer=tf.keras.initializers.HeNormal() ,
                                   kernel_regularizer=tf.keras.regularizers.L2( 1e-5 )
                                    )( x )

    x = tf.keras.layers.BatchNormalization()( x )
    x = tf.keras.layers.PReLU( alpha_initializer=tf.initializers.constant(relu_alpha),shared_axes=[1, 2])(x)
    return x

def dense( x , filters , dropout_rate ):
    x = tf.keras.layers.Dense( filters , kernel_regularizer=tf.keras.regularizers.L2( 0.1 ) , bias_regularizer=tf.keras.regularizers.L2( 0.1 ) )( x )
    x = tf.keras.layers.PReLU( alpha_initializer=tf.initializers.constant(relu_alpha))( x )
    x = tf.keras.layers.Dropout( dropout_rate )( x )
    return x


# No. of convolution layers to be added.
num_blocks = 6
# Num filters for each conv layer.
num_filters = [ 16 , 32 , 64 , 128 , 256 , 256 ]
# Kernel sizes for each conv layer.
kernel_sizes = [ 3 , 3 , 3 , 3 , 3 , 3 ]

# Init a Input Layer.
inputs = tf.keras.layers.Input( shape=MODEL_INPUT_IMAGE_SIZE + [ 3 ] , name='input_image' )

# Add conv blocks sequentially
x = inputs
for i in range( num_blocks ):
    x = conv( x , num_filters=num_filters[ i ] , kernel_size=kernel_sizes[ i ] )
    x = tf.keras.layers.MaxPooling2D()( x )

# Flatten the output of the last Conv layer.
x = tf.keras.layers.Flatten()( x )
conv_output = x 

# Add Dense layers ( Dense -> LeakyReLU -> Dropout )
x1 = dense( conv_output , 256 , 0.6 )
x1 = dense( x1 , 64 , 0.4 )
x1 = dense( x1 , 32 , 0.2 )
outputs_lower = tf.keras.layers.Dense( 1 , activation='relu' , name='output_lower_bound' )( x1 )


x2 = dense( conv_output , 256 , 0.6 )
x2 = dense( x2 , 64 , 0.4 )
x2 = dense( x2 , 32 , 0.2 )
outputs_upper = tf.keras.layers.Dense( 1 , activation='relu' , name='output_upper_bound' )( x2 )

# Build the Model
model = tf.keras.models.Model( inputs , [ outputs_lower , outputs_upper ] )

# Uncomment the below to view the summary of the model.
model.summary()
# tf.keras.utils.plot_model( model , to_file='architecture.png' )




# Initial learning rate
learning_rate = 0.001

num_epochs = 5 #@param {type: "number"}
batch_size = 128 #@param {type: "number"}
# batch_size = 128 #@param {type: "number"}

# Batch and repeat `train_ds` and `test_ds`.
train_ds = train_ds.batch( batch_size )
test_ds = test_ds.batch( batch_size )

# Init ModelCheckpoint callback
save_dir_ = 'model_1'  #@param {type: "string"}
save_dir = save_dir_ + '/{epoch:02d}-{val_loss:.2f}.h5'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint( 
    save_dir , 
    save_best_only=True , 
    monitor='val_loss' , 
    mode='min' , 
)

tb_log_name = 'model_1'  #@param {type: "string"}
# Init TensorBoard Callback
logdir = os.path.join( "tb_logs" , tb_log_name )
tensorboard_callback = tf.keras.callbacks.TensorBoard( logdir )

# Init LR Scheduler
def scheduler( epochs , learning_rate ):
    if epochs < num_epochs * 0.25:
        return learning_rate
    elif epochs < num_epochs * 0.5:
        return 0.0005
    elif epochs < num_epochs * 0.75:
        return 0.0001
    else:
        return 0.000095

lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler( scheduler )

# Init Early Stopping callback
early_stopping_callback = tf.keras.callbacks.EarlyStopping( monitor='val_loss' , patience=10 )

# Compile the model
model.compile( 
    loss=tf.keras.losses.mean_absolute_error ,
    optimizer = tf.keras.optimizers.Adam( learning_rate ) , 
    metrics=[ 'mae' ]
)



model.fit( 
    train_ds, 
    epochs=num_epochs,  
    validation_data=test_ds, 
    callbacks=[ checkpoint_callback , tensorboard_callback , lr_schedule_callback , early_stopping_callback ]
)


p = model.evaluate( test_ds )
print( p )


model.save("my_model.h5")





"""

def representative_data_gen():
    dataset_list = tf.data.Dataset.list_files('./utkface_23k/*')
    for i in range(100):
        image = next(iter(dataset_list))
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (128,128))
        image = tf.cast(image / 255., tf.float32)
        image = tf.expand_dims(image, 0)
# Model has only one input so each data point has one element
    yield [image]
converter =tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model=converter.convert()


with open('model.tflite', 'wb') as f:
    f.write(tflite_model)


"""

# model_name = 'model_gender_augmented' #@param {type: "string"}
# model_name_ = model_name + '.h5'


# converter = tf.lite.TFLiteConverter.from_keras_model( model )
# converter.optimizations = [ tf.lite.Optimize.DEFAULT ]
# converter.target_spec.supported_types = [ tf.float16 ]
# buffer = converter.convert()

# open( '{}_q.tflite'.format( "work" ) , 'wb' ).write( buffer )

# model.save("my_model")