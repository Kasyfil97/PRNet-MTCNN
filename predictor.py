import tensorflow as tf
import numpy as np

def resBlock(x, num_outputs, 
             kernel_size=4, 
             stride=1, 
             activation_fn=tf.nn.relu, 
             use_batch_norm=True):
    assert num_outputs % 2 == 0  # num_outputs must be divided by channel_factor(2 here)
    
    shortcut = x
    if stride != 1 or x.shape[-1] != num_outputs:
        shortcut = tf.keras.layers.Conv2D(
            num_outputs, kernel_size=1, strides=stride,
            padding='SAME', use_bias=False)(shortcut)
    
    x = tf.keras.layers.Conv2D(
        num_outputs//2, kernel_size=1, strides=1,
        padding='SAME', use_bias=False)(x)
    x = tf.keras.layers.Conv2D(
        num_outputs//2, kernel_size=kernel_size, strides=stride,
        padding='SAME', use_bias=False)(x)
    x = tf.keras.layers.Conv2D(
        num_outputs, kernel_size=1, strides=1,
        padding='SAME', use_bias=False)(x)
    x = tf.keras.layers.Add()([shortcut, x])
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = activation_fn(x)
    
    return x

class ResFCN256(tf.keras.Model):
    def __init__(self, resolution_inp=256, resolution_op=256, channel=3, name='resfcn256'):
        super(ResFCN256, self).__init__(name=name)
        self.channel = channel
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        
        # Define regularizer
        self.regularizer = tf.keras.regularizers.l2(0.0002)
        
        # Define layers
        size = 16
        self.conv1 = tf.keras.layers.Conv2D(size, kernel_size=4, strides=1, padding='SAME',
                                          kernel_regularizer=self.regularizer)
        
        # Encoder layers
        self.encoder_layers = []
        sizes = [size * mult for mult in [2, 2, 4, 4, 8, 8, 16, 16, 32, 32]]
        strides = [2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
        for s, stride in zip(sizes, strides):
            self.encoder_layers.append(
                lambda x, s=s, stride=stride: resBlock(x, s, kernel_size=4, stride=stride))
        
        # Decoder layers
        self.decoder_layers = []
        sizes = [size * mult for mult in [32, 16, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 1, 1]]
        strides = [1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1]
        for s, stride in zip(sizes, strides):
            self.decoder_layers.append(
                tf.keras.layers.Conv2DTranspose(s, kernel_size=4, strides=stride, padding='SAME',
                                              kernel_regularizer=self.regularizer))
        
        # Final layers
        self.conv_final1 = tf.keras.layers.Conv2DTranspose(3, kernel_size=4, strides=1, padding='SAME',
                                                          kernel_regularizer=self.regularizer)
        self.conv_final2 = tf.keras.layers.Conv2DTranspose(3, kernel_size=4, strides=1, padding='SAME',
                                                          kernel_regularizer=self.regularizer)
        self.conv_final3 = tf.keras.layers.Conv2DTranspose(3, kernel_size=4, strides=1, padding='SAME',
                                                          activation='sigmoid',
                                                          kernel_regularizer=self.regularizer)
        
    def call(self, x, training=True):
        # Initial conv
        x = self.conv1(x)
        
        # Encoder
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Decoder
        for layer in self.decoder_layers:
            x = layer(x)
            if training:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.nn.relu(x)
        
        # Final convolutions
        x = self.conv_final1(x)
        x = self.conv_final2(x)
        x = self.conv_final3(x)
        
        return x

class PosPrediction:
    def __init__(self, resolution_inp=256, resolution_op=256):
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.MaxPos = resolution_inp * 1.1
        
        # Model akan di-load saat restore dipanggil
        self.model = None
        self.serving_fn = None
    
    def restore(self, model_path):
        self.model = tf.saved_model.load(model_path)
        self.serving_fn = self.model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    def predict(self, image):
        if self.serving_fn is None:
            raise ValueError("Model belum di-load. Panggil restore() terlebih dahulu.")
        
        # Prepare input
        if len(image.shape) == 3:
            image = image[np.newaxis, ...]
        
        # Convert ke tensor
        input_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        
        # Predict
        output = self.serving_fn(input_image=input_tensor)
        prediction = output['prediction'].numpy()
        prediction = np.squeeze(prediction)
        
        return prediction * self.MaxPos
    
    def predict_batch(self, images):
        pos = self.network(images, training=False)
        return pos.numpy() * self.MaxPos