import tensorflow as tf
from keras import layers, Model

seed=42
rng = tf.random.Generator.from_seed(seed, alg='philox')

class Vgg_Model(Model):

    def __init__(self):
        super(Vgg_Model, self).__init__()

        self.flatten=layers.Flatten()

        self.c1=layers.Conv2D(16,3,strides=(1,1),activation="relu",kernel_regularizer='l2')
        self.c2=layers.Conv2D(16,3,strides=(2,2),activation="relu",kernel_regularizer='l2')
        
        self.c3_a=layers.Conv2D(32,3,strides=(1,1),activation="relu",kernel_regularizer='l2',name="c3_a")
        self.c4_a=layers.Conv2D(32,3,strides=(2,2),activation="relu",kernel_regularizer='l2',name="c4_a")
        self.c5_a=layers.Conv2D(64,3,strides=(1,1),activation="relu",kernel_regularizer='l2',name="c5_a")
        self.c6_a=layers.Conv2D(64,3,strides=(1,1),activation="relu",kernel_regularizer='l2',name="c6_a")
        
        self.d1_a=layers.Dense(128,activation="relu",name="d1_a")
        self.d2_a=layers.Dense(2,activation="softmax",name="d2_a")

        self.c3_b=layers.Conv2D(32,3,strides=(1,1),activation="relu",kernel_regularizer='l2',name="c3_b")
        self.c4_b=layers.Conv2D(32,3,strides=(2,2),activation="relu",kernel_regularizer='l2',name="c4_b")
        self.c5_b=layers.Conv2D(64,3,strides=(1,1),activation="relu",kernel_regularizer='l2',name="c5_b")
        self.c6_b=layers.Conv2D(64,3,strides=(1,1),activation="relu",kernel_regularizer='l2',name="c6_b")

        self.d1_b=layers.Dense(128,activation="relu",name="d1_b")
        self.d2_b=layers.Dense(1,activation="relu",name="d2_b")

    def call(self,input,training=None, **kwargs):
        
        x=self.c1(input)
        x=self.c2(x)
        
        x_a=self.c3_a(x)
        x_a=self.c4_a(x_a)
        x_a=self.c5_a(x_a)
        x_a=self.c6_a(x_a)
        x_a=self.flatten(x_a)
        x_a=self.d1_a(x_a)
        x_a=self.d2_a(x_a)

        x_b=self.c3_b(x)
        x_b=self.c4_b(x_b)
        x_b=self.c5_b(x_b)
        x_b=self.c6_b(x_b)
        x_b=self.flatten(x_b)
        x_b=self.d1_b(x_b)
        x_b=self.d2_b(x_b)

        return (x_a, x_b)