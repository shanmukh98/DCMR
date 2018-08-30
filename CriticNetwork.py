import numpy as np
import math
from keras.initializers import normal, identity
from keras.models import model_from_json, load_model
# from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import *
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600


class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        K.set_session(sess)

        # Now create the model
        self.model, self.action, self.state = self.create_critic_network(
            state_size, action_size)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(
            state_size, action_size)
        self.action_grads = tf.gradients(
            self.model.output, self.action)  # GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + \
                (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self,state_size,action_size):
        print("Now we build the model")
        pos_1 = Input(shape=(7,))
        pos_2 = Input(shape=(7,))
        vel_1 = Input(shape=(6,))
        vel_2 = Input(shape=(6,))
        img_1 = Input(shape=(256,256,3))
        act = Input(shape=(4,))

        img = BatchNormalization()(img_1)
        img = Conv2D(16,activation="relu",kernel_size=(5,5),strides=(2,2))(img)

        img = Conv2D(32,activation="relu",kernel_size=(3,3),strides=(2,2))(img)
        img = Conv2D(64,activation="relu",kernel_size=(3,3))(img)
        img = BatchNormalization()(img)


        img = Conv2D(64,activation="relu",kernel_size=(3,3),strides=(2,2))(img)
        img = Conv2D(64,activation="relu",kernel_size=(3,3))(img)
        img = BatchNormalization()(img)


        img = Conv2D(64,activation="relu",kernel_size=(3,3),strides=(2,2))(img)
        img = Conv2D(64,activation="relu",kernel_size=(3,3))(img)
        img = BatchNormalization()(img)


        img = Conv2D(128,activation="relu",kernel_size=(3,3),strides=(2,2))(img)
        img = Conv2D(128,activation="relu",kernel_size=(3,3))(img)
        img = BatchNormalization()(img)

        img = Flatten()(img)





        state = concatenate([pos_1,pos_2,vel_1,vel_2,act])

        state = Dense(32,activation="relu")(state)
        state = BatchNormalization()(state)

        state = Dropout(0.25)(state)
        state = Dense(64,activation="relu")(state)

        x = concatenate([img,state])
        x = BatchNormalization()(x)
        x = Dense(128 , activation="elu")(x)
        x = Dense(4,activation="linear")(x)



        # I = Input(shape=)
        # S = Input(shape=[state_size])
        # A = Input(shape=[action_dim],name='action2')
        # w1 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        # a1 = Dense(HIDDEN2_UNITS, activation='linear')(A)
        # h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
        # h2 = merge([h1,a1],mode='sum')
        # h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        # V = Dense(action_dim, activation='linear')(h3)
        imp = [pos_1,pos_2,vel_1,vel_2,img_1]

        # S = Input(shape=[state_size])   
        # h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        # h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        # Steering = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)  
        # Acceleration = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)   
        # Brake = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1) 
        # V = merge([Steering,Acceleration,Brake],mode='concat')          
        model = Model(inputs=imp+[act],output=x)
        return model, act, imp
