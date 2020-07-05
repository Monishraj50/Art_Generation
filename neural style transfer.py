# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:53:49 2020

@author: Harinath
"""

import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from utils import *
import numpy as np
import tensorflow as tf
import pprint
%matplotlib inline



pp = pprint.PrettyPrinter(indent=4)
model = load_vgg_model("pretrained vgg19/imagenet-vgg-verydeep-19.mat")
pp.pprint(model)

content_image = scipy.misc.imread("images/monish.jpg")
imshow(content_image)

def content_cost(a_G,a_C):
    a_G=tf.convert_to_tensor(a_G)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled =tf.reshape(a_C,shape=[m,-1,n_C])
    a_G_unrolled =tf.reshape(a_G,shape=[m,-1,n_C])
    J_content = 1/(4*n_H*n_W*n_C)*tf.reduce_sum(tf.square(tf.subtract(a_C,a_G)))   
    return J_content

style_image = scipy.misc.imread("images/vector5.jpg")
imshow(style_image)



def gram_matrix(A):
    GM = tf.matmul(A,tf.transpose(A))
    return GM


def layer_style_cost(a_S,a_G):
    a_G=tf.convert_to_tensor(a_G)
    m,n_H,n_W,n_C = a_G.get_shape().as_list()
    a_S_unrolled  = tf.transpose(tf.reshape(a_S,shape=[n_H*n_W,n_C]),perm=[1,0])
    a_G_unrolled  = tf.transpose(tf.reshape(a_G,shape=[n_H*n_W,n_C]),perm=[1,0])
    GS = gram_matrix(a_S_unrolled)
    GG = gram_matrix(a_G_unrolled)
    J_layer_style= 1/((2*n_C*n_H*n_W)**2)*tf.reduce_sum(tf.square(tf.subtract(GS,GG)))
    return J_layer_style

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


def style_cost(STYLE_LAYERS,model):
    J_style=0
    for layer_name,coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style+=layer_style_cost(a_S,a_G)
    return J_style

def total_cost(J_content,J_style,alpha=10,beta=40):
    J=alpha*J_content + beta*J_style
    return J

tf.reset_default_graph()
sess = tf.InteractiveSession()

content_image = scipy.misc.imread("images/monish.jpg")
content_image = reshape_and_normalize_image(content_image)

style_image = scipy.misc.imread("images/vector5.jpg")
style_image = reshape_and_normalize_image(style_image)


generated_image=generate_noise_image(content_image)
imshow(generated_image[0])

model=load_vgg_model("pretrained vgg19/imagenet-vgg-verydeep-19.mat")

sess.run(model['input'].assign(content_image))
out = model['conv5_2']
a_C = sess.run(out)
a_G = out
J_content = content_cost(a_C,a_G)

sess.run(model['input'].assign(style_image))
J_style=style_cost(STYLE_LAYERS,model)


J=total_cost(J_content,J_style)

optimizer=tf.train.AdamOptimizer(0.2)
train_step=optimizer.minimize(J)

def model_nn(sess,input_image,iterations=1000):
    
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_image))
    for i in range(iterations):
        sess.run(train_step)
        generated_image=sess.run(model['input'])
        
        
        if i%50 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)
    save_image('output/generated_image.jpg', generated_image)
    return generated_image


model_nn(sess, generated_image)





    
    