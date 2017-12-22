from __future__ import division
import numpy as np
import tensorflow as tf

# rotate around z axis
so3_a = np.array([
    [0, -1, 0, 1, 0, 0, 0, 0, 0],
    [1, 0 , 0, 0, 1, 0, 0, 0, 0],
    [0, 0 , 0, 0, 0, 0, 0, 0, 1]
])

# rotate around y axis
so3_b = np.array([
    [0, 0, 1, 0, 0, 0, -1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0 , 0, 1],
    [0, 0, 0, 0, 1, 0, 0 , 0, 0]
])

# rotate around z axis
so3_y = np.array([
    [0, 0, 0, 0, 0, -1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0 , 0, 0, 1],
    [1, 0, 0, 0, 0, 0 , 0, 0, 0]
])

so3_a = tf.constant(so3_a, tf.float32)
so3_b = tf.constant(so3_b, tf.float32)
so3_y = tf.constant(so3_y, tf.float32)


# z : B x 1  (rotation angle about the axis z)
# y : B x 1  (rotation angle about the axis y)
# x : B x 1  (rotation angle about the axis x)

def so3_mat(z, y, x):
    B = tf.shape(z)[0]
    N = 1

    z = tf.clip_by_value(z, -np.pi, np.pi)                       # B x 1
    y = tf.clip_by_value(y, -np.pi, np.pi)                       # B x 1
    x = tf.clip_by_value(x, -np.pi, np.pi)                       # B x 1

    cosz = tf.cos(z)
    sinz = tf.sin(z)
    t_z  = tf.concat([sinz, cosz, tf.ones_like(sinz)], -1)       # B x 3
    soz  = tf.matmul(t_z, so3_a)
    soz  = tf.reshape(soz, (-1, 3, 3))

    cosy = tf.cos(y)
    siny = tf.sin(y)
    t_y  = tf.concat([siny, cosy, tf.ones_like(siny)], -1)
    soy  = tf.matmul(t_y, so3_b)
    soy  = tf.reshape(soy, (-1, 3, 3))

    cosx = tf.cos(x)
    sinx = tf.sin(x)
    t_x  = tf.concat([sinx, cosx, tf.ones_like(sinx)], -1)
    sox  = tf.matmul(t_x, so3_y)
    sox  = tf.reshape(sox, (-1, 3, 3))
    
    so3 = tf.matmul(tf.matmul(sox, soy), soz)
    return so3
# so3 : B x 3 x 3


