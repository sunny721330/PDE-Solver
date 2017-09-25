import time
import math
import tensorflow as tf
import numpy as np
from tensorflow.python.training.moving_averages import assign_moving_average
from tensorflow import random_normal_initializer as norm_init
from tensorflow import random_uniform_initializer as unif_init
from tensorflow import constant_initializer as const_init
from cfg_params import default_param
from tensorflow.python.ops import control_flow_ops
from math import exp, sqrt
from argparser import ArgParser

class PDESolver( object ):
    def __init__ ( self , sess, equation = 'Allen-Cahn' ):
        self.sess = sess
        self.equation = equation
        # parameters for the PDE
        params = default_param[equation]
        self.d = params['d']
        self.T = params['T']
        # parameters for the algorithm
        self.n_time = params['n_time']
        self.n_layer = params['n_layer']
        self.n_neuron = [ self.d , self.d +10 , self.d +10 , self.d ]
        self.batch_size = params['batch_size']
        self.valid_size = params['valid_size']
        self.n_maxstep = params['n_maxstep']
        self.n_displaystep = params['n_displaystep']
        self.learning_rate = params['learning_rate']
        if equation == 'Burger':
            self.learning_rate = tf.train.exponential_decay(0.1,self.n_maxstep+1,20000,0.1,staircase=True)
            self.alpha = self.d ** 2
            self.kappa = 1 / self.d
        self.Yini = params['Yini']
        self.muBar = params['muBar']
        self.sigmaBar = params['sigmaBar']
        # some basic constants and variables
        self.h = ( self.T +0.0)/ self.n_time
        self.sqrth = math.sqrt ( self.h )
        self.t_stamp = np.arange (0 , self.n_time )* self.h
        self._extra_train_ops = params['_extra_train_ops']
        self.initPoint = np.zeros(self.d)
        if equation == 'Pricing':
            self.rl = params['rl']
            self.rb = params['rb']
            self.initPoint = np.ones(self.d) * 100


    def train( self ):
        start_time = time.time()
        # train operations
        self.global_step = tf.get_variable( 'global_step', [], initializer=const_init(1), trainable=False, dtype=tf.int32 )
        trainable_vars = tf.trainable_variables()
        grads = tf.gradients( self.loss, trainable_vars )
        optimizer = tf.train.AdamOptimizer( self.learning_rate )
        apply_op = optimizer.apply_gradients( zip(grads, trainable_vars), global_step=self.global_step )
        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group( * train_ops )
        self.loss_history = []
        self.init_history = []
        dW_valid, X_valid = self.sample_path( self.valid_size )
        feed_dict_valid = { self.dW          : dW_valid,
                            self.X           : X_valid,
                            self.is_training : False}

        step = 1
        sess = self.sess
        sess.run(tf.global_variables_initializer())
        temp_loss = sess.run( self.loss, feed_dict = feed_dict_valid )
        temp_init = self.Y0.eval()[0]
        self.loss_history.append(temp_loss)
        self.init_history.append(temp_init)
        if self.equation == 'Burger':
            print( "u(0,x) = %f" % (self.u_fn(0, tf.constant(value=np.zeros([self.d])))) )
        print(" step : %5u , loss : %.4e , " % (0, temp_loss) + " Y0 : %.4e , runtime : %4u " % \
              (temp_init, time.time() - start_time + self.t_bd))

        for i in range(self.n_maxstep+1):
            step = sess.run( self.global_step )
            dW_train, X_train = self.sample_path( self.batch_size )
            sess.run( self.train_op, feed_dict={ self.dW : dW_train,
                                                 self.X  : X_train,
                                             self.is_training : True} )

            if step % self.n_displaystep == 0:
                temp_loss = sess.run(self.loss, feed_dict=feed_dict_valid)
                temp_init = self.Y0.eval()[0]
                self.loss_history.append( temp_loss )
                self.init_history.append( temp_init )
                print(" step : %5u , loss : %.4e , " % \
                  (step, temp_loss) + \
                  " Y0 : %.4e , runtime : %4u " % \
                  (temp_init, time.time() - start_time + self.t_bd))
            step += 1

        end_time = time.time()
        print(" running time : %.3fs " % \
              (end_time - start_time + self.t_bd ))


    def build( self ):
        start_time = time.time()
        # build the whole network by stacking subnetworks
        self.dW = tf.placeholder(tf.float64, [None, self.d, self.n_time], name='dW')
        self.X = tf.placeholder(tf.float64, [None, self.d, self.n_time+1], name='X')
        self.is_training = tf.placeholder( tf.bool )


        #self.Y0 = tf.Variable( np.random.uniform(low=self.Yini[0], high=self.Yini[1], size=[1]), dtype=tf.float64 )
        self.Y0 = tf.Variable( tf.random_uniform ([1] , minval = self.Yini[0], maxval = self.Yini[1], dtype = tf.float64 ));
        self.Z0 = tf.Variable( np.random.uniform(-.1, .1, size=[1, self.d]), dtype=tf.float64 )
        self.allones = tf.ones( shape = tf.stack([ tf.shape( self.dW )[0],1]) ,dtype = tf.float64 )
        Y = self.allones * self.Y0
        Z = tf.matmul( self.allones, self.Z0 )

        with tf.variable_scope( 'forward' ):
            for t in range(self.n_time):
                Y = Y - self.f_fn( self.t_stamp[ t ], self.X[: , : , t ], Y, Z ) * self.h \
                    + tf.reduce_sum( Z * self.dW[: , : , t ], 1, keep_dims = True )
                if t < self.n_time - 1:
                    Z = self._one_time_net( self.X[: , : , t+1], str( t +1))/ self.d

            delta = Y - self.g_fn( self.T, self.X[: , : , self.n_time ])
            self.clipped_delta = tf.clip_by_value( delta, -5000.0, 5000.0)
            self.loss = tf.reduce_mean( self.clipped_delta ** 2 )

        self.t_bd = time.time() - start_time

    def sample_path ( self, n_sample ):
        dW = np.random.normal(size=[ n_sample, self.d, self.n_time ]) * self.sqrth
        X = np.zeros([ n_sample, self.d, self.n_time+1 ])
        X[:, :, 0] = np.tile(self.initPoint, (n_sample, 1))
        #dW = np.random.normal(size = )
        for i in range( self.n_time ):
            if self.equation in ['Allen-Cahn', 'HJB']:
                X[:, :, i+1] = X[:, :, i] + sqrt(2) * dW[:, :, i]
            elif self.equation == 'Pricing':
                X[:,:,i+1] = exp((self.muBar- 0.5 * (self.sigmaBar ** 2) )*self.h) * \
                                    np.multiply(np.exp(self.sigmaBar * dW[:,:,i]),X[:,:,i])
            elif self.equation == 'Burger':
                X[:,:,i+1] = X[:,:,i] + self.d / sqrt(2) * dW[:,:,i]
        return dW, X

    def f_fn( self, t, X, Y, Z ):
        if self.equation == 'Allen-Cahn':
            ans = Y - tf.pow(Y, 3)
        elif self.equation == 'HJB':
            ans = -tf.reduce_sum(Z ** 2, 1, keep_dims=True)
        elif self.equation == 'Pricing':
            ans = -self.rl * Y - (self.muBar - self.rl) / self.sigmaBar * tf.reduce_sum(Z,1,keep_dims=True) +\
                  (self.rb - self.rl) * tf.maximum(Y * 0, 1 / self.sigmaBar * tf.reduce_sum(Z,1,keep_dims=True) - Y)
        elif self.equation == 'Burger':
            ans =  tf.multiply( Y - (2 + self.d) / (2 * self.d), tf.reduce_sum(Z,1,keep_dims=True) )
        return ans

    def g_fn( self , t , X ):
        # terminal conditions
        if self.equation == 'Allen-Cahn':
            ans = 0.5/(1 + 0.2 * tf.reduce_sum( X ** 2 , 1 , keep_dims = True ))
        elif self.equation == 'HJB':
            ans = tf.log(0.5 * (1 + tf.reduce_sum( X ** 2, 1, keep_dims=True )) )
        elif self.equation == 'Pricing':
            ans = tf.maximum(tf.reduce_max(X, 1, keep_dims=True)-120,0 * tf.reduce_max(X, 1, keep_dims=True)) - \
                  2 * tf.maximum(tf.reduce_max(X, 1, keep_dims=True)-150,0 * tf.reduce_max(X, 1, keep_dims=True))
        elif self.equation == 'Burger':
            ans = 1 - tf.reciprocal(1 + tf.exp(self.T + 1 / self.d * tf.reduce_sum(X,1,keep_dims=True)))
        return ans

    def u_fn(self, t, X):
        if self.equation == 'Burger':
            return (1 - 1 / (1 + tf.exp(t + self.kappa * tf.reduce_sum(X,keep_dims=True)))).eval()[0]


    def _one_time_net( self , x , name ):
        with tf.variable_scope( name ):
            x_norm = self._batch_norm(x, name = 'layer0_normal' )
            layer1 = self._one_layer( x_norm, self.n_neuron[1], name = 'layer1' )
            layer2 = self._one_layer( x_norm, self.n_neuron[2], name = 'layer2' )
            z = self._one_layer( layer2, self.n_neuron[3], activation_fn=tf.identity, name = 'final' )
        return z

    def _one_layer( self , input_, out_sz, activation_fn = tf.nn.relu, std =5.0 , name = 'linear' ):
        with tf.variable_scope( name ):
            shape = input_.get_shape().as_list()
            w = tf.get_variable( 'Matrix', [shape[1], out_sz], tf.float64,
                                 norm_init(stddev= std / np.sqrt(shape[1]+out_sz)) )
            hidden = tf.matmul( input_, w )
            hidden_bn = self._batch_norm( hidden, name = 'normal' )
        return activation_fn( hidden_bn )

    def _batch_norm( self, x, name ):
        with tf.variable_scope( name ):
            params_shape = [x.get_shape()[-1]]
            beta = tf.get_variable( 'beta', params_shape, tf.float64, norm_init(0.0 , stddev =0.1, dtype = tf.float64) )
            gamma = tf.get_variable( 'gamma', params_shape, tf.float64, unif_init(.1, .5, dtype=tf.float64) )
            mv_mean = tf.get_variable( 'moving_mean', params_shape, tf.float64, const_init(0, tf.float64), trainable=False )
            mv_var = tf.get_variable( 'moving_variance', params_shape, tf.float64, const_init(1, tf.float64), trainable=False )

            mean, variance = tf.nn.moments(x, [0])
            self._extra_train_ops.append( assign_moving_average(mv_mean, mean, .99) )
            self._extra_train_ops.append( assign_moving_average(mv_var, variance, .99) )
            mean, variance = control_flow_ops.cond(self.is_training, lambda :(mean, variance),\
                                                   lambda : (mv_mean,mv_var))
            y = tf.nn.batch_normalization( x, mean, variance, beta, gamma, 1e-6 )
            y.set_shape(x.get_shape())
            return y


def main():
    parser = ArgParser()
    args = parser.parse_args()
    tf.reset_default_graph()
    equation = args.equation
    with tf.Session() as sess:
        tf.set_random_seed(1)
        print("Begin to solve %s equation" % (equation))
        model = PDESolver( sess, equation=equation )
        model.build()
        model.train()
        writer = tf.summary.FileWriter("/tmp/log/", sess.graph)
        output = np.zeros(( len( model.init_history ) , 3))
        output [: , 0] = np.arange( len (model.init_history ) ) * model.n_displaystep
        output [: , 1] = model.loss_history
        output [: , 2] = model.init_history
        np.savetxt (  "./%s.csv" %(equation), output, fmt =[ '%d' , '%.5e' ,  '%.5e'] ,
        delimiter =" ," , header =" step , loss function , " + " target value , runtime " , comments = ' ')

if __name__ == '__main__':
    np.random.seed(1)
    main()
