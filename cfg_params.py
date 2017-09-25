default_param = {}
default_param['Allen-Cahn'] = { 'd': 100, 'T' : 0.3, 'n_time' : 20,
                                'n_layer' : 4, 'batch_size' : 64, 'valid_size' : 256, 'n_maxstep' : 4000,
                                'n_displaystep' : 100, 'learning_rate' : 5e-4, 'Yini' : [0.3,0.6],
                                '_extra_train_ops' : [], 'muBar' : 0,  'sigmaBar' : 1 }

default_param['HJB'] = {'d' : 100, 'T' : 1, 'n_time' : 20,
                        'n_layer' : 4, 'batch_size' : 64, 'valid_size' : 256, 'n_maxstep' : 2000,
                        'n_displaystep' : 100, 'learning_rate' : 0.01, 'Yini' : [0.3,0.6],
                        '_extra_train_ops' : [], 'muBar' : 0,  'sigmaBar' : 1}

default_param['Pricing'] = {'d' : 100, 'T' : 1/2, 'n_time' : 20,
                            'n_layer' : 4, 'batch_size' : 64, 'valid_size' : 256, 'n_maxstep' : 10000,
                            'n_displaystep' : 100, 'learning_rate' : 0.005, 'Yini' : [10,100],
                            '_extra_train_ops' : [], 'muBar' : 6e-2,  'sigmaBar' : 0.2,
                            'rl' : 0.04, 'rb' : 0.06}

default_param['Burger'] = {'d' : 20, 'T' : 1, 'n_time' : 80,
                            'n_layer' : 4, 'batch_size' : 64, 'valid_size' : 256, 'n_maxstep' : 60000,
                            'n_displaystep' : 100, 'learning_rate' : 0.005, 'Yini' : [0,1],
                            '_extra_train_ops' : [], 'muBar' : 0,  'sigmaBar' : 0,
                            }
