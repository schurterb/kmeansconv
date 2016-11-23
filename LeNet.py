# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 23:48:26 2016

@author: user

mnist data loader
"""
import theano

theano.config.exception_verbosity='high'

from cnnclassifier import CNN2D
from trainer import Trainer

testnet = CNN2D()

testtrainer = Trainer(testnet, epoch_length=100)

testtrainer.train(5)

