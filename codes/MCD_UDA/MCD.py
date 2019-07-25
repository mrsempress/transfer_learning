# encoding=utf-8
"""
    Created on 20:23 2019/07/24
    @author: Chenxi Huang
    MCD is adversarial taht need a generator and two classifiers.
"""
import Network


def Generator(source, target, pixelda=False):
    if source == 'usps' or target == 'usps':
        return Network.uFeature()
    elif source == 'svhn':
        return Network.s2mFeature()
    elif source == 'synth':
        return Network.s2gFeature()


def Classifier(source, target):
    if source == 'usps' or target == 'usps':
        return Network.uPredictor()
    if source == 'svhn':
        return Network.s2mPredictor()
    if source == 'synth':
        return Network.s2gPredictor()
