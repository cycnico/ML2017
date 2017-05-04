#!/usr/bin/env python
# -- coding: utf-8 --
import os
from termcolor import colored, cprint
import argparse
from keras.utils.visualize_util import plot
from keras.models import load_model

def main():
    emotion_classifier = load_model(model_path)
    emotion_classifier.summary()
    plot(emotion_classifier,to_file='model.png')
