# -*- coding: utf-8 -*-
import os

from code.config import train_data_path


def explore_data():

    for file_name in os.listdir(train_data_path):
        with open(os.path.join(train_data_path, file_name)) as f:
            for line in f:
                pass


