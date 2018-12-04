# -*- coding: utf-8 -*-


""" PATH """
import os

ROOT_PATH = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))
Data_PATH = os.path.join(ROOT_PATH, "Datasets")
train_data_path = os.path.join(Data_PATH, "ruijin_round2_train/ruijin_round2_train")
test_data_path = os.path.join(Data_PATH, "ruijin_round2_test_a/ruijin_round2_test_a")
test_b_data_path = os.path.join(Data_PATH, "ruijin_round2_test_b/ruijin_round2_test_b")


