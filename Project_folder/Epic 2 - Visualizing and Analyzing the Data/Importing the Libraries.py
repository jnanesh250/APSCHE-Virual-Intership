"""
Common imports for data analysis, visualization, and ML used in Epic 2
"""

import os
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# optional: imbalanced-learn utilities (uncomment if installed)
# from imblearn.combine import SMOTETomek

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# utilities for saving/loading models
import joblib

# configure basic logging
logging.basicConfig(level=logging.INFO)
