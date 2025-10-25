from prem_predict import Prem_Predictor
import pandas as pd
import numpy as np


pl = Prem_Predictor()
pl.train_model()

pl.predict_game('Arsenal', 'Crystal Palace')



