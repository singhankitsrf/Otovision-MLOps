import numpy as np
from otovision.metrics import expected_calibration_error, multiclass_brier_score

def test_perfect_probabilities_have_zero_brier():
    y=np.array([0,1,2]); probs=np.eye(3); assert multiclass_brier_score(y,probs,3)==0.0

def test_ece_is_bounded():
    y=np.array([0,1,1,0]); probs=np.array([[0.9,0.1],[0.2,0.8],[0.6,0.4],[0.7,0.3]]); ece=expected_calibration_error(y,probs); assert 0.0<=ece<=1.0
