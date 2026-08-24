import pytest
from otovision.triage import normalized_entropy, triage_decision

def test_confident_prediction_not_referred():
    result=triage_decision([0.92,0.02,0.02,0.02,0.02]); assert result["review_required"] is False; assert result["decision"]=="model_assisted_prediction"

def test_uniform_prediction_referred():
    result=triage_decision([0.2,0.2,0.2,0.2,0.2]); assert result["review_required"] is True; assert normalized_entropy([0.2]*5)==pytest.approx(1.0)
