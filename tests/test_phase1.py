"""Phase 1 tests -- built by aider."""
import pytest
from src.interpretflow.core import InterpretableModel, xgb, shap, ModelCard, ComplianceReport, CounterfactualEngine, CounterfactualExplanation

@pytest.fixture
def model():
    base_model = xgb.XGBClassifier()
    return InterpretableModel(base_model)

@pytest.fixture
def X_y():
    # Placeholder data for testing
    X = [[1, 2], [3, 4]]
    y = [0, 1]
    return X, y

def test_fit_predict(model, X_y):
    X, y = X_y
    model.fit(X, y)
    predictions, shap_attribution = model.predict(X, explain=True)
    assert len(predictions) == len(y)
    assert isinstance(shap_attribution, dict)

def test_generate_model_card(model):
    model_card = model.generate_model_card()
    assert isinstance(model_card, ModelCard)
    assert model_card.model_name == "InterpretableModel"

def test_compliance_report(model):
    compliance_report = model.compliance_report('SR_11_7')
    assert isinstance(compliance_report, ComplianceReport)
    assert compliance_report.standard == 'SR_11_7'

def test_counterfactual_engine(model, X_y):
    X, y = X_y
    model.fit(X, y)
    counterfactual_engine = CounterfactualEngine(model)
    explanation = counterfactual_engine.find_counterfactual(X[0], 1 - y[0])
    assert isinstance(explanation, CounterfactualExplanation)
