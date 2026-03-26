import numpy as np
from src.interpretflow.core import CounterfactualEngine, ModelCard, ComplianceMapper, ModelCardGenerator

def test_counterfactual_engine():
    # Mock model for testing
    class MockModel:
        def predict(self, x):
            return np.dot(x, [1, -1])

    engine = CounterfactualEngine(MockModel())
    input_data = np.array([1.0, 2.0])
    target_prediction = -1
    constraints = {0: 1.0}  # Feature 0 cannot change

    delta = engine.find_counterfactual(input_data, target_prediction, constraints)
    assert np.allclose(delta, [-3.0, 0.0]), f"Expected delta to be [-3.0, 0.0], but got {delta}"

def test_compliance_mapper():
    mapper = ComplianceMapper()
    model_card = ModelCard(training_data_description="Description", evaluation_metrics={"accuracy": 0.8}, model_name="ModelA", version="1.0")

    assert mapper.check_compliance(model_card, "SR_11_7")
    assert mapper.check_compliance(model_card, "EU_AI_ACT_13")
    assert mapper.check_compliance(model_card, "FINRA")

def test_model_card_generator():
    generator = ModelCardGenerator()
    model_card = ModelCard(training_data_description="Description", evaluation_metrics={"accuracy": 0.8}, model_name="ModelA", version="1.0")

    html = generator.generate(model_card)
    assert "<h1>ModelA</h1>" in html
    assert "<strong>Training Data Description:</strong> Description" in html
    assert "<li>accuracy: 0.8</li>" in html
