import numpy as np

from src.interpretflow.core import (
    ComplianceMapper,
    CounterfactualEngine,
    ModelCard,
    ModelCardGenerator,
)


def test_counterfactual_engine():
    """Verify find_counterfactual shifts the model output to the target."""

    class MockModel:
        def predict(self, x: np.ndarray) -> float:
            # f(x) = x[0] - x[1]; predict([2,0]) = 2 (far from target 0)
            return float(np.dot(x, [1.0, -1.0]))

    engine = CounterfactualEngine(MockModel())
    input_data = np.array([2.0, 0.0])
    target_prediction = 0.0
    constraints: dict[int, float] = {}  # no fixed features

    delta = engine.find_counterfactual(input_data, target_prediction, constraints)
    new_pred = MockModel().predict(input_data + delta)

    assert isinstance(delta, np.ndarray), "delta must be ndarray"
    assert abs(new_pred - target_prediction) < 1e-3, (
        f"Expected prediction {target_prediction}, got {new_pred}"
    )


def test_compliance_mapper():
    mapper = ComplianceMapper()
    model_card = ModelCard(
        training_data_description="Description",
        evaluation_metrics={"accuracy": 0.8},
        model_name="ModelA",
        version="1.0",
    )
    assert mapper.check_compliance(model_card, "SR_11_7")
    assert mapper.check_compliance(model_card, "EU_AI_ACT_13")
    assert mapper.check_compliance(model_card, "FINRA")


def test_model_card_generator():
    generator = ModelCardGenerator()
    model_card = ModelCard(
        training_data_description="Description",
        evaluation_metrics={"accuracy": 0.8},
        model_name="ModelA",
        version="1.0",
    )
    html = generator.generate(model_card)
    assert "<h1>ModelA</h1>" in html
    assert "<strong>Training Data Description:</strong> Description" in html
    assert "<li>accuracy: 0.8</li>" in html
