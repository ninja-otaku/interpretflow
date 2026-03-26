"""Core InterpretableModel wrapper -- Phase 1 scaffold."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]


@dataclass
class ModelCard:
    training_data_description: str = ""
    evaluation_metrics: dict[str, float] = field(default_factory=dict)
    model_name: str = ""
    version: str = ""

@dataclass
class ComplianceMapper:
    def check_compliance(self, model_card: ModelCard, regulation: str) -> bool:
        if regulation == "SR_11_7":
            return bool(
                model_card.training_data_description
                and model_card.evaluation_metrics
                and model_card.model_name
            )
        elif regulation == "EU_AI_ACT_13":
            return bool(
                model_card.training_data_description
                and model_card.evaluation_metrics
                and model_card.model_name
                and model_card.version
            )
        elif regulation == "FINRA":
            return bool(
                model_card.training_data_description
                and model_card.evaluation_metrics
                and model_card.model_name
                and model_card.evaluation_metrics.get("accuracy", 0) >= 0.7
            )
        else:
            raise ValueError(f"Unknown regulation: {regulation}")

@dataclass
class ModelCardGenerator:
    template: str = """
    <html>
    <head><title>Model Card</title></head>
    <body>
    <h1>{{ model_name }}</h1>
    <p><strong>Training Data Description:</strong> {{ training_data_description }}</p>
    <p><strong>Evaluation Metrics:</strong></p>
    <ul>
        {% for metric, value in evaluation_metrics.items() %}
        <li>{{ metric }}: {{ value }}</li>
        {% endfor %}
    </ul>
    </body>
    </html>
    """

    def generate(self, model_card: ModelCard) -> str:
        from jinja2 import Template
        template = Template(self.template)
        return template.render(model_name=model_card.model_name,
                              training_data_description=model_card.training_data_description,
                              evaluation_metrics=model_card.evaluation_metrics)

@dataclass
class CounterfactualEngine:
    model: Any

    def find_counterfactual(
        self, input_data: np.ndarray, target_prediction: int, constraints: dict[int, float]
    ) -> np.ndarray:
        def objective(delta: np.ndarray) -> float:
            return float(np.sum(np.abs(delta)))

        def constraint(delta: np.ndarray) -> float:
            return float(self.model.predict(input_data + delta) - target_prediction)

        cons = {'type': 'eq', 'fun': constraint}
        bounds = [
            (None, None) if i not in constraints else (constraints[i], constraints[i])
            for i in range(input_data.shape[0])
        ]
        
        result = minimize(objective, np.zeros_like(input_data), method='SLSQP', bounds=bounds, constraints=cons)
        return np.asarray(result.x)
