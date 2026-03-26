# InterpretFlow

[![CI](https://github.com/ninja-otaku/interpretflow/actions/workflows/ci.yml/badge.svg)](https://github.com/ninja-otaku/interpretflow/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)

**Interpretability-first ML pipeline wrapper for regulated industries.**

Train any scikit-learn / XGBoost model and get SHAP explanations,
counterfactuals, an audit log, and a compliance report
(SR 11-7, EU AI Act Art.13, FINRA) -- automatically.

```python
from interpretflow import InterpretableModel
model = InterpretableModel(base_model="xgboost")
model.fit(X_train, y_train)
report = model.compliance_report(standard="SR_11_7")
```

> **Status:** Phase 1 in active development.

## Install

```bash
pip install interpretflow
pip install "interpretflow[full]"  # + counterfactuals, PDF reports
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs welcome!

## License

Apache 2.0 -- see [LICENSE](LICENSE).
