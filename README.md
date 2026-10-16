# Market Risk Modeling for a Bermudan Callable CMS Note

This repository contains the market-risk workstream for a callable inverse-CMS note. Pricing is owned by a separate team. The code here treats the pricing model as a callable function and uses it for full revaluation under historical, Monte Carlo, and stress scenarios.

## Scope

- Full-revaluation historical simulation
- PCA Gaussian Monte Carlo VaR
- PCA Student-t Monte Carlo VaR
- 95% and 99% VaR
- 97.5% and 99% expected shortfall
- Maximum simulated loss
- Kupiec and Christoffersen VaR backtests
- ES diagnostics
- Standard stress scenarios
- Rate-volatility scenario matrix
- Estimation-window and runtime comparisons

## Important modeling distinction

The Hull-White model used by the pricing team is a risk-neutral pricing model. VaR scenarios must represent real-world market changes. This repository therefore estimates scenario distributions from historical market data and then sends the shocked market states to the pricing model.

## Expected market-data format

Use a wide CSV with one row per date and one column per risk factor:

```text
Date,OIS_1Y,OIS_2Y,OIS_5Y,OIS_10Y,OIS_30Y,VOL_1Y2Y,VOL_2Y2Y,VOL_5Y2Y,VOL_10Y2Y,ISSUER_SPREAD
2024-01-02,0.0481,0.0440,0.0400,0.0390,0.0410,0.0067,0.0069,0.0072,0.0075,0.0120
```

Rates, normal volatilities, and spreads use decimal units. For example, 100 bp is `0.01`.

## Pricing-team interface

The risk engine expects:

```python
def price_callable_cms(market_state: pandas.Series) -> float:
    """Return the full clean value for one unit of notional."""
```

The pricing adapter must consume the shocked OIS curve, CMS projection inputs, swaption volatility surface, issuer spread when included, and all contractual call rules. It must allow the optimal call date to change under each scenario.

Replace `examples/mock_pricer.py` with an adapter to the production pricer. The included mock pricer is only for integration testing and must not be used for final risk numbers.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`.

## Run the demonstration

```bash
python examples/run_demo.py
```

The demonstration uses synthetic data and writes CSV output to `results/`.

## Tests

```bash
pytest -q
```

## Recommended empirical design

Use a one-day holding period. Compare 250-, 500-, and 1,000-day estimation windows. Use at least 10,000 Monte Carlo scenarios and preferably 50,000 for final results. Preserve joint historical shocks across curve, volatility, and spread factors. Report model runtime, VaR exceptions, tail-loss behavior, window sensitivity, and stress losses alongside VaR and ES.

## Project status

The framework is operational with synthetic data. Final results require the historical risk-factor file and the pricing team's callable revaluation function.
