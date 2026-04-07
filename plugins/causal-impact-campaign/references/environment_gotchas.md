# Reference: Environment & Dependency Gotchas

## numpy version conflict (tfcausalimpact vs CausalPy)

These two packages **cannot coexist** in the same Python environment:

| Package | numpy | pandas | Notes |
|---|---|---|---|
| `tfcausalimpact` | < 2.0 | <= 2.2 | TensorFlow 2.16 needs numpy 1.x |
| `CausalPy` | >= 2.0 | >= 3.0 | PyMC/PyTensor needs numpy 2.x |

**Workflow:** Run tfcausalimpact first (numpy<2), then `pip install "numpy>=2"`, then run
CausalPy in a separate script. Never import both in the same process.

## CausalPy on macOS

Requires `cores=1` in `sample_kwargs` — the default multiprocessing fork causes
`RuntimeError: An attempt has been made to start a new process before the current
process has finished its bootstrapping phase`. Fix:

```python
model=cp.pymc_models.LinearRegression(
    sample_kwargs={"random_seed": 42, "chains": 4, "draws": 2000, "tune": 1000, "cores": 1}
)
```

## CausalPy short treatment windows (< 7 days)

CausalPy may fail with `Dimension(s) 'draw', 'chain' do not exist` xarray error on
very short treatment windows (< 7 days). The ITS design requires enough post-period
data for the Bayesian regression to produce valid posterior predictions.

**Fix:** Skip CausalPy for short campaigns and note the limitation. Use RDiT as the
lead method for significance claims on short campaigns.

## Weather data: Open-Meteo API

Best free source for daily weather covariates. No API key needed.

```bash
curl -sk "https://archive-api.open-meteo.com/v1/archive?latitude=51.5074&longitude=-0.1278&start_date=2024-10-01&end_date=2026-03-15&daily=temperature_2m_mean,precipitation_sum&timezone=Europe/London"
```

For recent days not yet in the archive, backfill from the forecast API:
`https://api.open-meteo.com/v1/forecast` (same parameters).

**SSL note:** Corporate proxies may block the API. Use `curl -sk` to bypass.

## Python version mismatch

`pip3 install` may install to a different Python version's site-packages. Always use:
```bash
python3 -m pip install <package>
```
Verify with `python3 -m pip show <package> | grep Location`.
