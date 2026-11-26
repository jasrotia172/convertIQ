# Verification record

Verified locally on 26 November 2026 (Asia/Kolkata).

- Environment: Darwin arm64, Python 3.14.5; exact packages recorded in `requirements-lock.txt`.
- Dependency resolution: `python -m pip check` returned **No broken requirements found**.
- Static checks: `ruff check .` passed; Python files formatted with Ruff.
- Tests: **43 passed, 0 failures, 0 errors, 0 skipped**. Evidence: `test-results.xml` and `tests.log`.
- Full real-data training: 9 families, 17 original/engineered candidates, a train-fitted screening candidate and calibration comparison. Current version: `20261126T124223Z-b3e5f2`.
- Notebook: all **8 code cells executed**, no error outputs.
- API: localhost `/health` returned HTTP 200 with `model_ready: true`; all specified endpoint contracts, batch consistency, validation rejection, missing-model behavior and explanation additivity exercised with FastAPI TestClient.
- Streamlit: localhost health returned `ok`; Executive Overview visually inspected in Chrome. Every page rendered in AppTest. Manual prediction, empty filters, CSV parsing/scoring/export, labeled monitoring replay and synthetic drift interactions passed automated checks. The CSV test supplies real held-out CSV bytes through the upload function; it does not automate the native file picker.
- Figures: final evaluation panel and SHAP waterfall visually inspected. SHAP export rerun using the noninteractive Agg backend; original categorical labels are shown in the waterfall.
- Monitoring demo: two 600-row historical batches yielded labeled metrics; a 631-row synthetic-drift batch triggered product-duration and exit-rate alerts and had **no performance metrics**. All three records are marked simulated.
- Git repository initialized. No remote was configured and no content was published.

## Execution limits

Dockerfile and Compose configuration are supplied, but **Docker is not installed on this machine**, so container build/runtime verification was not performed. Other Python versions and operating systems were not executed here.

The test suite reports upstream deprecation warnings from Starlette/SHAP and a scikit-learn warning for the intentional single-class monitoring test. They did not fail checks. The model's historical single-split results do not establish future-period, early-session or causal performance.
