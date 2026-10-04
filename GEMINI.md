\# mvtec-defect-mlops



Goal: anomaly-detection service for industrial defects (MVTec AD; categories: bottle, screw, capsule) with reproducible training, experiment tracking, later serving and monitoring.



Stack: Python 3.11, uv, PyTorch, Anomalib (Patchcore), MLflow (sqlite backend), DVC, pytest, ruff, GitHub Actions.



Layout: src/mvtec\_defect/, tests/, params.yaml, data/ (gitignored), artifacts/ and metrics/ (DVC-managed).



Rules:

\- small commits

\- run ruff and pytest before finishing a task

\- never invent metric values

\- never commit data, mlruns, mlflow.db or weights

\- show command output

\- ask before adding dependencies

\- log non-obvious decisions in docs/decisions.md

