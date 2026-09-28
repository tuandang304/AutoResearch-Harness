# New study template

Copy this folder to projects/<study-name>. Complete brief.md, create reviewed
ideas.json, and optionally copy configs/default.yaml to config.yaml.
Validate with `python -m autoresearch --project projects/<study-name> --dry-run`.
The pipeline creates runs/ automatically. data/ and artifacts/ are optional,
ignored local storage and are not automatically consumed by the pipeline.
