# Research projects

Use one directory per study under projects/. Shared infrastructure is independent
of any particular research topic.

## Inputs

brief.md describes hypotheses, datasets, access permissions, baselines, budgets,
metrics and limitations. It is a planning document; the launcher reads ideas.json.
Use projects/regularization/ideas.json for the schema. Put essential constraints
in the Abstract because some stages do not consume every descriptive field.

An optional config.yaml fully replaces shared defaults; it is not merged.
Keep exp_name: run. Placing data in project/data does not automatically upload it;
specify and verify access in experiment code.

```bash
cp -r projects/_template projects/my-study
cp configs/default.yaml projects/my-study/config.yaml
cp projects/regularization/ideas.json projects/my-study/ideas.json
# Edit the hypothesis, data, evaluation and budgets before execution.
python -m autoresearch --project projects/my-study --dry-run
```

The existing ideation tool accepts --workshop-file projects/my-study/brief.md.
Inspect generated candidates and save selected inputs as ideas.json. Generation
does not establish novelty or verify literature.

## Outputs

The launcher creates project runs/ automatically. Input snapshots, raw results,
code, status, usage, figures and manuscripts stay together. Use artifacts/ for
curated local exports and explicitly select files for publication. README-only
sample folders describe planned studies, not completed experiments.

## Migration

The shared bfts_config.yaml moved to configs/default.yaml. Generic examples moved
to projects/regularization/. Update scripts referencing their old paths. The
root launcher and engine import paths remain supported.

Historical results moved intact from experiments/ to
projects/uav-lowlight/archive/experiments/. No experiment data was deleted or
rewritten. Embedded absolute paths may refer to old locations: resolve them
manually when inspecting archived artifacts. This is not transparent resume.

Old public UAV validation guidance and input were removed. A small execution
configuration remains in tests/fixtures/pipeline.yaml solely for regression tests.
New studies use the normal shared defaults or their own full configuration.
