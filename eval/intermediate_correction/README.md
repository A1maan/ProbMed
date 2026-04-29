# Intermediate-Layer Correction

This experiment applies the LR probe direction at an intermediate hidden layer,
then lets the remaining transformer layers produce the final Yes/No logits.

Run from this directory:

```bash
cd ProbMed/eval/intermediate_correction
```

Model entry points:

```bash
bash llavamed/run_intermediate_correction.sh
bash chexagent/run_intermediate_correction.sh
bash medgemma/run_intermediate_correction.sh
```

Each model also has a curated Python entry point with model-specific defaults:

```bash
python llavamed/intermediate_correction_llavamed.py
python chexagent/intermediate_correction_chexagent.py
python medgemma/intermediate_correction_medgemma.py
```

Defaults:

- LLaVA-Med uses layer 15, matching the earlier peak discussed for that model.
- CheXagent and MedGemma use `best_layer` from their existing
  `analysis_summary.json` files.
- `--mode probe` pushes the last-token representation along the sign predicted
  by the LR probe.
- `--mode oracle` pushes toward the ground-truth sign and is useful as an upper
  bound/sanity check, not as a deployable correction.
- `--strength` controls the hidden-state step size along the normalized LR
  direction.

Outputs are written under:

```text
<model>/results/layer_<layer-or-best>/<mode>_strength_<strength>/
```

Each run saves per-question JSONL records and a summary JSON with baseline and
corrected accuracy, adversarial accuracy, and paired adversarial metrics.
