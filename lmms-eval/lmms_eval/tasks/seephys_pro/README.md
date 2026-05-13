# SeePhys Pro

This directory contains the lmms-eval task definitions and judging logic for SeePhys Pro.
No benchmark data or result artifacts are included in the repository.

## Data

Put the parquet files under:

```text
data/seephys_pro/
```

Expected filenames:

- `level1-00000-of-00001.parquet`
- `level2-00000-of-00001.parquet`
- `level3-00000-of-00001.parquet`
- `level4-00000-of-00001.parquet`
- `level1_testmini-00000-of-00001.parquet`
- `level2_testmini-00000-of-00001.parquet`
- `level3_testmini-00000-of-00001.parquet`
- `level4_testmini-00000-of-00001.parquet`

## Task entry points

- `seephys_pro`
- `seephys_pro_all`
- `seephys_pro_total`
- `seephys_pro_testmini`
- `seephys_pro_level1` to `seephys_pro_level4`
- matching `*_testmini` variants

Legacy model ids from `lmms-eval-source` are kept as well, including `openai_compatible`, `local_api_compatible`, and `qwen2_5_vl_interleave`.

## Modes

- `quick_extract: true` uses rule-based extraction.
- `quick_extract: false` uses the LLM judge path.

## Example

```bash
uv run python -m lmms_eval \
  --model openai_compatible \
  --model_args model_version=gpt-5.2,azure_openai=False,continual_mode=True,response_persistent_folder=./cache \
  --tasks seephys_pro_all \
  --batch_size 500 \
  --output_path ./logs \
  --log_samples
```

## Environment variables

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_API_BASE`
- `AZURE_OPENAI_API_VERSION`
- `JUDGE_MODEL_NAME`
- `JUDGE_API_KEY`
- `JUDGE_BASE_URL`
