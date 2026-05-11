# MBTI Personality Validation Summary

## Files

- Code: `MBTI/code/mbti_93.py`, `MBTI/code/mbti_eval.py`
- Dataset: `MBTI/data/MBTI_doubled_93.json`
- Final CSV: `MBTI/results/remote_runs/mbti_remote_summary_final.csv`
- Canonical CSV: `MBTI/results/remote_runs/mbti_remote_summary_canonical.csv`
- Remote status CSV: `MBTI/results/remote_runs/mbti_remote_status_final.csv`

## Method

- Decoding: deterministic next-token A/B logit scoring (`decode_method=logit`)
- Trials per model: 1
- MBTI items: 186 total, 184 used, 2 skipped because they crossed dimensions
- Success criterion: for a guided variant, the measured winner on the target dimension must match the guided type

## Base Results

| Model | Type | E/I | S/N | T/F | J/P |
|---|---:|---:|---:|---:|---:|
| Llama-3.2-3B | INFJ | 20/22 | 20/32 | 17/29 | 24/20 |
| Qwen2.5-3B | ENTJ | 23/19 | 20/32 | 28/18 | 26/18 |
| Qwen2.5-7B | ISTJ | 20/22 | 27/25 | 24/22 | 26/18 |

## Adjustment Success

Overall non-Base success: **19/24**.

| Model | Adjusted Type | Target Dimension | Measured Score | Measured Winner | Correct |
|---|---:|---|---:|---:|---:|
| Llama-3.2-3B | E | E/I | 22/20 | E | Yes |
| Llama-3.2-3B | I | E/I | 18/24 | I | Yes |
| Llama-3.2-3B | S | S/N | 25/27 | N | No |
| Llama-3.2-3B | N | S/N | 14/38 | N | Yes |
| Llama-3.2-3B | F | T/F | 8/38 | F | Yes |
| Llama-3.2-3B | T | T/F | 28/18 | T | Yes |
| Llama-3.2-3B | J | J/P | 24/20 | J | Yes |
| Llama-3.2-3B | P | J/P | 22/22 | J/tie | No/tie |
| Qwen2.5-3B | E | E/I | 23/19 | E | Yes |
| Qwen2.5-3B | I | E/I | 21/21 | E/tie | No/tie |
| Qwen2.5-3B | S | S/N | 28/24 | S | Yes |
| Qwen2.5-3B | N | S/N | 15/37 | N | Yes |
| Qwen2.5-3B | F | T/F | 21/25 | F | Yes |
| Qwen2.5-3B | T | T/F | 32/14 | T | Yes |
| Qwen2.5-3B | J | J/P | 29/15 | J | Yes |
| Qwen2.5-3B | P | J/P | 23/21 | J | No |
| Qwen2.5-7B | E | E/I | 23/19 | E | Yes |
| Qwen2.5-7B | I | E/I | 18/24 | I | Yes |
| Qwen2.5-7B | S | S/N | 28/24 | S | Yes |
| Qwen2.5-7B | N | S/N | 20/32 | N | Yes |
| Qwen2.5-7B | F | T/F | 20/26 | F | Yes |
| Qwen2.5-7B | T | T/F | 27/19 | T | Yes |
| Qwen2.5-7B | J | J/P | 29/15 | J | Yes |
| Qwen2.5-7B | P | J/P | 25/19 | J | No |

## Untested Types

All listed personality variants have now been tested.

## Tested Coverage By Server

| Server Tag | Model Variants |
|---|---|
| s10_llama_jp | Llama-3.2-3B Base, Llama-3.2-3B J, Llama-3.2-3B P |
| s12_llama_sn | Llama-3.2-3B Base, Llama-3.2-3B S, Llama-3.2-3B N |
| s1_qwen3b_ft | Qwen2.5-3B Base, Qwen2.5-3B F, Qwen2.5-3B T |
| s2_llama_ei | Llama-3.2-3B Base, Llama-3.2-3B E, Llama-3.2-3B I |
| s3_llama_ft | Llama-3.2-3B Base, Llama-3.2-3B F, Llama-3.2-3B T |
| s4_qwen7b_sn | Qwen2.5-7B Base, Qwen2.5-7B S, Qwen2.5-7B N |
| s5_qwen7b_ei | Qwen2.5-7B Base, Qwen2.5-7B E, Qwen2.5-7B I |
| s6_qwen7b_jp | Qwen2.5-7B Base, Qwen2.5-7B J, Qwen2.5-7B P |
| s7_qwen3b_jp | Qwen2.5-3B Base, Qwen2.5-3B J, Qwen2.5-3B P |
| s8_qwen3b_ei | Qwen2.5-3B Base, Qwen2.5-3B E, Qwen2.5-3B I |
| s9_qwen7b_ft | Qwen2.5-7B Base, Qwen2.5-7B F, Qwen2.5-7B T |
| s11_qwen3b_sn | Qwen2.5-3B Base, Qwen2.5-3B S, Qwen2.5-3B N |

## Notes

- Llama-3.2-3B J/P is now measured: J-guided remains J on J/P=24/20; P-guided is a J/P tie at 22/22 and is counted as not successful under the strict rule.
- Llama-3.2-3B S/N is now measured: S-guided remains N on S/N=25/27 and is counted as not successful; N-guided -> ENFJ with S/N=14/38 and is counted as successful.
- Qwen2.5-3B S/N is now measured: S-guided -> ESTJ with S/N=28/24, N-guided -> ENTJ with S/N=15/37.
- Qwen2.5-7B F/T is measured: F-guided -> INFJ with T/F=20/26, T-guided -> ISTJ with T/F=27/19.
- Main failures are Llama-3.2-3B S-guided, P-guided variants, and the Qwen2.5-3B I-guided tie.
