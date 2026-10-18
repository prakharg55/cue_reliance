# Logs

One log per trained model, exactly as printed by the training scripts. The analysis
scripts read the summary lines, such as `RESULT`, which holds the finished model's
worst-group accuracy and cue reliance.

File names give the run group, the model, ρ and the seed, followed by the cluster job
ID. For example, `civ2_qwen_r0.97_s0_11489602.log` is Qwen2.5-0.5B on CivilComments
at ρ = 0.97 with seed 0.

| Prefix | Runs | Models | Script in `training/` |
|---|---|---|---|
| `civ2_`, `biosv2_` | Qwen2.5-0.5B and 1.5B, Pythia-410m and 1.4b | 144 | `train_civil.py`, `train_bios.py` |
| `civ3_`, `bios3_` | Qwen2.5-0.5B, Pythia-410m and OLMo-2-1B with all repairs | 108 | `train_civil_repairs.py`, `train_bios_repairs.py` |
| `hs_civ2_`, `hs_biosv2_` | High-seed arm, 12 more seeds at ρ = 0.65, 0.85 and 0.97 | 108 | `train_civil.py`, `train_bios.py` |
| `ew_civ_` | Cue reliance measured after every epoch | 72 | `train_civil_epochs.py` |
| `rob_full_`, `rob_it_` | Full fine-tuning and an instruction-tuned base | 36 | `train_civil_robustness.py` |
| `pbo_` | Placebo edit | 18 | `train_civil_placebo.py` |
| `pubredo` | Eight public toxicity classifiers | 4 files | `eval_public.py`, `eval_public_offline.py` |

The ten populations of the main analysis are the eight `civ2_` and `biosv2_` cells plus
the two OLMo-2 cells from `civ3_` and `bios3_`.
