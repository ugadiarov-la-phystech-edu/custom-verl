#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo-replay-ess-nu1-fresh0.5-fp8-kvfp8-tisc8-tokenmean-orz72k-qwen2.5-7b-2+6
#
# The Qwen2.5-7B 2+6 FP8-KV replay arm (..._dapo17k_2+6_..._fp8_kv-fp8_tis-c8_token-mean_qwen2.5-7b.sh, every
# parameter inherited) trained OPEN-REASONER-ZERO style: ORZ's data, prompt format, stop strings and scorer.
#
# Differences from the DAPO arm (every knob below is an env default; override as usual):
#   * TRAIN_FILE=orz-math-72k.parquet: 47,981 ORZ problems (scripts/convert_orz72k_to_verl_parquet.py:
#     deduplicated, decontaminated against AIME-2024/2025, ground truths <= 40 chars), data_source=math_dapo, the
#     user message is ORZ's inner instruction ("You must put your answer inside <answer> </answer> tags ... \boxed{}
#     ... This is the problem:\n<problem>").
#   * TEST_FILE=aime-2024-orz + aime-2025-orz: the same 30 + 30 problems and answers as the DAPO AIME parquets, x32
#     rows each, with the ORZ instruction; data_source aime2024_orz / aime2025_orz, so the metrics are
#     val-core/aime2024_orz/acc/mean@1 and val-core/aime2025_orz/acc/mean@1 (avg@32, comparable to the DAPO arms'
#     AIME numbers).
#   * CHAT_TEMPLATE_FILE=verl/utils/tokenizer/chat_templates/orz.jinja: ORZ's conversation template (verbatim from
#     Open-Reasoner-Zero-7B's tokenizer_config.json): "A conversation between User and Assistant. ... <think> </think>
#     ... <answer> </answer> ... User: <message>\nAssistant: <think>". Only the template is swapped: Qwen2.5-7B's
#     tokenizer is kept (ORZ-7B's tokenizer has the same vocabulary, special tokens and EOS; the two differ only in
#     chat_template), so no ChatML / "You are a helpful assistant." system prompt is rendered.
#   * ROLLOUT_STOP='["User:","Human:","Assistant:","</answer>"]': ORZ's stop strings (orz_7b_ppo.py generate_vllm),
#     for training and validation sampling. The matched string stays in the response, so the scorer sees the closing
#     </answer>. Without them the base model keeps writing invented User:/Assistant: turns after </answer> until
#     <|endoftext|>: wasted generation, and the LAST <answer> block - possibly an invented one - would be scored.
#     A stop string split across a partial-rollout pause/resume boundary can be missed (rare).
#   * REWARD_FILE=rewards/orz_tag_aware_math.py (compute_score, +1 / -1 like math_dapo): the last <answer> block
#     wins; inside it the last \boxed{} > last "Answer:" line > the block text; then math_dapo string equality ->
#     ORZ's is_equiv -> sympy parse_latex in a forked child with a hard 1 s deadline (ORZ_MATH_SYMPY_TIMEOUT).
#     Also used for validation (a custom reward function ignores data_source).
#   * data_tag=ORZ72K-AIME24ORZ in exp_name (the DAPO arms: DAPO17K-AIME24).
#   * total_rollout_steps=96000 (DAPO arm: 66000): ~2 epochs of the 47,981 prompts. Constant LR, so it only sets the
#     prompt budget and the LR-horizon assertion (lr_decay_steps defaults to it).
#
# Kept: Qwen2.5-7B base, 2+6 layout, B-36, FP8 weights + FP8 KV with VERL_QWEN2_K_BIAS_CENTER=1, TIS C=8 REINFORCE,
# token-mean, replay tau=8 / k=32, min-ESS brake 1.1 / x0.5, nu=1, fresh share 0.5, concurrency 32 / max_num_seqs
# 512, 8,192-token responses (ORZ: 8,000), val T=1.0 / top_p=1.0, test / save every 40, dynamic batch size.
# On remote_h200 pass TRAIN_FILE / TEST_FILE with the /data2/datasets/orz/... paths.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)
export TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/orz/orz-math-72k.parquet"}
export TEST_FILE=${TEST_FILE:-"['/home/jovyan/datasets/math_datasets/orz/aime-2024-orz.parquet','/home/jovyan/datasets/math_datasets/orz/aime-2025-orz.parquet']"}
export CHAT_TEMPLATE_FILE=${CHAT_TEMPLATE_FILE:-"${REPO_ROOT}/verl/utils/tokenizer/chat_templates/orz.jinja"}
export ROLLOUT_STOP=${ROLLOUT_STOP:-'["User:","Human:","Assistant:","</answer>"]'}
export REWARD_FILE=${REWARD_FILE:-"${REPO_ROOT}/rewards/orz_tag_aware_math.py"}
export REWARD_FN=${REWARD_FN:-compute_score}
export data_tag=${data_tag:-"ORZ72K-AIME24ORZ"}
export total_rollout_steps=${total_rollout_steps:-96000}

exec bash "$(dirname "${BASH_SOURCE[0]}")/grpo_novcpo_8gpu_dapo17k_2+6_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.5_fp8_kv-fp8_tis-c8_token-mean_qwen2.5-7b.sh" "$@"
