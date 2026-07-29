# Fine-tuning a local model for locode

## Fine-tuning vs distillation — the actual answer

They are not competing options here. **"Distillation" in the practical LLM
sense IS fine-tuning** — the question is only where the training labels
come from:

| Approach | Labels from | What it buys you |
|---|---|---|
| Oracle SFT (this repo's `locode datagen`) | Deterministic solver, verified by execution | Perfect harness-protocol adherence: tool formats, edit discipline, plan/goal/ask behavior |
| Distillation-style SFT | A stronger teacher model's rollouts, **kept only if the verifier passes** (rejection sampling) | Teacher's judgment and diversity on top of protocol |
| True logit distillation | Teacher's token probabilities | Impractical across different tokenizers/families — skip it |
| RL (GRPO/PPO with verifier rewards) | The student's own rollouts | The real capability gains; do this *after* SFT, not instead |

**Recommendation: staged.** Stage 0 = oracle seed (done — this dataset).
Stage 1 = distillation-style SFT from Qwen 3.6 27B with verification.
Stage 2 (optional, later) = RL against the same verifiers.

**What SFT will and won't do — be realistic:** it will dramatically cut
malformed tool calls, wrong-tool choices, failed `edit_file` matches,
chattiness, and protocol violations (mutating in plan mode, forgetting
`GOAL_MET`, never asking clarifying questions). It will **not** make a 9B
model reason like a 27B. Protocol reliability is exactly where small local
models bleed in agentic harnesses, so this is still a large practical win.

## Base model choice

1. **Qwen 3.5 9B — start here.** Apache 2.0, strong native function-calling
   lineage, QLoRA fits a single 24 GB GPU.
2. **Gemma 4 E4B — cheapest iteration loop** for pipeline debugging;
   train on the *prompted* format (Gemma's tool templating via Ollama is
   the shakier path). Mind the Gemma Terms of Use for derivatives.
3. **Qwen 3.6 27B — later.** Distilling 27B-teacher → 27B-student is
   self-distillation: it narrows the model onto the harness distribution
   (still useful) rather than adding capability. Needs ~2×24 GB or an
   A100 for QLoRA.

## Generating the data

```bash
# Stage 0 — oracle (free, instant, format-perfect). Already shipped.
locode datagen --out data/oracle --count 400 --seed 7 --workers 12

# Stage 1 — teacher distillation with rejection sampling.
# Serve the teacher (needs real GPU memory):
vllm serve Qwen/Qwen3.6-27B --reasoning-parser qwen3 \
     --enable-auto-tool-choice --tool-call-parser qwen3_coder
locode datagen --out data/teacher --count 3000 \
     --teacher-url http://localhost:8000/v1 --teacher-model Qwen/Qwen3.6-27B

# Or a cloud teacher — GLM-5.2 (MIT weights, agent-trained) is the current
# best open pick. Metered OpenAI-compatible API:
locode datagen --out data/glm --count 3000 --workers 6 \
     --teacher-url <Z.ai OpenAI-compatible base> --teacher-model glm-5.2 \
     --teacher-key $LOCODE_TEACHER_KEY
# GLM Coding Plan / trial endpoints speak Anthropic format instead:
locode datagen --out data/glm --count 500 \
     --teacher-url <Z.ai Anthropic-compatible base> --teacher-model glm-5.2 \
     --teacher-key $KEY --teacher-format anthropic
```

Only trajectories that pass each template's programmatic verifier are
kept, so a wandering teacher costs attempts, not dataset quality. Teacher
mode runs single-worker by design (the GPU is the bottleneck anyway).

**Scale honestly:** 400 oracle samples teach format. For breadth you want
thousands of teacher trajectories over *varied real repos* — add your own
templates in `locode/datagen.py` (the `Template` dataclass is ~5 fields:
build a repo, phrase a prompt, optionally script an oracle, verify), point
templates at clones of small OSS projects, and re-run. Diversity of repos
and phrasings matters more than raw count.

## Data formats

- `train_native.jsonl` — `{"messages": [...], "tools": [...]}` with
  structured `tool_calls`. For trainers that apply the model's chat
  template with tools (LLaMA-Factory `tools` field, or
  `tokenizer.apply_chat_template(..., tools=...)` in a custom collator).
  Use for **Qwen** targets.
- `train_prompted.jsonl` — plain text messages using locode's Hermes-style
  `<tool_call>` protocol, including malformed→corrected **self-heal**
  sequences (~15% of tool calls). Trains with *any* chat template.
  Use for **Gemma** targets, and run the result with
  `locode --tool-mode prompted`.
- Real paths are scrubbed to `/home/dev/<repo>`; every tool result in the
  data was actually executed, never invented.

## Training recipe (QLoRA via Unsloth or LLaMA-Factory)

- LoRA r=16–32, alpha=2r, dropout 0.05, on all attention+MLP projections
- lr 1e-4–2e-4, cosine, warmup 3%, **2 epochs** (3 max — protocol data
  overfits fast), bf16, 4-bit base
- **Assistant-only loss** (`train_on_inputs: false` / label-mask user and
  tool turns) — this is the single most important switch; without it the
  model learns to imitate tool *outputs*
- Sequence length 8192, sample packing on; 9B QLoRA on 24 GB: micro-batch
  1–2, grad-accum to effective 16
- Mix ~10–20% general instruction data (e.g. a small open SFT set) to
  limit catastrophic forgetting of plain chat

## Evaluation — the only one that matters

1. Val-set token accuracy is a smoke test, nothing more.
2. **Live harness eval:** point locode at the fine-tune and measure on
   held-out generated tasks + a handful of hand-written real tasks:
   task success rate, malformed-call rate, mean steps per task,
   plan-mode violation rate, GOAL_MET false-positive rate.
3. Compare against the un-tuned base with identical prompts. If malformed
   calls don't drop sharply, the data/template pipeline has a bug — stop
   and inspect before scaling.

## Licensing — read this twice

- **Qwen** (Apache 2.0): outputs freely usable for training. ✅
- **Gemma**: fine-tunes are "Model Derivatives" under the Gemma Terms —
  redistribution carries conditions. Fine for internal use.
- **Claude / GPT as teacher: NO.** Anthropic's and OpenAI's terms prohibit
  using outputs to train competing models. Use open-weight teachers.
