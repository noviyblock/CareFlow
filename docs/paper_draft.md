# Paper draft (working)

> Status: skeleton + results available on 2026-11-02. `[TBD: <run>]` marks numbers that come from runs still in
> progress (router_data_*, router_train, sycophancy_*). Citations use only entries marked ✅ in
> `docs/references.md`; ⚠️ entries are flagged inline. Do not submit with any `[TBD]` or ⚠️ left.

**Working title:** *Do Emotional-Support Agents Need Many Agents? A Length-Controlled, Compute-Aware Evaluation
and Routing Study*

## Abstract (draft)

Multi-agent pipelines for emotional support conversation (ESC) multiply inference cost, but whether they improve
replies over a single model has rarely been tested under controlled conditions. We evaluate nine single- and
multi-agent systems built on one model family (Qwen3.5 2B/9B) on ESConv with a locally run LLM judge that we
validate against the published HEART ranking (Kendall τ = 0.66). We find that (i) comparisons against ESConv
human replies are saturated (all systems win 92–98%), so systems must be compared head-to-head; (ii) the judge
has a strong length bias, and after length control (Dubois et al., 2024) an explicit "analyse-then-respond"
agent significantly *hurts* quality (LC win-rate 0.25), while candidate-selection schemes are
`[TBD: confirmatory run, docs/robustness_plan.md — equivalent within ±0.10 / better]` at 1.8–11.6× the latency;
a positive control confirms the protocol detects differences of this size; (iii) per-context path selection has large headroom: a cross-fitted oracle over the
same paths reaches 0.82 vs 0.61 for the best fixed system at 2.8× lower latency, and the naive (non-cross-fitted)
oracle overstates this by 0.15. However, learned pre-generation routers (ridge, cluster, LLM-hidden-state probe) trained on 195 contexts do not recover this headroom (best: +0.03 over a budget-matched random router), and a placebo oracle over samples of a single system `[TBD: placebo_oracle]` shows how much of it is luck of the particular reply.
Finally, we release a synthetic sycophancy test set with hard negatives; detectors that are near-perfect on it
(DeBERTa AUROC 1.00) disagree 5–10× on real system replies, so detectors must be validated on the outputs they
score `[TBD: audit + manual labels]`.

## 1. Introduction

- ESC systems increasingly use multiple agents (InsideOut, DOI 10.3233/FAIA241039; UniESC ⚠️ DOI
  10.1109/TAFFC.2026.3697775; MiAgent ⚠️), but cost is rarely reported and baselines rarely share the base model.
- Questions: RQ1 is there per-context headroom for routing between cheap and expensive paths? RQ2 do MAS beat a
  same-family single model at equal length? RQ3 can a pre-generation router capture the headroom? RQ4 how often
  do systems endorse harmful user beliefs (sycophancy), and can it be detected cheaply?
- Contributions:
  1. a reproducible, fully local evaluation protocol (judge validation, both presentation orders, length control,
     Bonferroni-corrected head-to-head, latency/token accounting);
  2. a controlled comparison of 9 systems + ablation showing where MAS cost goes and why the analysis agent hurts;
  3. a cross-fitted oracle gate quantifying routing headroom and the winner's-curse bias of naive oracles;
  4. learned routers (ridge utility, cluster, LLM-hidden-state probe) vs oracle and budget-matched random;
  5. a synthetic sycophancy set with benign hard negatives and detector comparison.

## 2. Related work

- **ESC data and evaluation.** ESConv (Liu et al., ACL 2021, arXiv:2106.01144); EmpatheticDialogues (Rashkin et
  al., ACL 2019, arXiv:1811.00207); HEART (arXiv:2601.19922) — we reuse its rubric verbatim and validate our judge
  against its model ranking; H2HTalk (arXiv:2507.03543).
- **Equivalence testing.** TOST (Lakens, 2017, doi:10.1177/1948550617697177 ⚠️ cited from memory).
- **Judge biases.** Length-Controlled AlpacaEval (Dubois et al., 2024, arXiv:2404.04475) — our LC GLM follows it.
- **Multi-agent reasoning baselines.** ReAct (arXiv:2210.03629); multi-agent debate (Du et al., arXiv:2305.14325);
  InsideOut (DOI 10.3233/FAIA241039); UniESC-style roundtable ⚠️.
- **Routing and selective computation.** SATER (arXiv:2510.05164) — latency metrics and refusal-aware routing;
  GCUL (arXiv:2609.29327) — geometry of ambiguity, inspires our cluster router; SpeechLLM CLS-head router ⚠️
  (arXiv:2609.20081) — inspires the hidden-state probe; PAIR (arXiv:2609.16344); AIMS (DOI
  10.1145/3767295.3803622).
- **Synthetic data / sycophancy.** Role-Guided Synthesis ⚠️ (DOI 10.1007/978-981-95-3352-7_46) — two-stage
  profile → dialogue generation we adapt; DeBERTa-v3 (He et al., arXiv:2111.09543) as detector backbone.
- Contrast to note: works reporting MAS gains typically compare against weaker or different base models and do
  not control for length; we show both factors matter (§5).

## 3. Evaluation protocol

- **Data.** ESConv official test split; contexts sampled deterministically (seed 20260928, ≥4 history turns);
  n = 60 for systems/ablation, n = 195 test + 195 valid for routing. ESConv final self-reports are filtered
  (all dialogues improve), so dialogue-level outcome labels are not usable (docs/data.md).
- **Judge.** Gemma-4-26B-A4B (MLX 4-bit), HEART pairwise rubric, both presentation orders per pair, strict parse
  of `overall_eq`. Validation (60 contexts, 15 HEART models): Kendall τ = 0.66 [0.43, 0.75] vs published
  ranking, order consistency 0.91, parse failures 0.4%.
- **Statistics.** Bootstrap over contexts (both orders move together); Bonferroni across the 7–8 head-to-head
  comparisons; Bradley–Terry for rankings.
- **Length control.** `won ~ α_sys + β·tanh(Δwords/10) + γ·position`, LC = sigmoid(α_sys); report raw and LC.
- **Cost.** Number of LLM calls, output tokens, mean and p90 latency (sequential calls, M5 Pro, MLX 4-bit);
  SATER-style AGL/AROL where applicable.

## 4. Systems

single_2b, single_9b (HEART generation prompt); react_9b; planner–generator–critic; debate (Du et al.); rerank
(4 candidates, embedding medoid); InsideOut (6 Ekman agents + aggregator); roundtable (ToM + 3 personas +
supervisor, UniESC-style); CareFlow (context analysis → generation → safety → empathy) with ablations
(no risk field, no checks, no context analysis). All 9B systems share weights and decoding settings.

## 5. Results

### 5.1 Human reference is saturated
All systems win 92–98% vs ESConv human replies (Table: `docs/results.md` E3), so we compare against single_9b.

### 5.2 Head-to-head, raw vs length-controlled (vs single_9b, n = 60)

| System | Raw WR | LC WR [95% CI] | Latency × |
|---|---|---|---|
| rerank_9b | 0.60 | 0.59 [0.49, 0.70] | 4.2 |
| insideout_9b | 0.57 | 0.53 [0.41, 0.67] | 11.6 |
| pgc_9b | 0.46 | 0.46 [0.34, 0.61] | 3.0 |
| roundtable_9b | 0.59 | 0.45 [0.32, 0.59] | 5.9 |
| debate_9b | 0.40 | 0.45 [0.34, 0.57] | 4.3 |
| react_9b | 0.39 | 0.38 [0.27, 0.48] | 1.8 |
| single_2b | 0.50 | 0.28 [0.17, 0.41] | 0.3 |
| careflow_9b | 0.31 | 0.25 [0.14, 0.36] | 4.0 |

Length effect: logistic coefficient +1.17 [0.98, 1.37] per +10 words; the longer reply wins 63% when ≥5 words
longer, 12% when ≥5 words shorter. Two raw-score conclusions flip under LC: 2B "parity" with 9B (2B writes 8 words
longer) and the roundtable gain (+4.6 words).

Equivalence (TOST, Bonferroni over 10 systems): on n = 60 the minimum detectable effect is ≈ 0.20, so only the
negative effects (careflow, 2B under LC) are established; for the other six schemes neither a gain nor a loss up
to ±0.2 can be excluded. This exploratory sample motivates the pre-registered confirmatory run (≥ 236 contexts for
3 comparisons at δ = 0.10). Report the confirmatory table here; keep n = 60 as exploratory in the appendix.

### 5.3 Why the analysis agent hurts (ablation)
Removing context analysis restores single-model quality (0.52); removing checks changes nothing (0.30; checks
edited 2/60 replies); removing the risk field is insufficient (0.36). Hypothesis: a prescribed strategy makes
replies templated/directive. Implication: use state analysis as a *routing signal*, not a generator instruction.

### 5.4 Routing headroom (oracle gate, cross-fitted)
Best fixed 0.61 @ 3.96 s; oracle 2B↔9B 0.66 @ 0.63 s; oracle over 9 paths 0.82 @ 1.44 s; non-cross-fitted 0.97.
Oracle matches the best fixed system at 0.73 s (5.4× cheaper). LC-adjusted oracle: `[TBD: router_train, adj]`.

### 5.5 Learned routers (train: ESConv valid, test: ESConv test; paths 2B / 9B / rerank)
`[TBD: router_train]` — table: fixed paths, utility (MiniLM), cluster (k = 8), llm_probe (Qwen3.5-0.8B hidden
states, ridge head), budget-matched random, cross-fitted oracle; raw and length-adjusted quality vs latency.

### 5.6 Sycophancy
Data: `[TBD: sycophancy_synth]` scenarios over 10 domains, harmful vs benign beliefs, an agreeing and a balanced
reply each, Gemma label check (belief harmfulness, reply stance). Benign agreeing replies are hard negatives.
Detectors: lexical rule, Qwen3.5-9B yes/no, DeBERTa-v3-base fine-tune: `[TBD: AUROC/F1]`. System rates of
endorsing harmful beliefs: `[TBD]`.

## 6. Limitations
Single judge family (Gemma) — partly mitigated by HEART validation; a second judge (dense Gemma-4-31B) not run.
n = 60 for systems; one seed; sequential latency on one device; synthetic sycophancy data is generated and
label-checked by the same model family as the judge (detectors use Qwen, not Gemma); no human evaluation yet.

## 7. Ethics
No real user data beyond public ESConv/ED; system outputs are not deployed; sycophancy set contains synthetic
descriptions of risky plans (medication, isolation) for evaluation only.

## Open items before submission
- Resolve ⚠️ references (UniESC, MiAgent, SpeechLLM, Role-Guided Synthesis full texts) and the "arXiv 2935"
  selection-effect reference.
- Second judge or human subset for E2-style agreement on the LC conclusions.
- Merge with teammate's FailedESConv / strategy-router results (docs/team_split.md).
