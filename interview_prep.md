# Mock Interview — sae-mamba Project Deep-Dive

Practice Q&A for the project deep-dive portion of an Anthropic Research Engineer interview. Questions are grouped by theme; answers are in first person, focused on **why** I chose each approach, what I considered and rejected, and the limitations I'd address given more time.

---

## A. Motivation

### Q1: Why did you pick this project?

Three reasons.

1. **Open, testable question.** SAEs have been applied to transformers, vision transformers, VLMs, and protein models — but never to state space models. Whether Mamba's recurrent state is sparsely decomposable is a yes/no question I could actually answer in a 3-day compute budget, which matters for independent research where a 6-month commitment before knowing if the premise works is too expensive.
2. **The comparison structure is natural.** Mamba and Pythia at matched 2.8B scale, trained on the same data (The Pile), with the same tokenizer family (GPT-NeoX), give a clean A/B for "does this technique translate across architecture families." That's scaffolding for real conclusions, not just characterization of one model.
3. **Directly adjacent to Anthropic interp.** Olsson's induction heads, Gao's TopK SAE, Lindsey's crosscoders, the mamba_ssm hook conventions — I wanted to work in the vocabulary of the team I was applying to.

### Q2: Why induction specifically? There are many behaviors you could have localized.

Induction is the most-studied circuit in transformer interp (Olsson et al. 2022). That gives me two things: (1) a Pythia control where I know where the answer "should" be based on prior literature, so any strange Mamba finding can be contrasted against a baseline. (2) A clean behavioral signature — repeated random bigrams are easy to construct, unambiguously have a "correct" next-token prediction, and the signal is sharp enough to show up in SAE features without fine-tuning.

If I'd picked factual recall or arithmetic I'd have spent days arguing about whether I had the right clean/corrupted pairs. Induction eliminates that.

### Q3: Induction in SSMs has already been studied (Arora et al. "Zoology"). Isn't this just reproduction?

Fair pushback. What's new isn't "does Mamba do induction" — Arora et al. show behaviorally that it does. What's new is **where, mechanistically, inside the mixer**. Their paper reports that Mamba is worse at induction than attention; they don't localize the circuit. My contribution is (a) activation patching across the five internal submodules × 17 layers, (b) slice-level patching of x_proj into (Δ, B, C), and (c) the finding that the 16-dim C matrix at L30 carries 80% of the signal. That's a circuit-level claim, not a behavioral one.

---

## B. Methodology choices

### Q4: Why TopK SAE and not L1 / Gated / JumpReLU?

TopK (Gao et al. 2024) gives me a direct sparsity knob — set K, that's the active count — without the hyperparameter search that L1 needs. L1 also has the "shrinkage" issue where high-activation features are systematically under-reconstructed. JumpReLU is stronger but needs STE tricks that add one more point of failure. For a 3-day project, TopK is the right point on the complexity-robustness Pareto frontier.

For a full paper, I'd redo headline numbers with JumpReLU as a sensitivity check. My §8c SAE-hyperparameter sweep verified the headline survives K ∈ {32, 64, 128} and x ∈ {8, 16, 32}, which is the coarser version of that worry.

### Q5: How did you pick layer 32 for the SAE?

50% depth of Mamba-1 2.8B's 64 layers. I didn't optimize for "best induction layer" because that would be contamination — I wanted to pick a layer independent of where the mechanism lives, then let patching tell me.

What I found: the causal locus is L30, not L32. The L32 residual contains L30's contribution, so reading features there is fine, but if redoing I'd train SAEs at multiple layers and pick the one where induction-feature scores peak.

### Q6: Why synthetic induction pairs with random tokens? Natural text seems more realistic.

Synthetic pairs have no confounds: uniform random tokens guarantee the only thing distinguishing clean from corrupted is whether the second pattern matches the first. Natural text fights semantic plausibility, co-occurrence statistics, frequency effects — any of which might drive the gap more than induction itself.

I validated on natural text post-hoc: induction features fire 5.4× higher on 26,807 Pile bigram repeats, and the C-matrix patch on real-text pairs is +0.83 (vs synthetic +0.80). The synthetic stimulus generalizes, so the conclusion survives.

### Q7: Why activation patching, not attribution patching?

Attribution patching linearizes the patch via a gradient. For Mamba, the selective scan's `exp(A·Δ)` term is highly non-linear in the activations. I was worried the linearization would be inaccurate at exactly the critical parts.

Turns out I was right — my cross-layer Ridge regression (L30 internal SAE → L32 induction features, 40,960 dims → 10 dims) has CV R² = −1.63. A linear approximation across a single Mamba layer is already negative-R² bad. Attribution patching relies on a similar linearization across the patching site; I don't trust it here.

### Q8: You only used one seed for the original scoring. Why is that OK?

The *positions* of the induction (48–55) are deterministic; only the token identity varies. Patch damage is a mean over 128 pairs × 8 positions = 1,024 induction positions, which is enough for statistical stability at the 1–3% level.

The real question is whether the *identified feature set* is stable across seeds. I validated that separately (multi-seed robustness): Jaccard = 1.0 for top-10 features across 5 seeds. So both the mean and the feature set are stable.

---

## C. Design decisions

### Q9: Why 10M tokens for SAE training?

Trade-off. Published SAE work uses 50M–1B; at that scale a 4× H100 pod would take 3 days just for extraction, and 10M × 2560-dim fp32 is already ~100 GB per layer per model — near disk limits on a constrained pod.

Risk of undertraining: dead features + poor reconstruction. My FVE ≈ 0.7 across configurations and dead-feature counts < 5% suggest it's not catastrophically undertrained, and Jaccard = 1.0 across seeds tells me the feature identities are stable. For a paper I'd scale to 100M tokens; for an independent 3-day sprint, 10M is the right point.

### Q10: Why normalize activations per-dimension? That's not standard.

It's a methodological finding from Phase 2 of the main report. Without normalization, Pythia's middle-layer SAEs diverge to FVE below −5 — catastrophic failure. Reason: Pythia's per-dim activation variance grows 55× from L0 to L28 (0.13 → 7.2), and the SAE encoder bias never converges at certain layer scales. Mamba's variance growth is much tighter; it trains fine without normalization.

For a cross-architecture comparison, normalization is a prerequisite — otherwise any "Mamba beats Pythia" claim is attributable to the training recipe, not the architecture. I flagged it in the main report as a concern for broader SAE literature; several published "middle-layer transformer SAE FVE" numbers may have been trained without normalization and therefore be unreliable.

### Q11: Why compare with Pythia specifically?

Matched scale (both 2.8B), matched pretraining data (both trained on the Pile), matched tokenizer family (GPT-NeoX BPE; vocabs 50,277 / 50,304). This isolates architecture. Any difference between Mamba-1 and Pythia-2.8B has to come from SSM vs. attention, not scale or data.

I considered GPT-J 6B (different data + scale) and LLaMA (different tokenizer); both would have added confounds.

### Q12: The `force_slow_forward` fix sounds hacky. Why not something cleaner?

It is hacky, and I flagged it as such. The underlying issue: HF Mamba's default forward dispatches to `cuda_kernels_forward`, which calls `causal_conv1d_fn(x, self.conv1d.weight, self.conv1d.bias, ...)` directly — passing the Conv1d weight and bias as tensors rather than calling `self.conv1d(x)`. Forward hooks on the Conv1d module therefore never fire.

The clean fix is to monkey-patch `cuda_kernels_forward` to call `self.conv1d(x)` explicitly, or submit the upstream PR. Both are more effort. `force_slow_forward` replaces the dispatcher with `slow_forward` (which does call `self.conv1d(x)`) on a per-instance basis. It's slower per forward but the capture/patch sweeps use batch size ≤ 8, so the cost is fine.

For a production context I'd submit the upstream PR; for a 3-day sprint, the one-liner helper did the job.

---

## D. Results interpretation

### Q13: 83% patch damage from 16 dims — is that unusually sharp?

Yes, at the higher end of what's been reported. For context: Olsson et al.'s induction heads in GPT-style models show damage distributed across multiple heads at multiple layers. Even the "name-mover heads" in Wang et al.'s IOI paper get perhaps 50–60% damage for the dominant head. 83% from a single 16-dim slice is unusual.

That said, selective scan has a specific structural role for C (`y = C · h`); of course patching C has disproportionate impact because it's literally the readout channel. What convinces me it's real rather than an SAE artifact is that the SAE-feature metric and the next-token logit metric agree in direction (47.5% logit damage).

### Q14: Explain the necessary-vs-sufficient finding.

Two interventions:
- **Necessity** (patch corrupted → clean): replace clean C with corrupted; induction drops 80%. Removing this 16-dim signal almost completely kills induction.
- **Sufficiency** (patch clean → corrupted): start from a corrupted run, inject clean C; induction recovers only 5.6%.

The asymmetry tells me the mechanism is **distributed in state-building and concentrated at readout**. The SSM state h is built across layers 0–29 from the clean input tokens; L30's C is a readout direction that depends on h. When h is wrong (because earlier tokens were corrupted), no amount of injecting the "right" C produces the right answer — `y = C · h` with a wrong h still gives a wrong y.

The mechanism isn't a one-layer lookup; it's state-accumulation ending in a concentrated readout.

### Q15: The linear probe says C is only 57% decodable but 80% causal. Seems contradictory.

It's the **representation-vs-computation distinction**. A linear probe measures whether a feature *encodes* clean-vs-corrupted in a linearly readable form. Patching measures whether it *causes* the downstream behavior change.

C contains only 57% of the label (low decodability) but causes 80% of the effect (high patch damage). Not contradictory once you notice `y = C · h` is a multiplication. C on its own is a vector of directions; it doesn't encode "is this the matched pattern?" as a linear readout. The match detection happens in the *product* C · h, where C is combined with the state.

Meanwhile Δ_pre is 97.7% decodable but 1% causal — it redundantly encodes a lot of clean-vs-corrupted information that the network doesn't use.

Useful caution against "find where the probe works and call that the mechanism." The probe and the causal locus can diverge, and the causal locus is the one that matters for behavior.

---

## E. Limitations, what you'd do differently

### Q16: One model per family — how confident are you the findings generalize?

Moderately confident for Mamba-1 2.8B; less so as a universal claim.

**For**: replicates at Mamba-130M (L15 C-matrix, +1.64 logit damage, 28% of gap — same slice, different depth). Two data points at different scales.

**Against**: Mamba-370M shows weaker localization (max +0.04 logit damage, no clear locus). Either the mechanism isn't as concentrated at that intermediate scale, or the logit-metric under-reports on a very-confident model. Honestly uncertain.

A defensible claim: "the C-matrix slice of x_proj is the canonical location for induction in selective-scan Mamba at 2.8B scale; whether this holds at all scales needs more work." The Mamba-2 result (15× weaker induction under SSD) tells me the localization is specific to selective scan, not SSMs in general.

### Q17: You mention Mamba-2 has 15× weaker induction. Could that be an SAE artifact?

Possible but unlikely. Evidence against:
- The L32 Mamba-2 SAE trained with identical recipe (FVE 0.71, dead < 5%) — comparable quality to Mamba-1's.
- Pattern-length sweep: gap is 0.16–0.28 across plen ∈ {4, 8, 16, 32} — grows with pattern length but stays 10–20× below Mamba-1. Not an artifact of short patterns.
- Independent: Arora et al. "Zoology" reports Mamba-2 (SSD) underperforms selective-scan Mamba on in-context learning tasks.

Residual concern: would a Mamba-2-specific SAE (different K, different layer) find induction more cleanly? I tested K ∈ {32, 64, 128} and x ∈ {8, 16, 32}; none changed it dramatically. Not exhaustive but suggestive.

### Q18: If you had two more weeks, what would you do?

In order:
1. **Gradient attribution at the state level.** ∂(induction feature) / ∂(SSM hidden state h). Maps which state components route through C. Separates "state has the info" from "C reads it out."
2. **Crosscoder between Mamba L30 and Pythia L10.** Lindsey et al. 2024 style. Stronger universal-features claim than the CKA = 0.80 I currently have.
3. **Scaling to Mamba-1 at 1.4B and 7B.** Three data points (130M, 2.8B, 7B) would pin down whether the C-locus scales cleanly or shifts with parameter count.
4. **Full natural-text Phase-B sweep**, not just slice-level. Closes the "synthetic-only patching" caveat entirely.

### Q19: Your sufficiency test gave 5.6% rescue; residual-stream sufficiency gave the trivial 100%. Is there a better sufficiency experiment?

Yes — residual-stream patching is too coarse (overwrites everything downstream). Slice-level is right granularity but has the opposite issue: it only restores 5.6% because the state h is wrong.

The cleanest sufficiency test I didn't get to: **state patching**. Capture the SSM hidden state h at L30 on a clean run; inject it into a corrupted run; let only the clean C proceed. That isolates "is clean C + clean state sufficient?" — which my current experiment doesn't test.

Implementing it requires monkey-patching MambaMixer.forward to expose h; non-trivial but a week's work.

---

## F. Anthropic-specific framing

### Q20: How does this project map onto Anthropic interp research?

Three contact points:
- **Sparse autoencoders** (Cunningham, Sharkey, Bricken, Rajamanoharan, etc.): the project is built on the TopK SAE recipe Gao et al. developed with the Anthropic line. AuxK loss, dead-feature resampling, normalized-activation training — all Anthropic-team ideas I reused.
- **Induction circuits** (Olsson et al.): the research question is defined relative to Olsson's attention-head framing. Phase B + Phase C is explicitly "Mamba vs. transformer, using the same method Olsson used."
- **Crosscoders** (Lindsey, Chughtai 2024): my planned Mamba ↔ Pythia crosscoder — which I didn't get to — is the natural next step from my main project's SAE CKA = 0.80 finding. Crosscoders would test the shared-dictionary claim more mechanistically than CKA can.

### Q21: What would you work on at Anthropic if hired?

Three directions I'd be drawn to:
- **Feature drift during fine-tuning / RLHF.** Train SAE on base model, watch which features change after post-training — an alignment-relevant diagnostic.
- **Scaling the circuit-level methodology.** Mamba induction generalizes; factual recall, refusal behaviors, reasoning steps should be accessible with the same toolkit (capture + patch + SAE on pre-sensitive activations).
- **Cross-model universality.** CKA = 0.80 + crosscoder testing: if architecture-independent feature dictionaries exist, they're a load-bearing tool for safety reasoning.

I'd also be happy to be the RE who makes training / capture / eval pipelines reusable. My repo already organizes that way (`src/` with typed signatures, `scripts/` numbered + documented, `report_*.md` as small papers, `web/index_*.html` as reviewable dashboards).

---

## G. Pushback / hard questions

### Q22: You found 16 dims carry 80% of induction. But C is *only* 16 dim in the SSM. Isn't this just "a Linear output has high impact"?

Counter: **patching Δ_pre (160 dim, same x_proj output, same layer) gives only +0.012 damage**. So "smaller dim = higher impact" doesn't predict the result. Independent check: patching dt_proj (which produces discrete Δ for SSM) at any layer gives ≤ 0.01. SSM-parameter patching isn't automatically high-impact; only C is.

### Q23: Internal and L32 SAE top-10 feature sets don't overlap. How do you know you're measuring the same "induction feature"?

You can't, at the feature-index level — SAEs are basis-dependent. But I can show:
- Both SAEs, patched at L30 x_proj, give the same causal localization (+0.69–0.84 across 5 configs).
- The cross-layer linear map has negative R² — not linearly equivalent, expected because selective scan is non-linear between sites.
- Qualitative max-activating text matches between sites (both have features that fire on "supra shoes supra shoes" or "CT-based cup ... cup").

I'm measuring the same *mechanism* in two bases, not the same *feature indices*. That's the typical pattern in the SAE interp literature.

### Q24: 64 synthetic pairs is low n. What's your error bar on the 0.833 headline number?

No formal error bar because it's a batch mean, not a statistical test. Bounds:
- Multi-seed robustness: 5 seeds × 128 pairs each → same Jaccard = 1.0, same causal ordering.
- Retrained-vs-original-SAE replication: L30 x_proj = 0.809 (retrained) vs 0.833 (original), ∆ = 0.024 — noise floor at this n is ~±0.02.
- Pattern-length sweep: 0.73–0.77 across plen ∈ {4, 8, 16} — wider band because plen changes mechanism slightly.

With more compute: 10 seeds × 512 pairs and report ± 2σ.

### Q25: Natural-text validation uses 26,807 positions but patching was only 128 pairs. Isn't n pretty small for the patching?

Yes. I kept it small because each sweep re-forwards 128 × 17 × 5 = 10,880 forward passes under slow_forward. More pairs scales compute linearly.

The result (C = +0.831) matches synthetic (+0.800) within 4%, consistent with my ±0.02 noise floor. I'm confident it's not a lucky draw. For a paper I'd use 512–1024 pairs.

---

## H. Softer questions

### Q26: Three days to do all this — be honest about trade-offs.

Three:
- **SAE sizes on the small end** of published work (40,960 features at 10M tokens). Real paper would be 200K features at 100M tokens.
- **Error bars are informal.** I lean on cross-experimental consistency (retrained SAE, multi-seed, different hyperparameters) rather than statistical tests.
- **No crosscheck with attribution or edge patching.** Result could have an activation-patching-specific artifact (though I'm not aware of one). Second methodology would be a useful cross-check.

None fatal, but I'd address them for conference submission.

### Q27: What surprised you most?

The C-vs-Δ asymmetry. Going in, if induction localized anywhere specific inside x_proj I expected **Δ** — "the model decides this position is important and updates the state strongly."

What I found: Δ is causally inert (+0.012). Induction is in C, which **reads out** rather than writes. Mechanistically makes sense in hindsight — C selects which direction of the state to project to residual, and pattern memory lives in the state — but I had to rewrite my mental model after the slice-patching experiment. The linear-probe finding (C is 57% decodable but 80% causal) was the third round of updating: even C doesn't "contain" the match; the multiplication `C · h` does.

### Q28: Close out — the single most interesting claim?

"In Mamba-1 2.8B, the attention-free induction mechanism is executed by a 16-dimensional state-readout direction at layer 30. Those 16 dims — 0.01% of the hidden state at that depth — don't themselves contain linearly-decodable pattern information, but multiplying them with the accumulated SSM state produces the match-detected signal. Remove those 16 dims and 80% of induction-feature activation disappears; 48% of next-token prediction logit disappears with it."

That's the sentence I lead with.
