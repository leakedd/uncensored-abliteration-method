# Uncensored / Abliteration Method

This repository documents the weight editing and evaluation behind **SenseNova-Loop-9B Uncensored**, a Q4_K_M GGUF derived from the release checkpoint. The project studied how the weight edit changes refusals and affects ordinary answers in English and French. [Download the model on Hugging Face](https://huggingface.co/leakked/SenseNova-Loop-9B-UNCENSORED-HERETIC-GGUF)

## How abliteration works

Abliteration is a post-training weight edit. It projects a selected component out of matrices that write into the model’s residual stream. For an orthonormal basis $Q$ and a matrix $W$ whose output coordinates lie in that stream, the projection is $W' = W - Q(Q^\top W)$. The basis can contain one direction or several. A direction associated with refusals can also carry language and style information, so removing it can affect unrelated answers. [Arditi et al., NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/hash/f545448535dfde4f9786555403ab7c49-Abstract-Conference.html)

## How the SenseNova checkpoint was edited

1. **Start with the release checkpoint.** The source was the Q8_0 GGUF of SenseNova-Loop-9B. The edited model retained the original chat template.
2. **Inspect the activations.** The experiment compared internal activations associated with different response behaviors. SenseNova Loop reuses decoder blocks across two passes and has two residual streams. Activation captures must respect those boundaries for the comparison to be meaningful.
3. **Project out a subspace.** A single direction produced little change in this experiment. The selected edit used multiple directions and reduced overlap with language and style features before projecting the subspace out of the weights.
4. **Quantize and compare.** The resulting model was exported as Q4_K_M. An unedited Q4 checkpoint made with the same quantization recipe served as a baseline for measuring the additional drift from the edit.

## Technical details

### Why the activation capture mattered

SenseNova Loop has 32 physical decoder blocks and executes 24 of them twice, for 56 block executions per forward pass. A physical block index identifies a set of weights. An execution index identifies one use of those weights, with its own incoming activations. Mixing those indices can make a difference between two passes look like a difference between prompt groups. [Base architecture](https://huggingface.co/sensenova/SenseNova-Loop-9B#model-architecture)

The mHC router carries two residual streams through the loop. In the runtime used for the experiment, a capture hook initially treated a strided tensor view as a contiguous buffer. Reading the view that way associated some values with the wrong coordinates. The corrected analysis accounted for the tensor strides and kept the two streams and the two visits distinct when comparing activations. This matters because a projection is only meaningful in the coordinate system of the weights it edits.

### Why use a subspace

A single direction can correlate with refusals without being sufficient to change how the model responds. The first one-direction intervention had little effect in this checkpoint. The experiment therefore revisited the activations after intervening and looked for additional directions outside the basis already removed. Orthogonalizing those directions keeps redundant directions from counting as independent changes. The retained basis had eight directions.

The candidate subspace also overlapped with patterns found in ordinary answers and with differences between French and English activations. The project reduced that overlap before editing the weights. Early edits produced spelling errors and repetitive reasoning in French, which is why the evaluation tracked drift separately for each language.

### What the projection changes

Take a residual-writing matrix $W$ with shape $d_{model} \times d_{in}$ and an orthonormal basis $Q$ with shape $d_{model} \times k$. The columns of $Q$ live in output space. For an input $x$, $Q^\top Wx$ gives the coordinates of the write along the selected directions. Multiplying by $Q$ maps those coordinates back into the residual stream. The edit subtracts that contribution for every possible $x$.

$
W_{edited} = (I - QQ^\top)W
\qquad\Longrightarrow\qquad
Q^\top W_{edited} = 0.
$

The identity applies to each edited matrix before quantization and provides a direct check on the projection. In SenseNova Loop, the residual writes of interest include the output projections of attention, state-space layers, and feed-forward layers. A weight edit to a reused matrix takes effect on both visits. The identity describes those matrix outputs. Other paths through the network can still contribute to the full residual state, and quantization can introduce a small component along the projected directions again.

### How to read the evaluation

The matched unedited Q4 baseline controls for the effect of lowering weight precision when measuring drift. At the same token history, KL divergence is sensitive to changes across the entire distribution of next-token probabilities, including when both checkpoints choose the same token. The factual set measures whether the final answers are correct, which explains why the table can show the same factual score alongside measurable distribution drift.

On a separate 45-prompt refusal evaluation, an automated judge marked 27 responses from the edited Q4 model as fulfilling the request. The project logged zero explicit refusals on that run. Those counts measure different outcomes: a response can avoid an explicit refusal while stalling, answering only part of the request, or interpreting it differently. Both counts are specific to that sample and scoring procedure.

## Internal evaluation

| Metric | Original Q8_0 | Edited Q4_K_M |
| --- | ---: | ---: |
| Factual questions, direct mode | 30/30 | 30/30 |
| Factual questions, reasoning mode | 29/30 | 30/30 |
| KL divergence versus matched unedited Q4, English | — | 0.070 |
| KL divergence versus matched unedited Q4, French | — | 0.075 |

The factual scores describe a 30-question internal test set. KL divergence measures how much the output distribution changes. A percentage of retained capability requires its own defined benchmark.

For an evaluation, set aside prompts before choosing the edit, keep the chat template and generation settings fixed, and track explicit refusals, incomplete answers, and factual errors as separate outcomes. Report the sample size and checkpoint precision alongside each result.

## Credits

Base model by SenseTime / OpenSenseNova. [SenseNova-Loop-9B model card](https://huggingface.co/sensenova/SenseNova-Loop-9B)
