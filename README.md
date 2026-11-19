# Uncensored / Abliteration Method

This repository documents the weight editing and evaluation behind **SenseNova-Loop-9B Uncensored**, a Q4_K_M GGUF derived from the release checkpoint. The project studied how the weight edit changes refusals and affects ordinary answers in English and French. [Download the model on Hugging Face](https://huggingface.co/leakked/SenseNova-Loop-9B-UNCENSORED-HERETIC-GGUF)

## How abliteration works

Abliteration is a post-training weight edit. It projects a selected component out of matrices that write into the model’s residual stream. For an orthonormal basis $Q$ and a matrix $W$ whose output coordinates lie in that stream, the projection is $W' = W - Q(Q^\top W)$. The basis can contain one direction or several. A direction associated with refusals can also carry language and style information, so removing it can affect unrelated answers. [Arditi et al., NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/hash/f545448535dfde4f9786555403ab7c49-Abstract-Conference.html)

## How the SenseNova checkpoint was edited

1. **Start with the release checkpoint.** The source was the Q8_0 GGUF of SenseNova-Loop-9B. The edited model retained the original chat template.
2. **Inspect the activations.** The experiment compared internal activations associated with different response behaviors. SenseNova Loop reuses decoder blocks across two passes and has two residual streams. Activation captures must respect those boundaries for the comparison to be meaningful.
3. **Project out a subspace.** A single direction produced little change in this experiment. The selected edit used multiple directions and reduced overlap with language and style features before projecting the subspace out of the weights.
4. **Quantize and compare.** The resulting model was exported as Q4_K_M. An unedited Q4 checkpoint made with the same quantization recipe served as a baseline for measuring the additional drift from the edit.

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
