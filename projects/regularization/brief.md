# Title
Regularization in small neural networks

## Keywords
dropout, classification, reproducibility, limited compute

## TL;DR
Test whether dropout helps a small classifier when training data is scarce.

## Abstract
Use synthetic two-dimensional classification data with controlled label noise.
Compare an unregularized two-layer MLP with dropout variants using fixed data
splits and multiple seeds. Report validation accuracy, runtime and uncertainty.
Use no dataset downloads, keep each experiment under five minutes, and target
one GPU with at most 16 GB memory. Report negative results without changing the
hypothesis after observing the test set.
