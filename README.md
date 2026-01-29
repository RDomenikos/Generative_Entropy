# Generative Entropy — Reproducible Code for “A data-derived entropy functional beyond fixed-form blind spots”

This repository contains the complete code used to generate **all figures and results** in the manuscript:

> **“A data-derived entropy functional beyond fixed-form blind spots”**  
> George-Rafael Domenikos

The code in this repo is intended to be **fully reproducible**: running the provided scripts reproduces the paper’s figures end-to-end.

---

## What this repository does

- **Reproduces all paper figures** directly from code.
- **The naming of the .py file corresponds to the equivalent figure** (Fig_1 for figure 1 of the main text, Fig_S1 for figure 1 of the supplementary material)
- **Generates data programmatically** where applicable (simulations / synthetic experiments).
- **Accesses public/open datasets via standard libraries** when external data are required (e.g., market data), so there are no private datasets needed.
- Runs with **no manual parameter tuning**: the main scripts are configured to execute as-is and produce the same outputs reported in the paper.

