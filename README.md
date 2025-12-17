<h1 align="center">
HyRex<!-- omit from toc -->
</h1>
<h4 align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)


</h4>

HyRex is a Python+JAX package for differentiable computation hydrogen and helium recombination.  HyRex is a JAX implementation of HYREC-2, and matches HYREC-2 output at sub percent-level accuracy.  HyRex is meant to be used in conjunction with a differentiable Boltzmann solver for the CMB (see [ABCMB](https://github.com/TonyZhou729/ABCMB)), but can also be used as a standalone code.

## Installation
We recommend installing HyRex in a clean conda environment.  After downloading and unpacking the code, in the code directory run 
```
conda create -n HyRex
conda activate HyRex
pip install -r requirements.txt

```
optionally specifying your preferred python version after the environment name.  Note that this will automatically install JAX for CPU, since HyRex's CPU performance is superior to its GPU performance.  However, if you would rather install JAX for GPU, refer to the [JAX documentation](https://docs.jax.dev/en/latest/installation.html) for a quick JAX installation guide.

## Examples
We have included an [example notebook](https://github.com/TonyZhou729/HyRex/blob/main/notebooks/hyrex_intro.ipynb) that takes you through the basics of using HyRex; start there if you're new to the code!

## Issues
Please feel free to open an issue if you notice something is amiss in HyRex!
