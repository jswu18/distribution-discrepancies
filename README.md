# distribution_discrepancies

An exploration of different distribution discrepancies between two distributions P and Q. A brief overview:

|Distribution Discrepancy|P Requirements|Q Requirements|User Inputs|
|-|-|-|-|
|Maximum Mean Discrepancy (MMD)|At least 2 samples|At least 2 samples|Kernel
|Kernel Stein Discrepancy (KSD)|Distribution|At least 2 samples|Kernel, Distribution
|Fisher Divergence|Distribution|At least 1 sample|Distribution

To get set up:

1. Install `poetry`

```shell
pip install poetry
```

2. Install dependencies

```shell
poetry install
```
