# Bayesian Biomarker Effect Estimate for Combining Data from Multiple Biomarker Studies

In this study, we introduce a novel Bayesian Biomarker Pooling (BBP) method, employing Bayesian techniques to aggregate biomarker data from multiple study sources.

## Installation

```bash
# We suggest creating a separate environment. Below is an example using mamba (fast alternative of conda)
mamba create -n bbp_env python=3.11 -y
mamba activate bbp_env
# Install BBP package from github
pip install git+https://github.com/luyiyun/bayesian_biomark_pooling.git@main
```

## Usage

```python

# read data
dat = pd.read_csv(osp.join(save_root, "BRCA1_simu.csv"), index_col=0)

# fit model
model = BBP()

model.fit(dat)
baye_res = model.summary()
print(baye_res)
```

## Reproducing the Results

To reproduce the results in the paper, please follow the steps below:

1. Clone the repository:

```bash
git clone https://github.com/luyiyun/bayesian_biomark_pooling.git
cd bayesian_biomark_pooling
```

2. Create a new conda environment and install the required packages:

```bash
mamba create -n bbp_env python=3.11 -y
mamba activate bbp_env
pip install -e ".[develop]"
```

3. Run the scripts to reproduce the results:

```bash
cd experiments
bash run2.sh
```