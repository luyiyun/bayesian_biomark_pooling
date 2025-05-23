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
dat = pd.read_csv(osp.join(save_root, "example_data.csv"), index_col=0)

# fit model
model = BBP()

model.fit(dat)
baye_res = model.summary()
print(baye_res)
```

The example data is as follows:

```csv
,W,Y,X,S,H
0,-3.054489864402056,0,-0.3604401709908981,1,True
1,-2.84033779959993,1,0.5835341273827435,1,True
2,-3.078838674051937,1,-1.4385226489904384,1,True
3,-1.8356983674822998,0,2.118803030729321,1,True
4,-4.206691599804332,0,-1.3420444532864415,1,True
5,-2.1785013148080212,1,0.9198072605649881,1,True
6,-2.256561294339474,1,-1.121122678939223,1,True
7,-1.4774775212548814,1,1.1508830312317526,1,True
8,-3.8961222491114937,0,-0.3847740266090023,1,True
9,-4.186210017464947,1,0.15842290716221127,1,True
10,-3.596602584683985,0,0.05334375570673462,1,True
```

The columns are defined as follows:

- `W`: the local biomarker effect measurements
- `X`: the reference biomarker measurements
- `Y`: the binary disease outcome (1 for disease, 0 for control)
- `S`: the study indicator
- `H`: the indicator for whether the reference biomarker is available

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