# Bayesian Biomarker Effect Estimate for Combining Data from Multiple Biomarker Studies

In this study, we introduce a novel Bayesian Biomarker Pooling (BBP) method, employing Bayesian techniques to aggregate biomarker data from multiple study sources.

## Installation

```bash
# We suggest creating a separate environment. Below is an example using mamba (fast alternative of conda)
mamba create -n bbp_env python=3.11 -y
# Install mmAAVI package from github
pip install git+https://github.com/luyiyun/mmAAVI.git@main
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


## Reproduce the results

To reproduce the results, you can run the following code:

1. Clone the repository:

    ```bash
    git clone https://github.com/luyiyun/bayesian_biomarker_pooling.git
    cd bayesian_biomarker_pooling
    ```

2. Create a new conda/mamba environment and install the required packages:

    ```bash
    mamba create -n bbp_env python=3.11 -y
    mamba activate bbp_env
    pip install -e ".[develop]"  # install all dependencies for development in editable mode
    ```

3. Activate the conda environment:

    ```bash
    mamba activate bbp_env
    ```

4. Run the commands in `main.sh` to reproduce the results:

    ```bash
    bash main.sh
    ```

5. (Optional) If you want to generate the customized simulation data and run the model on them, you can run the following command:

    ```bash
    python main.py simulate --n_studies 4 --n_samples 20 --output_dir data/simu
    python main.py analyze --data_dir data/simu --output_dir results/simu
    ```
    you can run `python main.py --help` to see all available commands and options.