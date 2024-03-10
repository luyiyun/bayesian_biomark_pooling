from bayesian_biomarker_pooling.simulate import Simulator
from bayesian_biomarker_pooling.embp import EMBP


simulator = Simulator(type_outcome="continue")

df = simulator.simulate()
print(df)

model = EMBP()
model.fit(df["X"].values, df["S"].values, df["W"].values, df["Y"].values)
