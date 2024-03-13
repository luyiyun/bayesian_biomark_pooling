import logging

from bayesian_biomarker_pooling.simulate import Simulator
from bayesian_biomarker_pooling.embp import EMBP


logger = logging.getLogger("EMBP")
logger.setLevel(logging.DEBUG)
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setLevel(logging.DEBUG)

# simulator = Simulator(type_outcome="continue")

# df = simulator.simulate()
# print(df)

# model = EMBP()
# model.fit(df["X"].values, df["S"].values, df["W"].values, df["Y"].values)


simulator = Simulator(type_outcome="binary", n_knowX_per_studies=95)

df = simulator.simulate()
print(df)

model = EMBP(outcome_type="binary")
model.fit(df["X"].values, df["S"].values, df["W"].values, df["Y"].values)

print(simulator.parameters["beta_x"], model.params_["beta_x"])
