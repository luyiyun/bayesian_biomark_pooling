from bayesian_biomarker_pooling.simulate import Simulator
from bayesian_biomarker_pooling.embp import EMBP


simulator = Simulator(type_outcome="continue")

df = simulator.simulate()
print(df)


