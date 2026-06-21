import because

import numpy as np

np.random.seed(42)
N = 100
data = {
    'Brain': np.random.randn(N),
    'Clutch': np.random.randn(N),
    'Migration': np.random.randn(N),
    'Mass': np.random.randn(N),
    'Lifespan': np.random.randn(N)
}

equations = [
    "Brain ~ Size",
    "Mass ~ Size",
    "Clutch ~ Brain",
    "Migration ~ Brain",
    "Lifespan ~ Clutch + Migration"
]

res = because.fit(
    equations=equations,
    data=data,
    latent=["Size"],
    dsep=True,
    dsep_only=True,
    num_samples=10,
    num_warmup=10
)

for test in res['dsep_results']:
    if 'response' in test:
        print(f"Test: {test['response']} _|_ {test['test_node']} | {', '.join(test.get('conditioning_set', []))}")
    else:
        print(f"Induced Correlation: {test.get('var1')} ~ {test.get('var2')}")
