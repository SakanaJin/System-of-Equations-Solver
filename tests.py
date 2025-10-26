import json
import main

with open('tests.json') as f:
    tests = json.load(f)

for test in tests.values():
    sol = main.eqsolve(test["Coeff"], test["rhs"])
    plswork = main.matmul(test["Coeff"], sol)
    plswork = [int(x) for x in plswork]
    if not plswork == test["rhs"]:
        raise ValueError("Big Probs on test: {}\n{}".format(test, plswork))

print("All Tests Passed")

