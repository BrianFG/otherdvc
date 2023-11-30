import dvc.api
import pandas as pd
import json

exps = dvc.api.exp_show()
print(json.dumps(exps[2]))
df = pd.DataFrame(exps)
print(df)


print(dvc.api.metrics_show("exp2/evaluate/metrics.json"))