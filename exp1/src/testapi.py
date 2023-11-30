import dvc.api
import pandas as pd

exps = dvc.api.exp_show()
df = pd.DataFrame(exps)
print(df)