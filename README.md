# Sparsified SGD with Memory

Code for the experimental part of the paper [Sparsified SGD with Memory TODO link](). It contains the code for the following experiments:

- Theoretical convergence with different sparsification operator
- Comparison with QSGD
- Multi-core experiments

Use `notebooks/plots.ipynb` to visualize the results.

Please open an issue if you have questions or problems.

### Reproduce the results

To reproduce the results, you can download the datasets from [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html)

```bash
mkdir data
cd data/
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2
```

We decompress the libsvm file and use pickle format instead. It takes more space but is faster to load. You can create a file as follow

```python
import pickle
import os
from sklearn.datasets import load_svmlight_file

if not os.path.exists('data'):
    os.makedirs('data')

X, y = load_svmlight_file('data/rcv1_test.binary.bz2')
with open('rcv1.pickle', 'wb') as f:
    pickle.dump((X, y), f)
   
X, y = load_svmlight_file('data/epsilon_normalized.bz2')
with open('epsilon.pickle', 'wb') as f:
    pickle.dump((X, y), f)
```

After updating the path to the data files in `experiment.py` , you can run the baseline

```bash
python3 baselines.py ./data results/baselines
```

and then run our experiments, for example

```bash
python3 experiment.py rcv1-th ./data results/rcv1-th --nproc 10
```

