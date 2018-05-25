# Sparsified SGD with Memory

Code for the experimental part of the paper [Sparsified SGD with Memory TODO link](). It contains the code the following experiments:

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
from sklearn.datasets import load_svmlight_file
X, y = load_svmlight_file('rcv1_test.binary.bz2')
with open('rcv1_test.pickle', 'wb') as f:
    pickle.dump((X, y), f)
```

After updating the path to the data files in `experiment.py` , you can then run our experiments, for example

```bash
python3 experiment.py rcv1-th results/rcv1-th --nproc 10
```

