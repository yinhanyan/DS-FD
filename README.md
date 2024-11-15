This is the official repository of our VLDB 2024 paper: [Optimal Matrix Sketching over Sliding Windows](https://www.vldb.org/pvldb/vol17/p2149-yin.pdf)

* **2024/08/29**: This paper was nominated for the Best Research Paper Award at VLDB 2024! Thanks to editors and reviewers for their recognition and support.
* **2024/11/15**: We added a simple max-heap optimization trick, which can significantly improve the performance furthermore without any extra space overhead and precision loss. Refer to `dump_fd.py` for more details.

# Install dependencies

```bash
pip install -r requirements.txt
```

# Prepare datasets

```bash
mkdir dataset

# Generate SYNTHETIC data
python generate_synthetic.py    

# Download BIBD data
wget https://www.cise.ufl.edu/research/sparse/mat/JGD_BIBD/bibd_22_8.mat --directory-prefix ./dataset 

# Download PAMAP2 data
wget https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip -o dataset/PAMAP2_Dataset.zip
unzip dataset/PAMAP2_Dataset.zip -d dataset

# Download RAIL data
wget https://www.cise.ufl.edu/research/sparse/mat/Mittelmann/rail2586.mat --directory-prefix ./dataset 

# Download YEAR data
wget https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip --directory-prefix ./dataset 
unzip dataset/yearpredictionmsd.zip -d dataset
```

# Run experimental scripts

The filenames of experimental scripts are prefixed with `test_*.py`.

# Installation & Usage

Under your project folder, pull this git repository:

```bash
git clone https://github.com/yinhanyan/DS-FD.git swfd --depth 1
```

Then you can import it in your project folder:

```python
from swfd import SeqBasedSWFD
from swfd import RobustFrequentDirections
from swfd import FrequentDirectionsWithDump

import numpy as np

if __name__ == "__main__":
    l = 16
    N = 64
    d = 32
    R = 2**16 - 1
    epochs = 2048
    sw_fd = SeqBasedSWFD(N, R, d, l)

    random_vector = np.random.randn(1, d)
    a = np.sqrt(R) * random_vector / np.linalg.norm(random_vector, ord=2)
    sw_fd.fit(a)
    B_t, _, _, delta = sw_fd.get()
    print(B_t)
```