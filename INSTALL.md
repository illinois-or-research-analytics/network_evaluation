# Installation

Python 3.11 with:

- `graph-tool` (conda-forge)
- `pandas`, `numpy`, `scipy`, `scikit-learn`, `tqdm`, `matplotlib`,
  `seaborn` (conda)
- `networkit` (pip)
- `pymincut` (pip, from source)

## Conda recipe

```bash
conda create -n netev python=3.11 -y
conda activate netev
conda install -c conda-forge graph-tool -y
conda install numpy pandas scipy scikit-learn tqdm matplotlib seaborn -y
pip install networkit
pip install git+https://github.com/vikramr2/python-mincut
```

`pymincut` is built from source. Requires a C++ toolchain, `openmpi`,
and **`cmake >= 3.2` and `< 4.0`**. CMake 4.0+ also works if the old
policy is forced via `CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"`
on the pip install line.
