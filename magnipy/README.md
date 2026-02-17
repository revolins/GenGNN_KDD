# magnipy: Metric Space Magnitude Computations -- adapted for GenGNN

This is a repository for computing the **_magnitude of a metric space_**, which encodes the **effective size, diversity, and geometry** of a metric space. Given a dataset or distance matrix, **_magnitude_** measures the **effective number of distinct points** in the space at a scale of dissimilarity between observations.
We introduce the following codebase to compute and compare the magnitude of metric spaces.

## 🔍 Main Functionalities and Classes

### `Magnipy`: For in-depth magnitude computations on a single metric space.

The functionalities of `Magnipy` for an individual metric space include:  
- Computing the metric space's **distance matrix**
- Calculating the **similarity matrix** from the distances
- Executing an **automated scale-finding** procedure to find suitable evaluation scales
- Computing the **magnitude weight** of each point across multiple distance scales
- Evaluating and plotting **magnitude functions** across varying distance scales
- Estimating magnitude dimension profiles and calculating the **magnitude dimension** to quantify **intrinsic dimensionality**

### `Diversipy`: For comparing magnitude (and thus diversity) across multiple datasets.

The functionalities of `Diversipy` for a list of spaces (that share the same distance metric) include: 
- Executing an **automated scale-finding** procedure and the determining a **common evaluation interval across datasets**
- Computing **magnitude functions** across varying distance scales
- Calculating **MagArea**, the area under a magnitude function, a multi-scale measure of the **intrinsic diversity** of a dataset
- Calculating **MagDiff**, the area between magnitude functions, to measure the **difference in diversity** between datasets

## ⚙️ Dependencies

To get started,
1. From the root repository

Our dependencies are managed using the [`poetry`](https://python-poetry.org) package manager. Using your activated virtual environment, run the following to install `poetry`:

```python
$ pip install poetry
```

With `poetry` installed, run the following command from the main directory to download the necessary dependencies:

```python
$ poetry install
```