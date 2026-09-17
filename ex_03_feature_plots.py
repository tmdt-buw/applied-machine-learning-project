import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from matplotlib import pyplot as plt
    import seaborn as sns

    from ex_01_read_data import get_welding_data
    from ex_03_feature_extraction import extract_features

    return Path, extract_features, get_welding_data, mo


@app.cell
def _(Path):
    data_path = Path("data/Welding/data.csv")
    plot_path = Path("plots/ex_03")
    plot_path.mkdir(parents=True, exist_ok=True)
    return (data_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 3.3: Feature Exploration with Pairplots

    In this exercise, you will create a seaborn pairplot to visualize relationships between the features extracted from the welding data.
    """)
    return


@app.cell
def _(data_path, extract_features, get_welding_data):
    n_samples = 1_000
    data, labels, _ = get_welding_data(data_path, n_samples=n_samples)
    features = extract_features(data, labels)
    features.head()
    return


@app.cell
def _():
    # Exercise 3.3: create and save the seaborn pairplot here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 3.4: Describe Your Feature Selection

    Based on the pairplot visualization from Exercise 3.1, provide a detailed analysis of your feature selection:

    1. Which features appear most effective at separating the quality classes?
    2. Are there features that show strong correlation with each other? Would it make sense to remove any redundant features?
    3. Explain the physical significance of your selected features in the context of welding quality.
    4. If you were to select a subset of 5 most important features for classifying welding quality, which would you choose and why?
    5. Discuss any patterns or insights you discovered in the data visualization that might help with building a classification model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    _Write your feature selection analysis here._
    """)
    return


if __name__ == "__main__":
    app.run()
