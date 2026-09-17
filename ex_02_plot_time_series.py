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

    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from utils import save_plotly_figure_as_html
    from ex_01_read_data import get_welding_data

    return Path, get_welding_data, go, mo, np, save_plotly_figure_as_html


@app.cell
def _(Path):
    data_path = Path("data/Welding/data.csv")
    plot_path = Path("plots/ex_02")
    plot_path.mkdir(parents=True, exist_ok=True)
    return data_path, plot_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 2.1: Dataset Statistics Analysis

    In this exercise, we'll calculate and present comprehensive descriptive statistics for the welding dataset. This statistical overview will help us understand the fundamental characteristics of our data before visualization or modeling.

    1. Calculate the class distribution to understand data balance
    2. Count samples per experiment to assess data volume across experimental conditions
    3. Determine min/max values for voltage and current to understand the range of measurements
    4. Calculate mean and standard deviation to assess central tendency and variability
    5. Find median values to identify central points unaffected by outliers
    """)
    return


@app.cell
def _(data_path, get_welding_data):
    data, labels, exp_ids = get_welding_data(data_path)
    return


@app.cell
def _():
    # Exercise 2.1: compute and display the dataset statistics here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 2.2: Current and Voltage Distribution Visualization

    In this exercise, we'll create interactive boxplots to visualize and compare the distributions of voltage and current measurements in the welding data. Boxplots will help us identify central tendencies, spread, and potential outliers in our measurements.

    1. Create side-by-side boxplots for voltage and current using Plotly
    2. Display key statistics (median, quartiles, etc.) in a visual format in the plot
    3. Enable interactive exploration of the distributions
    4. Save the visualization for future reference
    """)
    return


@app.cell
def _(data_path, get_welding_data, plot_path):
    data_dist, labels_dist, exp_ids_dist = get_welding_data(data_path, n_samples=10_000)

    dist_save_path = plot_path / "voltage_current_distribution"
    return


@app.cell
def _():
    # Exercise 2.2: create and save the boxplots here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 2.3: Time-Series Sequence Visualization

    In this exercise, we'll implement functions to visualize the time-series patterns of welding cycles. These visualizations will help us understand the temporal relationships between voltage and current during the welding process and identify patterns associated with quality outcomes.

    1. Create dual-axis plots showing voltage and current over time (10 welding cycles -> sequence_length=10) using matplotlib
    2. Implement clear legends and labels for data interpretation
    3. Enable saving of visualizations for reporting and comparison
    """)
    return


@app.cell
def _(data_path, get_welding_data, plot_path):
    data_seq, labels_seq, exp_ids_seq = get_welding_data(
        data_path, n_samples=100, return_sequences=True, sequence_length=10
    )
    seq_save_path = plot_path / "welding_sample.png"
    return data_seq, exp_ids_seq, labels_seq


@app.function
def plot_random_sample(data, labels, save_path=None):
    """
    Plot a random welding sample with voltage and current on dual axes using matplotlib.

    Args:
        data (np.ndarray): Array containing voltage and current data
        labels (np.ndarray): Array containing class labels
        save_path (Path, optional): Where to save the figure. Defaults to None.
    """
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 2.4: Interactive Time-Series Visualization with Plotly

    In this exercise, we'll create enhanced interactive visualizations of welding samples using Plotly. These interactive plots will provide more advanced exploration capabilities for analyzing the time-series patterns.

    1. Create interactive plots with dual y-axes for voltage and current
    2. Implement time-based range sliders for detailed exploration
    3. Add unified tooltips for precise data reading
    4. Display quality metrics in the plot title
    5. Save interactive visualizations as HTML for sharing
    """)
    return


@app.cell
def _(go, np):
    def create_plotly_plot(
        data: np.ndarray, labels: np.ndarray, exp_ids: np.ndarray = None
    ) -> go.Figure:
        """
        Create an interactive Plotly visualization of a random welding sample.

        Args:
            data (np.ndarray): Array containing voltage and current data
            labels (np.ndarray): Array containing class labels
            exp_ids (np.ndarray, optional): Array containing experiment IDs. Defaults to None.

        Returns:
            plotly.graph_objects.Figure: Interactive Plotly figure object
        """
        pass

    return (create_plotly_plot,)


@app.cell
def _(
    create_plotly_plot,
    data_seq,
    exp_ids_seq,
    labels_seq,
    plot_path,
    save_plotly_figure_as_html,
):
    fig = create_plotly_plot(data_seq, labels_seq, exp_ids_seq)
    save_plotly_figure_as_html(fig, plot_path / "welding_samples")
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 2.5: Multiple Sample Comparison

    In this exercise, we'll generate and compare visualizations from multiple random welding samples. This comparison will help us identify common patterns and variations across different welding cycles.

    1. Generate multiple random sample visualizations using matplotlib of single welding cycles
    2. Create dual-axis plots showing voltage and current over time
    3. Implement clear legends and labels for data interpretation
    4. Save each visualization for comparison
    """)
    return


@app.cell
def _(data_path, get_welding_data):
    data_cycles, labels_cycles, exp_ids_cycles = get_welding_data(data_path, n_samples=1_000)
    return data_cycles, labels_cycles


@app.cell
def _(data_cycles, labels_cycles, plot_path):
    for i in range(5):
        plot_random_sample(
            data_cycles, labels_cycles, save_path=plot_path / f"welding_sample_{i}.png"
        )
    return


if __name__ == "__main__":
    app.run()
