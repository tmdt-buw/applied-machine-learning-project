from pathlib import Path
import plotly.graph_objects as go


def save_plotly_figure_as_html(fig: go.Figure, filename_path: str | Path):
    """
    Save a Plotly figure as an HTML file.
    
    Args:
        fig: The Plotly figure object to save
        filename_path: Name for the saved file (without extension)
    """
    if not isinstance(filename_path, Path):
        filename_path = Path(filename_path)

    if not filename_path.parent.exists():
        filename_path.parent.mkdir(parents=True)
        
    # Ensure filename has .html extension
    if not filename_path.suffix or filename_path.suffix != '.html':
        filename_path = filename_path.with_suffix('.html')
        
    # Save the figure
    fig.write_html(filename_path)

