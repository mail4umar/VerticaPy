# Pytest
import pytest

# Standard Python Modules


# Other Modules
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# VerticaPy
from vertica_highcharts.highcharts.highcharts import Highchart


def get_xaxis_label(obj):
    if isinstance(obj, plt.Axes):
        return obj.get_xlabel()
    elif isinstance(obj, go.Figure):
        return obj.layout["xaxis"]["title"]["text"]
    elif isinstance(obj, Highchart):
        return obj.options["xAxis"].title.text
    else:
        return None


def get_yaxis_label(obj):
    if isinstance(obj, plt.Axes):
        return obj.get_ylabel()
    elif isinstance(obj, go.Figure):
        return obj.layout["yaxis"]["title"]["text"]
    elif isinstance(obj, Highchart):
        return obj.options["yAxis"].title.text
    else:
        return None


def get_zaxis_label(obj):
    if isinstance(obj, plt.Axes):
        return obj.get_zlabel()
    elif isinstance(obj, go.Figure):
        return obj.layout["zaxis"]["title"]["text"]
    elif isinstance(obj, Highchart):
        return obj.options["zAxis"].title.text
    else:
        return None


def get_width(obj):
    if isinstance(obj, plt.Axes):
        return obj.get_figure().get_size_inches()[0]
    elif isinstance(obj, go.Figure):
        return obj.layout["width"]
    elif isinstance(obj, Highchart):
        return obj.options["chart"].width
    else:
        return None


def get_height(obj):
    if isinstance(obj, plt.Axes):
        return obj.get_figure().get_size_inches()[1]
    elif isinstance(obj, go.Figure):
        return obj.layout["height"]
    elif isinstance(obj, Highchart):
        return obj.options["chart"].height
    else:
        return None


def get_title(obj):
    if isinstance(obj, plt.Axes):
        return obj.get_title()
    elif isinstance(obj, go.Figure):
        return obj.layout["title"]["text"]
    elif isinstance(obj, Highchart):
        return obj.options["title"].text
    else:
        return None
