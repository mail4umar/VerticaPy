"""
(c)  Copyright  [2018-2023]  OpenText  or one of its
affiliates.  Licensed  under  the   Apache  License,
Version 2.0 (the  "License"); You  may  not use this
file except in compliance with the License.

You may obtain a copy of the License at:
http://www.apache.org/licenses/LICENSE-2.0

Unless  required  by applicable  law or  agreed to in
writing, software  distributed  under the  License is
distributed on an  "AS IS" BASIS,  WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.
See the  License for the specific  language governing
permissions and limitations under the License.
"""
# MATPLOTLIB
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# NUMPY
import numpy as np

# VerticaPy Modules
#from verticapy.plotting._plotly.base import updated_dict
from verticapy._config.config import ISNOTEBOOK
from verticapy.plotting._plotly.base import (
    compute_plot_variables, updated_dict, get_sunburst_attributes
)
import plotly.express as px
import plotly.graph_objects as go
from verticapy._config.colors import get_colors

def pie(
    vdf,
    method: str = "density",
    of=None,
    max_cardinality: int = 6,
    h: float = 0,
    donut: bool = False,
    rose: bool = False,
    ax = None,
    exploded: bool = True,
    **style_kwds,
):
    colors = get_colors()
    x, y, z, h, is_categorical = compute_plot_variables(
                                vdf, max_cardinality=max_cardinality, method=method, of=of, pie=True
                                )
    z.reverse()
    y.reverse()
    z=['None' if elm is None else elm for elm in z]

    #if not(rose):
    if donut:
        hole_fraction=0.2
    else:
        hole_fraction=0
    
    if exploded:
        exploded_parameters=[0] * len(y)
        exploded_parameters[y.index(max(y))]=0.2
    
    else:
        exploded_parameters=[]

    param = {
            "hole":hole_fraction,
            "pull":exploded_parameters,
            "marker_colors": colors,
        }
    
    labels, values = zip(*sorted(zip(z, y), key=lambda t: t[0]))
    fig = go.Figure(
        data=[go.Pie(
            labels=labels, 
            values=values,
            **updated_dict(param, style_kwds),
            sort=False,
            )])

    
    fig.update_layout(
        title_text=vdf._alias,
        title_x=0.5,
        title_xanchor="center")
    return fig
    # else: 
    #     try:
    #         y, z = zip(*sorted(zip(y, z), key=lambda t: t[0]))
            
    #     except:
    #         pass

    #     param = {
    #         "color": colors,
    #     }
    #     colors = updated_dict(param, style_kwds, -1)["color"]
    #     if isinstance(colors, str):
    #         colors = [colors] + gen_colors()
    #     else:
    #         colors = colors + gen_colors()
    #     style_kwds["color"] = colors
    #     N = len(z)
    #     colors = sorted(zip(z, colors[:N]), key=lambda t: t[0])
        
    #     width = 2 * np.pi / N
    #     rad = np.cumsum([width] * N)
        

    #     fig = px.bar_polar(
    #                r=z,
    #                theta=rad,
    #                color=colors,
    #                color_discrete_sequence=['purple'],
    #               )

    #     fig.update_layout(
    #         template=None,
    #         polar = dict(
    #             radialaxis = dict(range=[0, 5], showticklabels=False, ticks='',),
    #             angularaxis = dict(showticklabels=False, ticks='', linecolor='White'),
    #         ),
    #         polar_radialaxis_gridcolor="#ffffff",
    #         polar_angularaxis_gridcolor="#ffffff",
    #         polar_angularaxis_tickvals=z,
    #         #height=500,
    #         title=vdf.alias
    #     )
    #     fig.update_polars(radialaxis_showline=False, bgcolor='white')
    #     return fig



def nested_pie(
    vdf,
    columns: list,
    max_cardinality: tuple = None,
    h: tuple = None,
    ax=None,
    **style_kwds,
):
    ids,labels,parents,values=get_sunburst_attributes(vdf,columns,max_cardinality)
    trace = go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        outsidetextfont = {"size": 20, "color": "#377eb8"},
        marker = {"line": {"width": 2}},
    )

    layout = go.Layout(
        margin = go.layout.Margin(t=0, l=0, r=0, b=0)
    )

    figure = {
        'data': [trace],
        'layout': layout
    }

    fig = go.Figure(figure)
    return fig