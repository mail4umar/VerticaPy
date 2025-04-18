.. _chart_gallery.regression_plot:

===================================
Machine Learning - Regression Plots
===================================

.. Necessary Code Elements

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy as vp
    import verticapy.machine_learning.vertica as vml
    import numpy as np

    N = 100 # Number of Records

    x = np.random.normal(5, 1, N) # Normal Distribution
    e = np.random.random(N) # Noise

    data = vp.vDataFrame({
      "x": x,
      "y": x + e,
    })

    # Defining the Models
    model_lr = vml.LinearRegression()
    model_rf = vml.RandomForestRegressor()

    # Fitting the models
    model_lr.fit(data, "x", "y")
    model_rf.fit(data, "x", "y")

    # Adding the predictions to the vDataFrame
    model_lr.predict(data, "x", name = "x_lr", inplace = True)
    model_rf.predict(data, "x", name = "x_rf", inplace = True)

    # Computing the respective noises
    data["noise_lr"] = data["x"] - data["x_lr"]
    data["noise_rf"] = data["x"] - data["x_rf"]

    # Displaying the vDataFrame
    display(data)


General
-------

In this example, we aim to present several regression plots, including linear regression, tree-based algorithms, and various residual plots. It's important to note that these plots are purely illustrative and are based on generated data. To make the data more realistically representative, we introduce some noise, resulting in an approximately linear relationship.

Let's begin by importing ``verticapy``.

.. ipython:: python

    import verticapy as vp

Let's also import ``numpy`` to create a random dataset.

.. ipython:: python

    import numpy as np

Let's generate a dataset using the following data.

.. code-block:: python
        
    N = 100 # Number of Records

    x = np.random.normal(5, 1, N) # Normal Distribution
    e = np.random.random(N) # Noise

    data = vp.vDataFrame({
      "x": x,
      "y": x + e,
    })

Let's proceed by creating both a linear regression model and a random forest regressor model using the complete dataset. Following that, we can calculate the respective noise associated with each model.

.. code-block:: python
    
    # Importing the Vertica ML module
    import verticapy.machine_learning.vertica as vml

    # Defining the Models
    model_lr = vml.LinearRegression()
    model_rf = vml.RandomForestRegressor()

    # Fitting the models
    model_lr.fit(data, "x", "y")
    model_rf.fit(data, "x", "y")

    # Adding the predictions to the vDataFrame
    model_lr.predict(data, "x", name = "x_lr", inplace = True)
    model_rf.predict(data, "x", name = "x_rf", inplace = True)

    # Computing the respective noises
    data["noise_lr"] = data["x"] - data["x_lr"]
    data["noise_rf"] = data["x"] - data["x_rf"]

    # Displaying the vDataFrame
    display(data)

In the context of data visualization, we have the flexibility to harness multiple plotting libraries to craft a wide range of graphical representations. VerticaPy, as a versatile tool, provides support for several graphic libraries, such as Matplotlib, Highcharts, and Plotly. Each of these libraries offers unique features and capabilities, allowing us to choose the most suitable one for our specific data visualization needs.

.. image:: ../../docs/source/_static/plotting_libs.png
   :width: 80%
   :align: center

.. note::
    
    To select the desired plotting library, we simply need to use the :py:func:`~verticapy.set_option` function. VerticaPy offers the flexibility to smoothly transition between different plotting libraries. In instances where a particular graphic is not supported by the chosen library or is not supported within the VerticaPy framework, the tool will automatically generate a warning and then switch to an alternative library where the graphic can be created.

Please click on the tabs to view the various graphics generated by the different plotting libraries.

.. ipython:: python
    :suppress:

    import verticapy as vp

.. tab:: Plotly

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")

    We can switch to using the ``plotly`` module.

    .. code-block:: python
        
        vp.set_option("plotting_lib", "plotly")
    
    .. tab:: LR

      .. code-block:: python
          
          model_lr.plot()

      .. ipython:: python
          :suppress:
        
          fig = model_lr.plot()
          fig.write_html("figures/plotting_plotly_lr_1.html")

      .. raw:: html
          :file: SPHINX_DIRECTORY/figures/plotting_plotly_lr_1.html

      **Residual Plot**

      .. code-block:: python
          
          data.scatter(["y", "noise_lr"])

      .. ipython:: python
          :suppress:
        
          fig = data.scatter(["y", "noise_lr"])
          fig.write_html("figures/plotting_plotly_lr_2.html")

      .. raw:: html
          :file: SPHINX_DIRECTORY/figures/plotting_plotly_lr_2.html

    .. tab:: RF

      .. code-block:: python
          
          model_rf.plot()

      .. ipython:: python
          :suppress:
        
          fig = model_rf.plot()
          fig.write_html("figures/plotting_plotly_rf_1.html")

      .. raw:: html
          :file: SPHINX_DIRECTORY/figures/plotting_plotly_rf_1.html

      **Residual Plot**

      .. code-block:: python
          
          data.scatter(["y", "noise_rf"])

      .. ipython:: python
          :suppress:
        
          fig = data.scatter(["y", "noise_rf"])
          fig.write_html("figures/plotting_plotly_rf_2.html")

      .. raw:: html
          :file: SPHINX_DIRECTORY/figures/plotting_plotly_rf_2.html

.. tab:: Highcharts

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "highcharts")

    We can switch to using the ``highcharts`` module.

    .. code-block:: python
        
        vp.set_option("plotting_lib", "highcharts")

    .. tab:: LR

      .. code-block:: python
          
          model_lr.plot()

      .. ipython:: python
          :suppress:

          fig = model_lr.plot()
          html_text = fig.htmlcontent.replace("container", "plotting_highcharts_lr_1")
          with open("figures/plotting_highcharts_lr_1.html", "w") as file:
            file.write(html_text)

      .. raw:: html
          :file: SPHINX_DIRECTORY/figures/plotting_highcharts_lr_1.html

      **Residual Plot**

      .. code-block:: python
          
          data.scatter(["y", "noise_lr"])

      .. ipython:: python
          :suppress:

          fig = data.scatter(["y", "noise_lr"])
          html_text = fig.htmlcontent.replace("container", "plotting_highcharts_lr_2")
          with open("figures/plotting_highcharts_lr_2.html", "w") as file:
            file.write(html_text)

      .. raw:: html
          :file: SPHINX_DIRECTORY/figures/plotting_highcharts_lr_2.html

    .. tab:: RF

      .. code-block:: python
          
          model_rf.plot()

      .. ipython:: python
          :suppress:

          fig = model_rf.plot()
          html_text = fig.htmlcontent.replace("container", "plotting_highcharts_rf_1")
          with open("figures/plotting_highcharts_rf_1.html", "w") as file:
            file.write(html_text)

      .. raw:: html
          :file: SPHINX_DIRECTORY/figures/plotting_highcharts_rf_1.html

      **Residual Plot**

      .. code-block:: python
          
          data.scatter(["y", "noise_rf"])

      .. ipython:: python
          :suppress:

          fig = data.scatter(["y", "noise_rf"])
          html_text = fig.htmlcontent.replace("container", "plotting_highcharts_rf_2")
          with open("figures/plotting_highcharts_rf_2.html", "w") as file:
            file.write(html_text)

      .. raw:: html
          :file: SPHINX_DIRECTORY/figures/plotting_highcharts_rf_2.html
        
.. tab:: Matplotlib

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "matplotlib")

    We can switch to using the ``matplotlib`` module.

    .. code-block:: python
        
        vp.set_option("plotting_lib", "matplotlib")

    .. tab:: LR

      .. ipython:: python
          :okwarning:

          @savefig plotting_matplotlib_lr_1.png
          model_lr.plot()

      **Residual Plot**

      .. ipython:: python
          :okwarning:

          @savefig plotting_matplotlib_lr_2.png
          data.scatter(["y", "noise_lr"])

    .. tab:: RF

      .. ipython:: python
          :okwarning:

          @savefig plotting_matplotlib_rf_1.png
          model_rf.plot()

      **Residual Plot**

      .. ipython:: python
          :okwarning:

          @savefig plotting_matplotlib_rf_2.png
          data.scatter(["y", "noise_rf"])

___________________


Chart Customization
-------------------

VerticaPy empowers users with a high degree of flexibility when it comes to tailoring the visual aspects of their plots. 
This customization extends to essential elements such as **color schemes**, **text labels**, and **plot sizes**, as well as a wide range of other attributes that can be fine-tuned to align with specific design preferences and analytical requirements. Whether you want to make your visualizations more visually appealing or need to convey specific insights with precision, VerticaPy's customization options enable you to craft graphics that suit your exact needs.

.. Important:: Different customization parameters are available for Plotly, Highcharts, and Matplotlib. 
    For a comprehensive list of customization features, please consult the documentation of the respective 
    libraries: `plotly <https://plotly.com/python-api-reference/>`_, `matplotlib <https://matplotlib.org/stable/api/matplotlib_configuration_api.html>`_ and `highcharts <https://api.highcharts.com/highcharts/>`_.

Colors
~~~~~~

.. tab:: Plotly

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")

    **Custom colors**

    .. code-block:: python
        
        fig = model_lr.plot()
        fig.update_traces(marker = dict(color="red"))

    .. ipython:: python
        :suppress:

        fig = model_lr.plot()
        fig.update_traces(marker = dict(color="red"))
        fig.write_html("figures/plotting_plotly_lr_plot_custom_color_1.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_plotly_lr_plot_custom_color_1.html

.. tab:: Highcharts

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "highcharts")

    **Custom colors**

    .. code-block:: python
        
        model_lr.plot(colors = "red")

    .. ipython:: python
        :suppress:

        fig = model_lr.plot(colors = "red")
        html_text = fig.htmlcontent.replace("container", "plotting_highcharts_lr_plot_custom_color_1")
        with open("figures/plotting_highcharts_lr_plot_custom_color_1.html", "w") as file:
            file.write(html_text)

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_highcharts_lr_plot_custom_color_1.html

.. tab:: Matplolib

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "matplotlib")

    **Custom colors**

    .. ipython:: python

        @savefig plotting_matplotlib_lr_plot_custom_color_1.png
        model_lr.plot(colors = "red")

____

Size
~~~~

.. tab:: Plotly

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")

    **Custom Width and Height**

    .. code-block:: python
        
        model_lr.plot(width = 300, height = 300)

    .. ipython:: python
        :suppress:

        fig = model_lr.plot(width = 300, height = 300)
        fig.write_html("figures/plotting_plotly_lr_plot_custom_size.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_plotly_lr_plot_custom_size.html

.. tab:: Highcharts

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "highcharts")

    **Custom Width and Height**

    .. code-block:: python
        
        model_lr.plot(width = 500, height = 200)

    .. ipython:: python
        :suppress:

        fig = model_lr.plot(width = 500, height = 200)
        html_text = fig.htmlcontent.replace("container", "plotting_highcharts_lr_plot_custom_size")
        with open("figures/plotting_highcharts_lr_plot_custom_size.html", "w") as file:
            file.write(html_text)

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_highcharts_lr_plot_custom_size.html

.. tab:: Matplolib

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "matplotlib")

    **Custom Width and Height**

    .. ipython:: python
        :okwarning:

        @savefig plotting_matplotlib_lr_plot_single_custom_size.png
        model_lr.plot(width = 6, height = 3)

_____


Text
~~~~

.. tab:: Plotly

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")

    **Custom Title**

    .. code-block:: python
        
        model_lr.plot().update_layout(title_text = "Custom Title")

    .. ipython:: python
        :suppress:

        fig = model_lr.plot().update_layout(title_text = "Custom Title")
        fig.write_html("figures/plotting_plotly_lr_plot_custom_main_title.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_plotly_lr_plot_custom_main_title.html


    **Custom Axis Titles**

    .. code-block:: python
        
        model_lr.plot(yaxis_title = "Custom Y-Axis Title")

    .. ipython:: python
        :suppress:

        fig = model_lr.plot(yaxis_title = "Custom Y-Axis Title")
        fig.write_html("figures/plotting_plotly_lr_plot_custom_y_title.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_plotly_lr_plot_custom_y_title.html

.. tab:: Highcharts

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "highcharts")

    **Custom Title Text**

    .. code-block:: python
        
        model_lr.plot(title = {"text": "Custom Title"})

    .. ipython:: python
        :suppress:

        fig = model_lr.plot(title = {"text": "Custom Title"})
        html_text = fig.htmlcontent.replace("container", "plotting_highcharts_lr_plot_custom_text_title")
        with open("figures/plotting_highcharts_lr_plot_custom_text_title.html", "w") as file:
            file.write(html_text)

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_highcharts_lr_plot_custom_text_title.html

    **Custom Axis Titles**

    .. code-block:: python
        
        model_lr.plot(xAxis = {"title": {"text": "Custom X-Axis Title"}})

    .. ipython:: python
        :suppress:

        fig = model_lr.plot(xAxis = {"title": {"text": "Custom X-Axis Title"}})
        html_text = fig.htmlcontent.replace("container", "plotting_highcharts_lr_plot_custom_text_xtitle")
        with open("figures/plotting_highcharts_lr_plot_custom_text_xtitle.html", "w") as file:
            file.write(html_text)

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_highcharts_lr_plot_custom_text_xtitle.html

.. tab:: Matplolib

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "matplotlib")

    **Custom Title Text**

    .. ipython:: python
        :okwarning:

        @savefig plotting_matplotlib_lr_plot_custom_title_label.png
        model_lr.plot().set_title("Custom Title")

    **Custom Axis Titles**

    .. ipython:: python
        :okwarning:

        @savefig plotting_matplotlib_lr_plot_custom_yaxis_label.png
        model_lr.plot().set_ylabel("Custom Y Axis")

_____

