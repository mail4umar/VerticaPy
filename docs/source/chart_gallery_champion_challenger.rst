.. _chart_gallery.champion_challenger:

========================
Champion Challenger Plot
========================

.. Necessary Code Elements

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy as vp
    import verticapy.machine_learning.vertica.automl as vmla
    import numpy as np

    N = 500 # Number of Records
    k = 10 # step

    # Normal Distributions
    x = np.random.normal(5, 1, round(N / 2))
    y = np.random.normal(3, 1, round(N / 2))
    z = np.random.normal(3, 1, round(N / 2))

    # Creating a vDataFrame with two clusters
    data = vp.vDataFrame({
        "x": np.concatenate([x, x + k]),
        "y": np.concatenate([y, y + k]),
        "z": np.concatenate([z, z + k]),
        "c": [0 for i in range(round(N / 2))] + [1 for i in range(round(N / 2))]
    })

    # Defining the Model
    vp.drop("automl_demo")
    model = vmla.AutoML("automl_demo")

    # Fitting the Model
    model.fit(data, X = ["x", "y", "z"], y = "c")

General
-------

VerticaPy's AutoML (Automated Machine Learning) feature is a robust and user-friendly tool that simplifies the complex process of machine learning model development. It automates critical tasks such as algorithm selection, hyperparameter tuning, feature engineering, and cross-validation, significantly reducing the time and effort required to build high-performing predictive models. With its scalability, transparency, and ease of use, VerticaPy's AutoML empowers data analysts and scientists to efficiently harness the potential of machine learning, even in large-scale and complex data environments, enabling organizations to make data-driven decisions with confidence and precision.

Let's begin by importing ``verticapy``.

.. ipython:: python

    import verticapy as vp

Let's generate a dataset using the following data.

.. code-block:: python
        
    N = 500 # Number of Records
    k = 10 # step

    # Normal Distributions
    x = np.random.normal(5, 1, round(N / 2))
    y = np.random.normal(3, 1, round(N / 2))
    z = np.random.normal(3, 1, round(N / 2))

    # Creating a vDataFrame with two clusters
    data = vp.vDataFrame({
        "x": np.concatenate([x, x + k]),
        "y": np.concatenate([y, y + k]),
        "z": np.concatenate([z, z + k]),
        "c": [0 for i in range(round(N / 2))] + [1 for i in range(round(N / 2))]
    })

Let's proceed by creating an AutoML model using the complete dataset.

.. code-block:: python
    
    # Importing the Vertica Auto ML module
    import verticapy.machine_learning.vertica.automl as vmla

    # Defining the Model
    model = vmla.AutoML("automl_demo")

    # Fitting the Model
    model.fit(data, X = ["x", "y", "z"], y = "c")

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

    We can effortlessly create the champion-challenger bubble plot.

    .. code-block:: python
        
        model.plot()

    .. ipython:: python
        :suppress:
        :okwarning:
      
        fig = model.plot(width=650)
        fig.write_html("figures/plotting_plotly_champion.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_plotly_champion.html

.. tab:: Highcharts

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "highcharts")

    We can switch to using the ``highcharts`` module.

    .. code-block:: python
        
        vp.set_option("plotting_lib", "highcharts")

    We can effortlessly create the champion-challenger bubble plot.

    .. code-block:: python
        
        model.plot()

    .. ipython:: python
        :suppress:
        :okwarning:

        fig = model.plot()
        html_text = fig.htmlcontent.replace("container", "plotting_highcharts_champion")
        with open("figures/plotting_highcharts_champion.html", "w") as file:
          file.write(html_text)

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_highcharts_champion.html
        
.. tab:: Matplotlib

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "matplotlib")

    We can switch to using the ``matplotlib`` module.

    .. code-block:: python
        
        vp.set_option("plotting_lib", "matplotlib")

    We can effortlessly create the champion-challenger bubble plot.

    .. ipython:: python
        :okwarning:

        @savefig plotting_matplotlib_champion.png
        model.plot()

.. hint:: VerticaPy AutoML offers a wide range of parameters for optimizing computations and data preprocessing. For detailed information, please refer to the :ref:`AutoML`documentation.

___________________


Chart Customization
-------------------

VerticaPy empowers users with a high degree of flexibility when it comes to tailoring the visual aspects of their plots. 
This customization extends to essential elements such as **color schemes**, **text labels**, and **plot sizes**, as well as a wide range of other attributes that can be fine-tuned to align with specific design preferences and analytical requirements. Whether you want to make your visualizations more visually appealing or need to convey specific insights with precision, VerticaPy's customization options enable you to craft graphics that suit your exact needs.

.. note:: As champion challenger plots are essentially scatter and bubble plots, customization options are identical to those available for :ref:`scatter`.