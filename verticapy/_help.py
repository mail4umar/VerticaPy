"""
Copyright  (c)  2018-2025 Open Text  or  one  of its
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
import verticapy._config.config as conf
from verticapy._utils._logo import verticapy_logo_html, verticapy_logo_str

if conf.get_import_success("IPython"):
    from IPython.display import display, Markdown


def help_start() -> None:
    """
    VERTICAPY Interactive Help (FAQ).
    """
    img1 = verticapy_logo_html(size="10%")
    img2 = verticapy_logo_str()
    if conf.get_import_success("IPython"):
        message = img1
    else:
        message = img2
    message += (
        "\n\n&#128226; Welcome to the <b>VerticaPy</b> help module."
        "\n\nThis module can help you connect to Vertica, "
        "create a Virtual DataFrame, load your data, and more.\n "
        "- <b>[Enter  0]</b> Overview of the library\n "
        "- <b>[Enter  1]</b> Load an example dataset\n "
        "- <b>[Enter  2]</b> View an example of data analysis with VerticaPy\n "
        "- <b>[Enter  3]</b> Contribute on GitHub\n "
        "- <b>[Enter  4]</b> View the SQL code generated by a vDataFrame and "
        "the time elapsed for the query\n "
        "- <b>[Enter  5]</b> Load your own dataset into Vertica \n "
        "- <b>[Enter  6]</b> Write SQL queries in Jupyter\n "
        "- <b>[Enter -1]</b> Exit"
    )
    if not conf.get_import_success("IPython"):
        message = message.replace("<b>", "").replace("</b>", "")
    if conf.get_import_success("IPython"):
        display(Markdown(message))
    else:
        print(message)
    try:
        response = int(input())
    except TypeError:
        print("Invalid choice.\nPlease enter a number between 0 and 11.")
        try:
            response = int(input())
        except TypeError:
            print("Invalid choice.\nRerun the help_start function when you need help.")
            return
    link = None
    if response == 0:
        link = "https://www.vertica.com/python/documentation/1.1.x/html/getting_started.html"
    elif response == 1:
        link = "https://www.vertica.com/python/documentation/1.1.x/html/datasets.html"
    elif response == 2:
        link = "https://www.vertica.com/python/documentation/1.1.x/html/examples.html"
    elif response == 3:
        link = "https://github.com/vertica/VerticaPy/"
    elif response == 4:
        link = "https://www.vertica.com/python/documentation/1.1.x/html/api/verticapy.set_option.html"
    elif response == 5:
        link = "https://www.vertica.com/python/documentation/1.1.x/html/datasets.html"
    elif response == 6:
        link = "https://www.vertica.com/python/documentation/1.1.x/html/api/verticapy.jupyter.extensions.sql_magic.sql_magic.html"
    elif response == -1:
        message = "Thank you for using the VerticaPy help module."
    elif response == 666:
        message = (
            "Thank you so much for using this library. My only purpose is to solve "
            "real Big Data problems in the context of Data Science. I worked years "
            "to be able to create this API and give you a real way to analyze your "
            "data.\n\nYour devoted Data Scientist: <i>Badr Ouali</i>"
        )
    else:
        message = "Invalid choice.\nPlease enter a number between -1 and 6."
    if 0 <= response <= 6:
        if not conf.get_import_success("IPython"):
            message = f"Please go to {link}"
        else:
            message = f"Please go to <a href='{link}'>{link}</a>"
    if conf.get_import_success("IPython"):
        display(Markdown(message))
    else:
        print(message)
