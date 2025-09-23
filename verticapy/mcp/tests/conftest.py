"""
Pytest Configuration and Fixtures for MCP Tests

This module provides pytest fixtures for the MCP test suite,
following VerticaPy's established testing patterns.

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

import pytest
import random
import string
import sys
import os

# Add the parent directory to Python path to access server2.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import verticapy
from verticapy import drop
from verticapy.datasets import (
    load_titanic,
    load_iris,
    load_amazon,
    load_winequality,
    load_market,
    load_smart_meters,
    load_laliga,
    load_airline_passengers,
    load_pop_growth,
    load_gapminder,
    load_cities,
    load_world,
)

# Import MCP server functions and connection class
from server2 import (
    connect_to_vertica,
    disconnect_from_vertica,
    connection_manager,
    clear_vdf_cache
)
from connection import VerticaPyConnection

# Import config from the local tests directory
import sys
import os
tests_dir = os.path.dirname(os.path.abspath(__file__))
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

import config
from config import load_test_config, get_conn_info


@pytest.fixture(name="schema_loader", scope="session", autouse=True)
def load_test_schema():
    """
    Create a schema with a random name for test
    """
    alphabet = string.ascii_letters
    random_string = "".join(random.choice(alphabet) for i in range(4))
    schema_name = f"test_{random_string}"
    verticapy.create_schema(schema_name)
    yield schema_name
    verticapy.drop(schema_name, method="schema")


@pytest.fixture(scope="session", autouse=True)
def mcp_connection():
    """
    Establish MCP connection for the entire test session
    """
    # Load connection configuration
    config = load_test_config()
    conn_info = get_conn_info()
    
    print(f"\n🔌 Establishing MCP connection...")
    print(f"   Host: {conn_info['host']}")
    print(f"   Port: {conn_info['port']}")
    print(f"   Database: {conn_info['database']}")
    print(f"   User: {conn_info['user']}")
    
    # Connect to database
    result = connect_to_vertica()
    if not result.get("success"):
        pytest.fail(f"Failed to establish database connection: {result.get('error')}")
    
    print(f"✅ MCP connection established successfully")
    
    yield conn_info
    
    # Cleanup after all tests
    print(f"\n🧹 Cleaning up MCP connection...")
    try:
        clear_vdf_cache()
        disconnect_from_vertica()
        print(f"✅ MCP connection cleaned up successfully")
    except Exception as e:
        print(f"⚠️  Warning during MCP cleanup: {str(e)}")


@pytest.fixture(scope="session")
def test_config():
    """
    Provide test configuration for all tests
    """
    return load_test_config()


@pytest.fixture(scope="module")
def titanic_vd(schema_loader):
    """
    Create a vDataFrame for titanic dataset in test schema
    """
    titanic = load_titanic(schema_loader, "titanic")
    yield titanic
    drop(name=f"{schema_loader}.titanic")


@pytest.fixture(scope="module")
def iris_vd(schema_loader):
    """
    Create a vDataFrame for iris dataset in test schema
    """
    iris = load_iris(schema_loader, "iris")
    yield iris
    drop(name=f"{schema_loader}.iris")


@pytest.fixture(scope="module")
def amazon_vd(schema_loader):
    """
    Create a vDataFrame for amazon dataset in test schema
    """
    amazon = load_amazon(schema_loader, "amazon")
    yield amazon
    drop(name=f"{schema_loader}.amazon")


@pytest.fixture(scope="module")
def winequality_vd(schema_loader):
    """
    Create a vDataFrame for winequality dataset in test schema
    """
    winequality = load_winequality(schema_loader, "winequality")
    yield winequality
    drop(name=f"{schema_loader}.winequality")


@pytest.fixture(scope="module")
def market_vd(schema_loader):
    """
    Create a vDataFrame for market dataset in test schema
    """
    market = load_market(schema_loader, "market")
    yield market
    drop(name=f"{schema_loader}.market")


@pytest.fixture(scope="module")
def smart_meters_vd(schema_loader):
    """
    Create a vDataFrame for smart_meters dataset in test schema
    """
    smart_meters = load_smart_meters(schema_loader, "smart_meters")
    yield smart_meters
    drop(name=f"{schema_loader}.smart_meters")


@pytest.fixture(scope="module")
def laliga_vd(schema_loader):
    """
    Create a vDataFrame for laliga dataset in test schema
    """
    laliga = load_laliga(schema_loader, "laliga")
    yield laliga
    drop(name=f"{schema_loader}.laliga")


@pytest.fixture(scope="module")
def airline_vd(schema_loader):
    """
    Create a vDataFrame for airline_passengers dataset in test schema
    """
    airline = load_airline_passengers(schema_loader, "airline")
    yield airline
    drop(name=f"{schema_loader}.airline")


@pytest.fixture(scope="module")
def pop_growth_vd(schema_loader):
    """
    Create a vDataFrame for pop_growth dataset in test schema
    """
    pop_growth = load_pop_growth(schema_loader, "pop_growth")
    yield pop_growth
    drop(name=f"{schema_loader}.pop_growth")


@pytest.fixture(scope="module")
def gapminder_vd(schema_loader):
    """
    Create a vDataFrame for gapminder dataset in test schema
    """
    gapminder = load_gapminder(schema_loader, "gapminder")
    yield gapminder
    drop(name=f"{schema_loader}.gapminder")


@pytest.fixture(scope="module")
def cities_vd(schema_loader):
    """
    Create a vDataFrame for cities dataset in test schema
    """
    cities = load_cities(schema_loader, "cities")
    yield cities
    drop(name=f"{schema_loader}.cities")


@pytest.fixture(scope="module")
def world_vd(schema_loader):
    """
    Create a vDataFrame for world dataset in test schema
    """
    world = load_world(schema_loader, "world")
    yield world
    drop(name=f"{schema_loader}.world")


@pytest.fixture(scope="function")
def clean_vdf_cache():
    """
    Clean vDataFrame cache before and after each test function
    """
    # Clean before test
    clear_vdf_cache()
    yield
    # Clean after test
    clear_vdf_cache()


@pytest.fixture(scope="function") 
def titanic_vd_fun(schema_loader):
    """
    Create a vDataFrame for titanic dataset (function scope)
    """
    titanic = load_titanic(schema_loader, "titanic_fun")
    yield titanic
    drop(name=f"{schema_loader}.titanic_fun")


@pytest.fixture(scope="function")
def iris_vd_fun(schema_loader):
    """
    Create a vDataFrame for iris dataset (function scope)
    """
    iris = load_iris(schema_loader, "iris_fun")
    yield iris
    drop(name=f"{schema_loader}.iris_fun")


# Test data information fixtures
@pytest.fixture(scope="session")
def test_datasets_info():
    """
    Provide information about available test datasets
    """
    return {
        "titanic": {
            "numeric_columns": ["age", "fare", "survived", "pclass"],
            "categorical_columns": ["sex", "embarked"],
            "text_columns": ["name"],
            "expected_row_count_range": (800, 1000),  # Approximate range
        },
        "iris": {
            "numeric_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
            "categorical_columns": ["species"],
            "expected_row_count_range": (140, 160),
        },
        "winequality": {
            "numeric_columns": ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "quality"],
            "categorical_columns": [],
            "expected_row_count_range": (6000, 7000),
        },
    }


@pytest.fixture(scope="session")
def sample_test_operations():
    """
    Provide sample operations for testing transformations
    """
    return {
        "search_operations": [
            {"conditions": "survived = 1", "description": "Filter survivors"},
            {"conditions": "age > 30", "description": "Filter adults"},
            {"conditions": "pclass = 1", "description": "Filter first class"},
        ],
        "groupby_operations": [
            {
                "columns": ["sex"], 
                "expr": ["count(*) AS passenger_count", "avg(age) AS avg_age"],
                "description": "Group by gender"
            },
            {
                "columns": ["pclass"], 
                "expr": ["count(*) AS count", "avg(fare) AS avg_fare"],
                "description": "Group by class"
            },
        ],
        "statistical_metrics": [
            "describe", "mean", "max", "min", "count", "sum", "nunique"
        ],
        "parameterized_metrics": [
            {"metric": "nlargest", "params": {"n": 5}},
            {"metric": "nsmallest", "params": {"n": 3}},
            {"metric": "topk", "params": {"k": 4}},
        ]
    }