"""
Machine learning tools for VerticaPy MCP server.
"""

from typing import Any, Dict, List
import logging
import verticapy as vp
from .base import VerticaPyMCPTool
from ..exceptions import MCPToolError, MCPSessionError
from ..utils import serialize_model_info

logger = logging.getLogger(__name__)


class TrainModelTool(VerticaPyMCPTool):
    """Tool for training machine learning models."""
    
    def __init__(self):
        super().__init__(
            name="train_model",
            description="Train a machine learning model using VerticaPy"
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {
                "model_type": {
                    "type": "string",
                    "enum": [
                        "LinearRegression", "LogisticRegression", 
                        "RandomForestRegressor", "RandomForestClassifier",
                        "XGBRegressor", "XGBClassifier",
                        "KMeans", "DBSCAN"
                    ],
                    "description": "Type of ML model to train"
                },
                "dataframe_name": {
                    "type": "string",
                    "description": "Name of the vDataFrame containing training data"
                },
                "predictors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of predictor column names"
                },
                "response": {
                    "type": "string",
                    "description": "Response/target column name (not required for unsupervised)"
                },
                "model_name": {
                    "type": "string",
                    "description": "Name to assign to the trained model"
                },
                "parameters": {
                    "type": "object",
                    "description": "Model-specific parameters (optional)"
                }
            },
            "required": ["model_type", "dataframe_name", "predictors", "model_name"]
        }
    
    async def execute(self, args: Dict[str, Any], session: Any) -> Dict[str, Any]:
        """Train ML model."""
        try:
            model_type = args["model_type"]
            dataframe_name = args["dataframe_name"]
            predictors = args["predictors"]
            response = args.get("response")
            model_name = args["model_name"]
            parameters = args.get("parameters", {})
            
            # Check session limits
            if len(session.models) >= session.config.max_models:
                raise MCPSessionError(f"Maximum models limit ({session.config.max_models}) reached")
            
            # Get vDataFrame from session
            if dataframe_name not in session.dataframes:
                raise MCPSessionError(f"vDataFrame '{dataframe_name}' not found in session")
            
            vdf = session.dataframes[dataframe_name]
            
            # Get model class
            model_classes = {
                "LinearRegression": vp.LinearRegression,
                "LogisticRegression": vp.LogisticRegression,
                "RandomForestRegressor": vp.RandomForestRegressor,
                "RandomForestClassifier": vp.RandomForestClassifier,
                "XGBRegressor": vp.XGBRegressor,
                "XGBClassifier": vp.XGBClassifier,
                "KMeans": vp.KMeans,
                "DBSCAN": vp.DBSCAN
            }
            
            if model_type not in model_classes:
                raise MCPToolError(f"Unsupported model type: {model_type}")
            
            ModelClass = model_classes[model_type]
            
            # Create and train model
            model = ModelClass(**parameters)
            
            # Train model (supervised vs unsupervised)
            if model_type in ["KMeans", "DBSCAN"]:
                # Unsupervised learning
                model.fit(vdf, predictors)
            else:
                # Supervised learning
                if not response:
                    raise MCPToolError(f"Response column required for {model_type}")
                model.fit(vdf, predictors, response)
            
            # Store model in session
            session.models[model_name] = model
            
            # Get model info
            model_info = serialize_model_info(model)
            
            return {
                "success": True,
                "model_name": model_name,
                "model_type": model_type,
                "predictors": predictors,
                "response": response,
                "model_info": model_info,
                "message": f"Model '{model_name}' trained successfully"
            }
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise MCPToolError(f"Failed to train model: {str(e)}")


class EvaluateModelTool(VerticaPyMCPTool):
    """Tool for evaluating trained models."""
    
    def __init__(self):
        super().__init__(
            name="evaluate_model",
            description="Evaluate a trained model using various metrics"
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema.""" 
        return {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Name of the trained model to evaluate"
                },
                "dataframe_name": {
                    "type": "string",
                    "description": "Name of test vDataFrame"
                },
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["accuracy", "auc", "aic", "bic", "r2", "rmse", "mae", "precision", "recall", "f1"]
                    },
                    "description": "Metrics to calculate"
                },
                "y_true": {
                    "type": "string",
                    "description": "True values column name"
                },
                "y_score": {
                    "type": "string",
                    "description": "Predicted values column name (optional, will use model predictions)"
                }
            },
            "required": ["model_name", "dataframe_name", "metrics"]
        }
    
    async def execute(self, args: Dict[str, Any], session: Any) -> Dict[str, Any]:
        """Evaluate model."""
        try:
            model_name = args["model_name"]
            dataframe_name = args["dataframe_name"]
            metrics = args["metrics"]
            y_true = args.get("y_true")
            y_score = args.get("y_score")
            
            # Get model and dataframe from session
            if model_name not in session.models:
                raise MCPSessionError(f"Model '{model_name}' not found in session")
            if dataframe_name not in session.dataframes:
                raise MCPSessionError(f"vDataFrame '{dataframe_name}' not found in session")
            
            model = session.models[model_name]
            vdf = session.dataframes[dataframe_name]
            
            # Calculate metrics
            results = {}
            
            for metric in metrics:
                try:
                    if metric == "accuracy":
                        score = vp.accuracy_score(y_true, y_score or model.predict(vdf), vdf)
                    elif metric == "auc":
                        score = vp.auc(y_true, y_score or model.predict_proba(vdf), vdf)
                    elif metric == "aic":
                        score = vp.aic_score(y_true, y_score or model.predict(vdf), vdf)
                    elif metric == "bic":
                        score = vp.bic_score(y_true, y_score or model.predict(vdf), vdf)
                    elif metric == "r2":
                        score = vp.r2_score(y_true, y_score or model.predict(vdf), vdf)
                    elif metric == "rmse":
                        score = vp.rmse(y_true, y_score or model.predict(vdf), vdf)
                    elif metric == "mae":
                        score = vp.mae(y_true, y_score or model.predict(vdf), vdf)
                    elif metric == "precision":
                        score = vp.precision_score(y_true, y_score or model.predict(vdf), vdf)
                    elif metric == "recall":
                        score = vp.recall_score(y_true, y_score or model.predict(vdf), vdf)
                    elif metric == "f1":
                        score = vp.f1_score(y_true, y_score or model.predict(vdf), vdf)
                    else:
                        score = None
                        
                    results[metric] = score
                    
                except Exception as metric_error:
                    logger.warning(f"Could not calculate {metric}: {metric_error}")
                    results[metric] = f"Error: {str(metric_error)}"
            
            return {
                "success": True,
                "model_name": model_name,
                "dataframe_name": dataframe_name,
                "metrics": results,
                "y_true": y_true,
                "y_score": y_score
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise MCPToolError(f"Failed to evaluate model: {str(e)}")


class PredictModelTool(VerticaPyMCPTool):
    """Tool for making predictions with trained models."""
    
    def __init__(self):
        super().__init__(
            name="predict_model",
            description="Make predictions using a trained model"
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Name of the trained model"
                },
                "dataframe_name": {
                    "type": "string",
                    "description": "Name of vDataFrame to make predictions on"
                },
                "prediction_column": {
                    "type": "string",
                    "description": "Name for the prediction column",
                    "default": "prediction"
                },
                "probability": {
                    "type": "boolean",
                    "description": "Return probabilities for classification models",
                    "default": False
                }
            },
            "required": ["model_name", "dataframe_name"]
        }
    
    async def execute(self, args: Dict[str, Any], session: Any) -> Dict[str, Any]:
        """Make predictions."""
        try:
            model_name = args["model_name"]
            dataframe_name = args["dataframe_name"]
            prediction_column = args.get("prediction_column", "prediction")
            use_probability = args.get("probability", False)
            
            # Get model and dataframe from session
            if model_name not in session.models:
                raise MCPSessionError(f"Model '{model_name}' not found in session")
            if dataframe_name not in session.dataframes:
                raise MCPSessionError(f"vDataFrame '{dataframe_name}' not found in session")
            
            model = session.models[model_name]
            vdf = session.dataframes[dataframe_name]
            
            # Make predictions
            if use_probability and hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(vdf, name=prediction_column)
            else:
                predictions = model.predict(vdf, name=prediction_column)
            
            return {
                "success": True,
                "model_name": model_name,
                "dataframe_name": dataframe_name,
                "prediction_column": prediction_column,
                "prediction_type": "probability" if use_probability else "class",
                "message": f"Predictions added as column '{prediction_column}'"
            }
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise MCPToolError(f"Failed to make predictions: {str(e)}")