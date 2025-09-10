"""
Base classes for MCP tools using VerticaPy functions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class VerticaPyMCPTool(ABC):
    """Base class for VerticaPy MCP tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool's input schema."""
        pass
    
    @abstractmethod
    async def execute(self, args: Dict[str, Any], session: Any) -> Dict[str, Any]:
        """Execute the tool with given arguments and session state."""
        pass
    
    def validate_args(self, args: Dict[str, Any]) -> bool:
        """Validate tool arguments against schema."""
        return True