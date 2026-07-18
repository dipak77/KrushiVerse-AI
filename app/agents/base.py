from abc import ABC, abstractmethod

class BaseAgriAgent(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, query: str, context: dict) -> dict:
        """Execute agent-specific reasoning and tool utilization."""
        pass
