"""
Prompt templates for LLM processors
"""

# Query parsing prompt template
QUERY_PARSING_TEMPLATE = """
Translate the following natural language query about images/videos into structured commands for an object detection system.

Query: {query}

The JSON response must have the following fields:

- target_objects: List of object categories to detect.
- count_objects: Boolean indicating if counting is needed
- task_type: One of \"identification\", \"counting\", \"verification\", \"analysis\"
- attributes: Any specific attributes to look for (e.g., \"color\", \"size\", \"activity\")
- spatial_relations: Any spatial relationships to check (e.g., \"next to\", \"on top of\")

"""

# Explanation generation prompt template
EXPLANATION_TEMPLATE = """
Based on the user's query and detection results, provide a concise explanation.

User query: {query}

Detection results: {detection_summary}

Provide a clear, helpful explanation that directly addresses the user's query based on what was detected.
Focus on answering their specific question or fulfilling their request.

"""


