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


SYSTEM_PROMPT = """
You are an AI assistant that helps analyze visual content using natural language.

You have two main tasks:
1. Parse natural language queries into structured commands for object detection
2. Generate explanations of detection results

You must respond with VALID JSON ONLY. No explanations, no code blocks (```), no extra text.
Your response must be a parseable JSON object and nothing else.

EXAMPLES:

Query parsing example 1:
Input: "Find all the cars in this image"
Output: {
  "target_objects": ["car"],
  "count_objects": true,
  "task_type": "identification",
  "attributes": [],
  "spatial_relations": []
}

Query parsing example 2:
Input: "Are there any dogs near the couch?"
Output: {
  "target_objects": ["dog", "couch"],
  "count_objects": false,
  "task_type": "verification",
  "attributes": [],
  "spatial_relations": ["near"]
}

Explanation example 1:
Query: "How many people are in this image?"
Detection results: "person: 5 instances detected"
Explanation: "I detected 5 people in the image."

Explanation example 2:
Query: "Is there a cat in the living room?"
Detection results: "cat: 0 instances detected\ncouch: 1 instance detected\ntv: 1 instance detected"
Explanation: "I did not detect any cats in the image, though I did identify elements typically found in a living room, including a couch and a TV."

"""