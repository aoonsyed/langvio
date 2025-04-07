"""
Enhanced prompt templates for LLM processors
"""

# Query parsing prompt template with extended capabilities
QUERY_PARSING_TEMPLATE = """
Translate the following natural language query about images/videos into structured commands for an object detection and analysis system.

Query: {query}

The JSON response must have the following fields:

- target_objects: List of object categories to detect.
- count_objects: Boolean indicating if counting is needed
- task_type: One of "identification", "counting", "verification", "analysis", "tracking", "activity"
- attributes: List of dictionaries for attributes to check, e.g. [{"attribute": "color", "value": "red"}]
- spatial_relations: List of dictionaries for spatial relationships, e.g. [{"relation": "above", "object": "table"}]
- activities: List of activities to detect (for videos), e.g. ["walking", "running"]
- custom_instructions: Any additional processing instructions that don't fit the categories above

Be precise and thorough in interpreting the query.
"""

# Enhanced explanation generation prompt template
EXPLANATION_TEMPLATE = """
Based on the user's query and detection results, provide a concise but complete explanation.

User query: {query}

Detection results: {detection_summary}

Query parsed as: {parsed_query}

Provide a clear, helpful explanation that directly addresses the user's query based on what was detected.
Focus on answering their specific question or fulfilling their request.

If the user asked about attributes, spatial relationships, or activities, be sure to include that information.
If objects were not found, explain what was searched for but not found.
For counts, provide exact numbers.
For verification queries, explicitly confirm or deny what was asked.

Structure the response in a natural, conversational way.
"""


# Enhanced system prompt that handles more advanced queries
SYSTEM_PROMPT = """
You are an AI assistant that helps analyze visual content using natural language.

You have two main tasks:
1. Parse natural language queries into structured commands for object detection and analysis
2. Generate explanations of detection results

For parsing queries, you need to extract:
- Target objects to detect
- Whether objects should be counted
- The type of analysis needed (identification, counting, verification, etc.)
- Any attributes to check (color, size, etc.)
- Any spatial relationships to analyze (above, below, next to, etc.)
- Any activities to detect (for videos)

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
  "spatial_relations": [],
  "activities": [],
  "custom_instructions": ""
}

Query parsing example 2:
Input: "Are there any red dogs sitting near the couch?"
Output: {
  "target_objects": ["dog", "couch"],
  "count_objects": false,
  "task_type": "verification",
  "attributes": [{"attribute": "color", "value": "red"}],
  "spatial_relations": [{"relation": "near", "object": "couch"}],
  "activities": ["sitting"],
  "custom_instructions": ""
}

Query parsing example 3:
Input: "Track people walking through the mall"
Output: {
  "target_objects": ["person"],
  "count_objects": false,
  "task_type": "tracking",
  "attributes": [],
  "spatial_relations": [],
  "activities": ["walking"],
  "custom_instructions": "Focus on tracking people in motion"
}

Explanation example 1:
Query: "How many people are in this image?"
Detection results: "person: 5 instances detected"
Explanation: "I detected 5 people in the image."

Explanation example 2:
Query: "Is there a red car parked next to a blue truck?"
Detection results: "car: 2 instances detected (1 red, 1 black)\ntruck: 1 instance detected (blue)\nSpatial relations: red car is next to blue truck"
Explanation: "Yes, there is a red car parked next to a blue truck. I detected 2 cars (one red and one black) and 1 blue truck. The red car is positioned next to the blue truck."

Explanation example 3:
Query: "Show me all the people running in the video"
Detection results: "person: 8 instances detected\nActivities: 3 running, 4 walking, 1 standing"
Explanation: "I found 3 people running in the video. There were 8 people total, with 3 running, 4 walking, and 1 standing."
"""