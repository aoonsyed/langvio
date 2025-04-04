"""
Prompt templates for LLM processors
"""

# Query parsing prompt template
QUERY_PARSING_TEMPLATE = """
Translate the following natural language query about images/videos into structured commands for an object detection system.

Query: {query}

Return a JSON with these fields:
- target_objects: List of object categories to detect (e.g., "person", "car", "dog", etc.)
- count_objects: Boolean indicating if counting is needed
- task_type: One of "identification", "counting", "verification", "analysis"
- attributes: Any specific attributes to look for (e.g., "color", "size", "activity")
- spatial_relations: Any spatial relationships to check (e.g., "next to", "on top of")

JSON response:
"""

# Explanation generation prompt template
EXPLANATION_TEMPLATE = """
Based on the user's query and detection results, provide a concise explanation.

User query: {query}

Detection results: {detection_summary}

Provide a clear, helpful explanation that directly addresses the user's query based on what was detected.
Focus on answering their specific question or fulfilling their request.

Explanation:
"""

# Video analysis prompt template
VIDEO_ANALYSIS_TEMPLATE = """
Analyze the following video detection results to identify patterns and changes over time.

User query: {query}
Video length: {video_length} seconds
Frames analyzed: {frames_analyzed}
Frame rate: {frame_rate} fps

Detection summary:
{detection_summary}

Frame-by-frame notable objects:
{frame_details}

Provide an analysis that captures temporal patterns relevant to the user's query.
Focus on movements, appearances/disappearances, and changes in object positions over time.

Analysis:
"""

