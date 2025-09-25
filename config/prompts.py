"""
Centralized prompt templates for LLM interactions.
"""

# Chain-of-Thought prompt for topic optimization
OPTIMIZATION_PROMPT = """You are an expert in topic model optimization. Your task is to analyze the following list of topics and suggest actions to improve their quality.

Here is the current list of topics:
{topic_list_string}

Please follow these steps in your reasoning:

1. **Identify Overlaps:** Are there any topics that are semantically very similar or subsets of each other? List them and explain why.

2. **Identify Poor Quality Topics:** Are there any topics with low quality scores or keywords that seem incoherent and generic? List them.

3. **Propose Actions:** Based on your analysis, propose a list of actions. The only valid actions are 'MERGE' and 'SPLIT'. 'MERGE' should target two or more topic IDs. 'SPLIT' should target a single topic ID.

Your final output must be a JSON array of action objects. For example:
[
  {{"action": "MERGE", "targets": [3, 15], "reason": "Topics 3 and 15 both discuss cybersecurity risks and can be combined."}},
  {{"action": "SPLIT", "targets": [8], "reason": "Topic 8 is too broad, covering both hardware and software infrastructure, and should be separated."}}
]

If no actions are needed, return an empty array: []

JSON Output:"""

# Topic naming and summary prompt
TOPIC_NAMING_PROMPT = """You are an expert in topic analysis. Given the following keywords from a topic, provide a clear name and summary.

Keywords: {keywords}

Please provide your response in the following JSON format:
{{
  "name": "A concise, descriptive name for this topic (max 5 words)",
  "summary": "A brief summary explaining what this topic represents (max 2 sentences)"
}}

JSON Output:"""

# Topic quality evaluation prompt
QUALITY_EVALUATION_PROMPT = """You are an expert in topic quality assessment. Please evaluate the quality of this topic based on its name and keywords.

Topic Name: {topic_name}
Keywords: {keywords}

Please assess the topic on the following criteria and provide scores from 1-10:

1. **Coherence** (1-10): How well do the keywords relate to each other and the topic name?
2. **Uniqueness** (1-10): How distinct is this topic from other potential topics?

Please provide your response in the following JSON format:
{{
  "coherence": <score_1_to_10>,
  "uniqueness": <score_1_to_10>,
  "reason": "Brief explanation of your scoring (max 2 sentences)"
}}

JSON Output:"""

# Resilience mapping prompts
RESILIENCE_MAPPING_PROMPT = """You are an expert in digital resilience assessment. Your task is to map the following topic to the appropriate digital resilience dimensions.

Topic Name: {topic_name}
Topic Keywords: {keywords}
Topic Summary: {summary}

Digital Resilience Framework:

**Absorption Capacity (Absorb):** Organization's ability to withstand major impacts while maintaining core structure and operations. Focus on "defense" and "stability" - backup systems, alternative suppliers, predictive capabilities using digital technologies, operational stability maintenance.

**Adaptive Capacity (Adapt):** Organization's ability to proactively adjust structure, operations, and strategies to respond to new environments. Focus on "adjustment" and "flexibility" - process reallocation, digital response solutions, market repositioning.

**Transformation Capacity (Transform):** Organization's ability to undergo deep, fundamental changes to adapt to entirely new conditions. Focus on "change" and "evolution" - business model redesign, emerging technology adoption, ecosystem reconstruction.

Please classify this topic into one or more resilience dimensions and provide confidence scores.

Your response must be in the following JSON format:
{{
  "dimensions": {{
    "absorb": <confidence_score_0_to_1>,
    "adapt": <confidence_score_0_to_1>,
    "transform": <confidence_score_0_to_1>
  }},
  "primary_dimension": "<absorb|adapt|transform>",
  "reasoning": "Brief explanation of your classification (max 3 sentences)"
}}

JSON Output:"""

# Batch topic analysis prompt for cost efficiency
BATCH_OPTIMIZATION_PROMPT = """You are an expert in topic model optimization. Your task is to analyze the following batch of topics and suggest optimization actions.

Topics in this batch:
{topics_batch}

Please analyze these topics as a group and identify:

1. **Within-batch relationships:** Are any of these topics similar enough to merge?
2. **Quality issues:** Do any topics have coherence or clarity problems?
3. **Optimization opportunities:** What actions would improve the overall quality?

Provide your analysis and recommendations in JSON format:
{{
  "analysis": {{
    "similar_topics": [
      {{
        "topic_ids": [id1, id2, ...],
        "similarity_reason": "explanation"
      }}
    ],
    "low_quality_topics": [
      {{
        "topic_id": id,
        "quality_issues": "explanation"
      }}
    ]
  }},
  "recommendations": [
    {{
      "action": "MERGE|SPLIT",
      "targets": [topic_ids],
      "reason": "explanation",
      "expected_benefit": "expected improvement"
    }}
  ]
}}

If no actions are needed, set "recommendations" to an empty array.

JSON Output:"""

# System prompts for different LLM models
SYSTEM_PROMPTS = {
    "gpt-4": "You are a precise and analytical expert in natural language processing and topic modeling. Provide accurate, structured responses in JSON format as requested.",
    "gpt-3.5-turbo": "You are an expert in topic analysis. Always respond in the exact JSON format requested. Be concise and accurate.",
    "local": "You are an expert assistant. Respond only in the requested JSON format without any additional text."
}

def get_prompt_template(prompt_type: str, **kwargs) -> str:
    """
    Get a formatted prompt template.

    Args:
        prompt_type: Type of prompt ('optimization', 'naming', 'quality', 'resilience', 'batch')
        **kwargs: Variables to format into the prompt

    Returns:
        Formatted prompt string
    """
    prompts = {
        "optimization": OPTIMIZATION_PROMPT,
        "naming": TOPIC_NAMING_PROMPT,
        "quality": QUALITY_EVALUATION_PROMPT,
        "resilience": RESILIENCE_MAPPING_PROMPT,
        "batch": BATCH_OPTIMIZATION_PROMPT
    }

    if prompt_type not in prompts:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    return prompts[prompt_type].format(**kwargs)

def get_system_prompt(model_name: str) -> str:
    """Get system prompt for specific model."""
    for key in SYSTEM_PROMPTS:
        if key in model_name.lower():
            return SYSTEM_PROMPTS[key]
    return SYSTEM_PROMPTS["gpt-4"]  # Default