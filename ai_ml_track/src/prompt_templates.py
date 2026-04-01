def build_labeling_prompt(clusters: dict) -> str:
    cluster_text = ""
    for cluster_id, items in clusters.items():
        cluster_text += f"Cluster {cluster_id}: {items}\n"

    prompt = f"""
You are a financial transaction categorisation expert working 
for a Nigerian fintech company that processes thousands of 
transactions daily for tax purposes.

I have used a semantic embedding model to group the following 
financial transaction descriptions into clusters based on 
their meaning. Your job is to:

1. Give each cluster a short, meaningful label
2. Assign a confidence level: "high", "medium", or "low"
3. Write a clear explanation of why these items belong together
4. Identify any items that do not clearly belong in their 
   cluster and move them to "ungrouped"

Here are the clusters:

{cluster_text}

RULES:
- Base your labels on the semantic meaning of the transactions,
  not just the words used. For example, "AWS" and 
  "Amazon Web Services" refer to the same service.
- Distinguish carefully between similar but different services.
  For example, "Uber ride" is transport but "Uber Eats" is 
  food delivery — these are different categories.
- Confidence should be "high" when all items clearly belong 
  together, "medium" when most do but there is some ambiguity,
  and "low" when the grouping is uncertain.
- If a cluster contains only one item, still label it but 
  note in your explanation that it is a singleton group.
- You MUST respond with valid JSON only. No extra text, no 
  markdown, no code fences. Just the raw JSON object.

Your response must follow this exact structure:
{{
  "groups": [
    {{
      "label": "Category name here",
      "items": ["item1", "item2"],
      "confidence": "high",
      "explanation": "Your explanation here"
    }}
  ],
  "ungrouped": []
}}

EXAMPLE:
If you receive:
Cluster 0: ["Uber trip 1200", "Bolt ride 900"]
Cluster 1: ["Netflix subscription 4500", "NETFLIX.COM 4500"]

You should return:
{{
  "groups": [
    {{
      "label": "Ride-hailing",
      "items": ["Uber trip 1200", "Bolt ride 900"],
      "confidence": "high",
      "explanation": "Both Uber and Bolt are ride-hailing services operating in Nigeria."
    }},
    {{
      "label": "Streaming",
      "items": ["Netflix subscription 4500", "NETFLIX.COM 4500"],
      "confidence": "high",
      "explanation": "Both refer to Netflix. Different raw formats, same service."
    }}
  ],
  "ungrouped": []
}}
"""
    return prompt.strip()