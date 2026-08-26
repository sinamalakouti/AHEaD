"""Proposer and refiner prompt templates."""

SYSTEM_PROMPT = """As an expert on cross-cultural visual representation, your task is to generate precise visual descriptors. These descriptors will be used to evaluate the cultural alignment and accuracy of AI-generated images.

Your goal is to provide descriptors that capture visual elements of a typical scene of an activity in a specific country. Your descriptors should cover different typical variations of the scene for the activty. This includes both traditional and modern scenarios, thereby creating a robust ground truth for evaluation.

Universal Rules:
1.  Output Format: The output format must strictly follow the following JSON structure, without any addtional text or comments. Use the following example as reference for JSON format, but do not include keywrod json in the output.
    {{
      "descriptors": [
        {{
          "token": "A 1-4 word descriptive phrase.",
          "style": "traditional|modern|neutral",
          "necessity": "necessary|optional"
        }}
      ]
    }}
2.  Culturally-Aware Terminology: Use language that is culturally and contextually accurate. Prefer specific, common terms (e.g., "samovar," "sari") over generic ones where it enhances precision. However, use broader descriptions (e.g., "traditional West African attire," "Lunar New Year decorations") when a highly specific name is not essential to the scene's core meaning.
3.  Focus on the Core Scene: Descriptors must mainly focus on the activity and details about the activity itself. It should not be about actions or scenes before or after the activity.
4.  Capture Variation: When multiple common variations exist (e.g., eating at a table vs. on the floor), provide descriptors for each to ensure comprehensive coverage.
5.  Necessity Definition:
    - necessary: An element whose absence would render the scene culturally inaccurate or unrecognizable.
    - optional: An element that commonly appears and adds to a plausible representation but is not strictly required.
6.  If for a dimension no distinctive or representative descriptor exist, return an empty descriptor list.
"""

PROMPT_SETTING_TEMPLATE = """## DIMENSION: Setting and Background
Goal: To describe the overall environment, including the physical location, background, and key artistic or design elements that define the scene's atmosphere. 

Generate upto {max_items} visual descriptors for the concept: {concept}.

Guidelines:
- INCLUDE:
    - Ensure that the descriptors are accurate about a typical scene for the given activity in the given country while covering all the typical variations.
    - Location & Architecture & landmarks: The type of space, indoors or outdoors (e.g., "ornate temple interior," "bustling city street," "simple living room").
    - Art & Design: Prominent artistic styles, patterns, or design features (e.g., "calligraphy wall art," "geometric tile patterns," "minimalist decor").
    - Major Furnishings: Large, non-handheld items that define the space (e.g., "long communal table," "floor cushions and rugs").
- EXCLUDE: People, clothing, handheld objects, and specific actions or interactions.
"""


PROMPT_OBJECTS_TEMPLATE = """## DIMENSION: Objects
Goal: To identify the key objects, tools, foods, and vessels that are central to the activity. 

Generate upto {max_items} visual descriptors for the concept: {concept}.

Guidelines:
- Ensure that the descriptors are accurate about a typical scene for the given activity in the given country while covering all the typical variations.
- INCLUDE: Core items used in the activity (e.g., "samovar," "board game"), culturally specific vessels, tools, or other objects.
- For food, use visually descriptive categories (e.g., "bowls of noodle soup," "plate of kababs," "shared hot pot") rather than abstract national labels (e.g., "Chinese food"), as the descriptors must guide image generation.
- EXCLUDE: People, animals, clothing, architectural elements, actions, and general background décor.
"""


PROMPT_ATTIRE_TEMPLATE = """## DIMENSION: Attire
Goal: To describe the typical clothing, accessories, and appearance of people, capturing a plausible range of styles from traditional to modern.
Generate upto {max_items} visual descriptors for the concept: {concept}.

Guidelines:
- Descriptors should reflect the correct level of specificity. 
  -Use a specific garment name (e.g., 'sari') only when that garment is a defining, essential part of the scene. 
  -If multiple styles of clothing would be appropriate for the activity, use a broader, more inclusive description (e.g., 'traditional West African attire').
  -Unless the concept is strictly historical, provide descriptors for both traditional and contemporary styles, using the general term 'modern clothing' for the latter.
- INCLUDE: Garments, headwear, accessories, ceremonial body markings, and uniforms.
- EXCLUDE: Non-wearable objects (e.g., tools, furniture), actions, and gestures.
"""


PROMPT_INTERACTION_TEMPLATE = """## DIMENSION: Interaction and Gesture
Goal: To capture the actions, gestures, and social dynamics of the scene. 

Generate upto {max_items} visual descriptors for the concept: {concept}.

Guidelines:
- Ensure that the descriptors are accurate about a typical scene for the given activity in the given country while covering all the typical variations.
- INCLUDE: Key actions between people and objects (e.g., "pouring tea from samovar"), distinctive social gestures (e.g., "sharing from communal dish"), group formations or postures (e.g., "dancing in a circle"), and culturally specific interactions that are an integral part of the activity.
- EXCLUDE: Static descriptions of objects, clothing, or the setting. Focus on the verbs of the scene.
"""


PROMPT_SPATIAL_TEMPLATE = """## DIMENSION: Spatial Arrangement
Goal: To describe the physical layout and positioning of people and key objects relative to each other.

Generate upto {max_items} visual descriptors for the concept: {concept}.

Guidelines:
- Ensure that the descriptors are accurate about a typical scene for the given activity in the given country while covering all the typical variations.
- INCLUDE: The positioning of people relative to major objects or surfaces (e.g., "seated on floor around sofreh," "standing in a line"), and the overall composition of the scene if it is culturally distinct.
- Should cover the most common and culturally significant configurations. For instance, "seated on floor around spread/Sofreh" AND "seated around table" for eating in Iran.
- EXCLUDE: Detailed descriptions of clothing, objects, or specific gestures and actions.
"""


PROMPT_PEOPLE_TEMPLATE = """## DIMENSION: People
Goal: To describe visible human appearance so that people's look plausibly matches the country/region for the concept.

Generate upto {max_items} visual descriptors for the concept: {concept}.

Guidelines:
- INCLUDE: Typical appearance cues for people in this country/region (without stereotyping):
    - common hairstyles and facial hair
    - skin tone ranges that plausibly reflect the population
    - cover different genders

- EXCLUDE: Clothing items (use Attire), objects and background (use Objects/Setting), fine gestures (use Interaction), and unverifiable identity labels."""


refiner_prompt_template = """Refine candidate visual descriptors for evaluating AI generated images for the cultural alignment of the concept/activity in the country.
## Task
Select and clean descriptors with respect to the concept/activity, country, for each dimension.

## Dimensions
- Setting: what appears in the background of the scene/imgae. E.g., venues, architecture, fixtures, décor, environmental settings
- Objects: central objects in the image and activity. 
- Attire: clothing, accessories, headwear, footwear
- Interaction: actions, gestures, postures, and interaction between people and objects
- Spatial Layout: positioning patterns, arrangements relative to objects and people
- People: common features of people in the given country.

## Rules
1. Keep only culturally accurate descriptors according to the activity, country, and dimension.
2. Create a accurate and diverse set of descriptors that cover all the typical variations of the activity in the country.
3. Do not invent new descriptors and only clean the provided candidates.
4. Merge duplicates and overly specific items into one or more broader terms.
5. Descriptors must be strictly relevant to the dimension, with wrong/unrelted desriptors being removed.
6. Keep phrases concise (1-4 words)
7. Descriptors of each dimension should specificallly related to that dimension.
8. The number of descriptor should be upto {max_items}.
9. if no descriptors are relevant to the dimension, return an empty list.

## Output Format is JSON:
[{{"token":"item name", "style":"traditional|modern|neutral", "necessity":"necessary|optional"}}]

## INPUT:
Concept: {prompt} in {country}
Dimension: {dimension}
Candidate Descriptors: {candidate_descriptors}
"""
