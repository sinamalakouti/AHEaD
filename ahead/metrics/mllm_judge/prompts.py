"""Rubric prompts for MLLM judge (``score is <n>``)."""

ALIGN_BASE = """How well does the image depict this activity as it would typically appear in the given country?
Rate on a scale from 1 to 5:  
1 = Not well at all (completely wrong or unrelated)  
2 = Slightly relevant  
3 = Somewhat accurate  
4 = Mostly accurate  
5 = Exactly as expected for this country

TASK: 
PROMPT: {prompt}
COUNTRY: {country}

IMPORTANT: Respond with ONLY the score in this exact format:
score is <num>

Do not include any other text, explanation, or formatting. If image is not clear, use your best judgement to score the image."""

ALIGN_DESCRIPTORS = """How well does the image depict this activity as it would typically appear in the given country?
Rate on a scale from 1 to 5:  
1 = Not well at all (completely wrong or unrelated)  
2 = Slightly relevant  
3 = Somewhat accurate  
4 = Mostly accurate  
5 = Exactly as expected for this country

TASK: 
PROMPT: {prompt}
COUNTRY: {country}
DESCRIPTORS: {descriptors}

Answer above question according to the image and given descriptors. Note that descriptors describe different possible visualization of the activity in the country.
Descriptors mutually are exclusive and many of them are may be optional. 

IMPORTANT: Respond with ONLY the score in this exact format:
score is <num>

Do not include any other text, explanation, or formatting."""

HAL_BASE = """Does the image contain incorrect cultural or national elements based on the country mentioned?
Rate on a scale from 1 to 5:
1 = Not at all. All elements seem appropriate  
2 = Slightly. One minor element seems foreign  
3 = Somewhat. A few elements feel unusual  
4 = Mostly. Many parts image wrong or reflect another culture  
5 = Strongly. The image is completely wrong, most elements are wrong, the activity is wrong, or the scene belongs to another country.

TASK: 
PROMPT: {prompt}
COUNTRY: {country}

IMPORTANT: Respond with ONLY the score in this exact format:
score is <num>

Do not include any other text, explanation, or formatting. if image is not clear, use your best judgement to score the image."""

HAL_DESCRIPTORS = """Does the image contain incorrect cultural or national elements based on the country mentioned?
Rate on a scale from 1 to 5:
1 = Not at all. All elements seem appropriate  
2 = Slightly. One minor element seems foreign  
3 = Somewhat. A few elements feel unusual  
4 = Mostly. Many parts image wrong or reflect another culture  
5 = Strongly. The image is completely wrong, most elements are wrong, the activity is wrong, or the scene belongs to another country.

TASK: 
PROMPT: {prompt}
COUNTRY: {country}
DESCRIPTORS: {descriptors}


Answer above question according to the image and given descriptors. Note that descriptors describe different possible visualization of the activity in the country.
Descriptors mutually are exclusive and many of them are may be optional. 

IMPORTANT: Respond with ONLY the score in this exact format:
score is <num>

Do not include any other text, explanation, or formatting. If image is not clear, use your best judgement."""

EXAG_BASE = """How exaggerated is the image in terms of cultural elements (e.g. attire, objects, background, art/design)?
Rate on a scale from 1 to 5:
1 = Appropriate and balanced  
2 = Slightly decorative but within cultural norms  
3 = Several features feel idealized or overly emphasized  
4 = Many elements feel exaggerated and unrealistic  
5 = Strongly stereotypical, misleading, or culturally irrelevant

TASK: 
PROMPT: {prompt}
COUNTRY: {country}

IMPORTANT: Respond with ONLY the score in this exact format:
score is <num>

Do not include any other text, explanation, or formatting. If image is not clear, use your best judgement to score the image."""


# metric -> (base_template, descriptor_conditioned_template or None)
TEMPLATES = {
    "align": (ALIGN_BASE, ALIGN_DESCRIPTORS),
    "hal": (HAL_BASE, HAL_DESCRIPTORS),
    "exag": (EXAG_BASE, None),
}
