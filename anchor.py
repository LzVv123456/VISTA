IMAGE_TOKEN_INDEX = -200
IMAGE_TOKEN_LENGTH = 576
MINIGPT4_IMAGE_TOKEN_LENGTH = 32
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
SHIKRA_IMAGE_TOKEN_LENGTH = 256
SHIKRA_IMG_START_TOKEN = 32001
SHIKRA_IMG_END_TOKEN = 32002


INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:",
    "internvl": "USER: <ImageHere> <question> ASSISTANT:",
}

INSTRUCTION_TEMPLATE_NO_IMG = {
    "minigpt4": "###Human:<question> ###Assistant:",
    "instructblip": "<question>",
    "lrv_instruct": "###Human: <question> ###Assistant:",
    "shikra": "USER: <question> ASSISTANT:",
    "llava-1.5": "USER: <question> ASSISTANT:",
    "internvl": "USER: <question> ASSISTANT:",
}

IMAGE_PLACEHOLDER = {
    "minigpt4": "<Img><ImageHere></Img>",
    "instructblip": "<ImageHere>",
    "lrv_instruct": "<Img><ImageHere></Img>",
    "shikra": "<im_start><ImageHere><im_end>",
    "llava-1.5": "<ImageHere>",
    "internvl": "<ImageHere>",
}


SYSTEM_MESSAGE = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."


POPE_PATH = {
    "random": "./pope_coco/coco_pope_random.json",
    "popular": "./pope_coco/coco_pope_popular.json",
    "adversarial": "./pope_coco/coco_pope_adversarial.json",
}


GPT_EVAL_PROMPT = '''
You are a vision-language evaluator. Given an image and an AI-generated description, perform the following tasks:

1. List clearly visible contents in the image that are not mentioned in the description.
2. List hallucinated contents in the description that are not present in the image.
3. List contents accurately described in the description that match the image.

For each task, include objects, object properties (e.g., color, count, position), and relationships between objects. You must answer each content with a single word, separating different contents by commas. If no contents apply, write "None". Make sure there is no overlapping words between three tasks.

Answer 1: [Missing contents]
Answer 2: [Hallucinated contents]
Answer 3: [Accurate contents]
'''


# OpenAI API Key
API_KEY = "Your OpenAI API Key"