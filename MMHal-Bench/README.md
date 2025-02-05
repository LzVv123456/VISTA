---
arxiv: 2309.14525
license: apache-2.0
task_categories:
- visual-question-answering
- image-to-text
language:
- en
pretty_name: MMHal-Bench
size_categories:
- n<1K
---
### Overview

MMHal-Bench is a new evaluation benchmark specifically designed for hallucintation in Large Multimodal Models (LMM). It contains 96 challenging questions based on images from OpenImages, and their corresponding ground-truth answers and image contents.

You may check `response_template.json` for more details. In the folder `responses` we have included some example responses from representative LMMs.

### Usage

To evaluate your own model on MMHal-Bench, first generate model responses to the image-question pairs. You may check the template `get_response.py` about how to read and write to the response file.

After that, you may let GPT-4 rate your model's responses automatically. You will need package `openai` installed and an API key. Then, run `eval_gpt4.py`:

```
python eval_gpt4.py \
    --response [JSON file with model responses] \
    --evaluation [JSON file with GPT-4 evaluation to be saved] \
    --api-key [your OpenAI API key, starting with 'sk-'] \
    --gpt-model [GPT model to be used, or 'gpt-4-0314' by default]
```

Please note that the GPT-4 API calls are not free. Depending on your model response lengths, evaluating each question may use 1.5k-2k tokens. Also, GPT-4 responses are not deterministic, so you may get different results with the same responses.

At the end of the outputs, you can see the evaluation results like this:

```
Average score: 2.05
Hallucination rate: 0.61
Average score for each question type: 2.33,1.25,2,2.5,1.5,3.33,2.33,1.17
```