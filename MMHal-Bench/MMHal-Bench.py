import json
import os

import datasets


_DESCRIPTION = """\
MMHal-Bench is a new evaluation benchmark specifically designed for hallucintation in Large Multimodal Models (LMM). It contains 96 challenging questions based on images from OpenImages, and their corresponding ground-truth answers and image contents.
"""

_CITATION = """\
@article{2023llavarlhf,
author      = {Zhiqing Sun and Sheng Shen and Shengcao Cao and Haotian Liu and Chunyuan Li and Yikang Shen and Chuang Gan and Liang-Yan Gui and Yu-Xiong Wang and Yiming Yang and Kurt Keutzer and Trevor Darrell},
title       = {Aligning Large Multimodal Models with Factually Augmented RLHF},
publisher   = {arXiv:2309.14525},
year        = {2023}
}
"""

_URLS = {
    "test": "test_data.zip",
}

# Example usage:
# from datasets import load_dataset
# dataset = load_dataset('MMHal-Bench.py')
# print(dataset['test'][0])

class MMHalBench(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("int32"),
                    "question_type": datasets.Value("string"),
                    "question_topic": datasets.Value("string"),
                    "image_id": datasets.Value("string"),
                    "image_src": datasets.Value("string"),
                    "image_content": datasets.features.Sequence(datasets.Value("string")),
                    "question": datasets.Value("string"),
                    "gt_answer": datasets.Value("string"),
                    "model_answer": datasets.Value("string"),
                    "image_path": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://llava-rlhf.github.io/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)
        return [datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),]

    def _generate_examples(self, filepath, split="test"):
        json_file = os.path.join(filepath, "response_template.json")
        image_dir = os.path.join(filepath, "images")
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
            for idx, line in enumerate(data):
                image_file = os.path.split(line["image_src"])[1]
                line['image_path'] = os.path.join(image_dir, image_file)
                line['id'] = idx
                yield idx, line
