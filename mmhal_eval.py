import os
import json
import argparse
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from tqdm import tqdm

import torch

import myutils
from llava.utils import disable_torch_init
from model_loader import ModelLoader
from steering_vector import obtain_vsv, add_logits_flag, remove_logits_flag
from llm_layers import add_vsv_layers, remove_vsv_layers


def parse_args():
    parser = argparse.ArgumentParser(description="CHAIR evaluation on MLLMs.")
    # General arguments
    parser.add_argument("--exp_folder", type=str, default="mmhal_eval", help="save folder name")
    parser.add_argument("--model", type=str, help="model")
    parser.add_argument("--input-path", type=str, default='./MMHal-Bench/response_template.json', help='input template')
    parser.add_argument("--data-path", type=str, default="./MMHal-Bench/images", help="image data path")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--subset-size", type=int, default=-1)

    # Visual steering vector arguments
    parser.add_argument("--vsv", action="store_true", help='Use visual steering vector')
    parser.add_argument("--lamda", type=float, default=0.1)
    parser.add_argument("--layers", default=None)

    # self logits augmentation
    parser.add_argument("--logits_aug", action="store_true", help='Use penultimate logits augmentation')
    parser.add_argument("--logits_layers", type=str, default='25,30', help='Layer for penultimate logits augmentation')
    parser.add_argument("--logits_alpha", type=float, default=0.3, help='Alpha for penultimate logits augmentation')

    # Decoding arguments
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)

    # Miscellaneous arguments
    parser.add_argument("--seed", type=int, default=1994)
    parser.add_argument("--num-workers", type=int, default=1)

    return parser.parse_args()


def get_file_name(args):
    file_name = "_".join(myutils.prepare_common_fileparts(args))
    return file_name


def load_image(image_file, data_path):
    # get name of the image
    img_name = image_file.split("/")[-1]
    img_path = os.path.join(data_path, img_name)
    if os.path.exists(img_path):
        return Image.open(img_path).convert("RGB")
    else:
        if image_file.startswith('http') or image_file.startswith('https'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # bath size should be 1 as we are generating image specific steering vectors
    assert args.batch_size == 1, "Batch size should be 1"
    # seed everything
    myutils.seed_everything(args.seed)
    # disable torch init
    disable_torch_init()
    # init_folder_structure
    args.save_dir = myutils.init_folder_structure(args)
    # prepare file name
    args.file_name = get_file_name(args)

    # prepare save file
    result_file = os.path.join(args.save_dir, args.file_name + ".jsonl")
    if os.path.exists(result_file):
        exit(f"Result file {result_file} already exists. Exiting.")
        # os.remove(result_file)
    f = open(result_file, "w", encoding="utf-8")

    # get model loader
    model_loader = ModelLoader(args.model)
    # get data
    json_data = json.load(open(args.input_path, 'r'))
    if args.subset_size > 0:
        json_data = json_data[:args.subset_size]
    # prepare template
    template = myutils.prepare_template(args)

    for _, line in tqdm(enumerate(json_data), total=len(json_data)):
        with torch.inference_mode():
            image = model_loader.image_processor(load_image(line['image_src'], args.data_path))
            query = [line['question']] 

            with myutils.maybe_autocast(args.model, model_loader.vlm_model.device):
                questions, kwargs = model_loader.prepare_inputs_for_model(template, query, image)
                
                # add visual steering vectors
                if args.vsv:
                    neg_kwargs = model_loader.prepare_neg_prompt(args, questions, template=template)
                    pos_kwargs = model_loader.prepare_pos_prompt(args, kwargs)
                    # generate visual steering vectors
                    visual_vector, _ = obtain_vsv(args, model_loader.llm_model, [[neg_kwargs, pos_kwargs]], rank=1)
                    # add steering vectors
                    add_vsv_layers(model_loader.llm_model, torch.stack([visual_vector], dim=1).cuda(), [args.lamda], args.layers)

                # add logits augmentation flag
                add_logits_flag(model_loader.llm_model, args)

                # generate
                if args.do_sample:
                    kwargs['top_p'] = args.top_p
                    kwargs['top_k'] = args.top_k

                outputs = model_loader.llm_model.generate(
                    do_sample=args.do_sample,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    num_beams=args.num_beams,
                    output_attentions=False,
                    output_hidden_states=True if args.logits_aug else False,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    return_dict=True,
                    **kwargs
                    )
                
                # remove logits augmentation flag
                remove_logits_flag(model_loader.llm_model)
                
                if args.vsv:
                    # remove steering vectors 
                    remove_vsv_layers(model_loader.llm_model)

            # decode outputs
            output_text = model_loader.decode(outputs)

        # print(idx, response)
        line['model_answer'] = output_text[0]

    json.dump(json_data, f, indent=2)
    f.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)