import os
import json
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset

import myutils
from eval_data_loader import COCODataSet
from llava.utils import disable_torch_init
from model_loader import ModelLoader
from steering_vector import obtain_vsv, add_logits_flag, remove_logits_flag
from llm_layers import add_vsv_layers, remove_vsv_layers


def parse_args():
    parser = argparse.ArgumentParser(description="CHAIR evaluation on MLLMs.")
    # General arguments
    parser.add_argument("--exp_folder", type=str, default="chair_eval", help="save folder name")
    parser.add_argument("--model", type=str, help="model")
    parser.add_argument("--data-path", type=str, default="../download_datasets/COCO_2014/val2014", help="data path")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--subset-size", type=int, default=500)

    # Visual steering vector arguments
    parser.add_argument("--vsv", action="store_true", help='Use visual steering vector')
    parser.add_argument("--lamda", type=float, default=0.1)
    parser.add_argument("--layers", default=None)

    # penultimate logits augmentation
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
    f = open(result_file, "w", encoding="utf-8")

    # get model loader
    model_loader = ModelLoader(args.model)
    # get dataloader
    coco_dataset = COCODataSet(data_path=args.data_path, trans=model_loader.image_processor)
    # get a randomly sample subdataset without replacement and fixed seed
    if args.subset_size > 0 and args.subset_size < len(coco_dataset):
        subset_indices = np.random.choice(len(coco_dataset), args.subset_size, replace=False)
        coco_dataset = Subset(coco_dataset, subset_indices)

    coco_loader = DataLoader(coco_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)
    # prepare template
    template = myutils.prepare_template(args)

    # inference
    for _, data in tqdm(enumerate(coco_loader), total=len(coco_loader)):
        with torch.inference_mode():
            img_id = data["img_id"]
            image = data["image"]
            batch_size = img_id.shape[0]
            query = ["Please help me describe the image in detail."] * batch_size

            with myutils.maybe_autocast(args.model, model_loader.vlm_model.device):
                # prepare inputs
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

                # generate
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

            output_text = model_loader.decode(outputs)

        # write to file
        for i in range(len(output_text)):
            f.write(json.dumps({"image_id": int(img_id[i]), "caption": output_text[i]}) + "\n")
        f.flush()
    f.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)