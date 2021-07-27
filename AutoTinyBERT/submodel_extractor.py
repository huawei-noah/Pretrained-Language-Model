import os
import json
import torch
import argparse

from transformer.modeling_extractor import SuperBertModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument('--arch',
                        type=str,
                        required=True)
    parser.add_argument('--output',
                        type=str,
                        required=True)
    parser.add_argument('--kd', action='store_true')

    args = parser.parse_args()

    model = SuperBertModel.from_pretrained(args.model)
    size = 0
    for n, p in model.named_parameters():
        size += p.nelement()
        print('n: {}#@#p: {}'.format(n, p.nelement()))

    print('the model size is : {}'.format(size))

    arch = json.loads(json.dumps(eval(args.arch)))

    print('kd: {}'.format(args.kd))

    kd = True if args.kd else False
    model.module.set_sample_config(arch, kd) if hasattr(model, 'module') \
        else model.set_sample_config(arch, kd)

    size = 0
    for n, p in model.named_parameters():
        size += p.nelement()
        print('n: {}#@#p: {}'.format(n, p.nelement()))

    print('the extracted model size is : {}'.format(size))

    model_to_save = model.module if hasattr(model, 'module') else model

    model_output = os.path.join(args.output, 'pytorch_model.bin')
    torch.save(model_to_save.state_dict(), model_output)


if __name__ == "__main__":
    main()

