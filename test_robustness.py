#!/usr/bin/env python
""" COCO validation script

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import time
import torch
import torch.nn.parallel
from contextlib import suppress
from effdet.anchors import Anchors, AnchorLabeler

from effdet import create_model, create_evaluator, create_dataset, create_loader
from effdet.data import resolve_input_config
from timm.utils import AverageMeter, setup_default_logging
from timm.models.layers import set_layer_config
from attacks import AttackerBuilder
from torchvision.utils import save_image
import os
import numpy as np
from utils import print_coco_results
has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True


def add_bool_arg(parser, name, default=False, help=''):  # FIXME move to utils
    dest_name = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=dest_name, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=dest_name, action='store_false', help=help)
    parser.set_defaults(**{dest_name: default})


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('root', metavar='DIR',
                    help='path to dataset root')
parser.add_argument('--dataset', default='coco', type=str, metavar='DATASET',
                    help='Name of dataset (default: "coco"')
parser.add_argument('--split', default='val',
                    help='validation split')
parser.add_argument('--corruption_ids', default='1-2', type=str, help='corruption id in CocoC')
parser.add_argument('--corruption-severity', default=0, type=int, help='corruption severilty in CocoC')
parser.add_argument('--model', '-m', metavar='MODEL', default='tf_efficientdet_d1',
                    help='model architecture (default: tf_efficientdet_d1)')
add_bool_arg(parser, 'redundant-bias', default=None,
                    help='override model config for redundant bias layers')
add_bool_arg(parser, 'soft-nms', default=None, help='override model config for soft-nms')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='Override num_classes in model config if set. For fine-tuning from pretrained.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='bilinear', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--fill-color', default=None, type=str, metavar='NAME',
                    help='Image augmentation fill (background) color ("mean" or int)')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--results', default='', type=str, metavar='FILENAME',
                    help='JSON filename for evaluation results')
parser.add_argument('--load-adv', default=None, type=str, help='use bn_clean. when \
    load clean_bn, use bn_adv. when load adv_bn')
parser.add_argument('--attacker', default=None, type=str, help='make adversarial sample')
parser.add_argument('--visualize', default=0, type=int, help='number of images batch to visualize')

def validate(args):
    setup_default_logging()

    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    args.pretrained = args.pretrained or not args.checkpoint  # might as well try to validate something
    args.prefetcher = not args.no_prefetcher

    # create model
    with set_layer_config(scriptable=args.torchscript):
        extra_args = {}
        if args.img_size is not None:
            extra_args = dict(image_size=(args.img_size, args.img_size))
        if args.load_adv is not None:
            extra_args['load_adv']=args.load_adv

        bench = create_model(
            args.model,
            bench_task='predict',
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            redundant_bias=args.redundant_bias,
            soft_nms=args.soft_nms,
            checkpoint_path=args.checkpoint,
            checkpoint_ema=args.use_ema,
            **extra_args,
        )
    model_config = bench.config

    param_count = sum([m.numel() for m in bench.parameters()])
    print('Model %s created, param count: %d' % (args.model, param_count))

    bench = bench.cuda()

    amp_autocast = suppress
    if args.apex_amp:
        bench = amp.initialize(bench, opt_level='O1')
        print('Using NVIDIA APEX AMP. Validating in mixed precision.')
    elif args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        print('Using native Torch AMP. Validating in mixed precision.')
    else:
        print('AMP not enabled. Validating in float32.')

    if args.num_gpu > 1:
        bench = torch.nn.DataParallel(bench, device_ids=list(range(args.num_gpu)))
    print(args.corruption_ids)
    if args.corruption_ids:
        allCorruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
        corruptions=['None']
        if '-' in args.corruption_ids:
            corupt_start, corupt_end = args.corruption_ids.split('-')
            corupt_start, corupt_end=int(corupt_start), int(corupt_end)
            for corupt in allCorruptions[corupt_start:corupt_end+1]:
                corruptions.append(corupt)
        elif ',' in args.corruption_ids:
            for corrupt_id in args.corruption_ids.split(','):
                corruptions.append(allCorruptions[int(corrupt_id)])
    
    corruption_severity=int(args.corruption_severity)
    
    aggregated_results = {}
    for corr_i, corruption in enumerate(corruptions):
        aggregated_results[corruption] = {}
        
        dataset = create_dataset(args.dataset, args.root, args.split, \
            corruption, corruption_severity)
        input_config = resolve_input_config(args, model_config)

        loader = create_loader(
            dataset,
            input_size=input_config['input_size'],
            batch_size=args.batch_size,
            use_prefetcher=args.prefetcher,
            interpolation=input_config['interpolation'],
            fill_color=input_config['fill_color'],
            mean=input_config['mean'],
            std=input_config['std'],
            num_workers=args.workers,
            pin_mem=args.pin_mem,
            anchor_labeler=None)

        evaluator = create_evaluator(args.dataset, dataset, pred_yxyx=False)
        bench.eval()
        batch_time = AverageMeter()
        end = time.time()
        last_idx = len(loader) - 1
        
        for i, (input, target) in enumerate(loader):
            
            if i<args.visualize:
                os.makedirs('visualize', exist_ok=True)
                save_image(input, "visualize/ori_%s_img%d.png" % (corruption,i))
            with torch.no_grad():
                with amp_autocast():
                    output = bench(input, target)
                    
                evaluator.add_predictions(output, target)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.log_freq == 0 or i == last_idx:
                    print(
                        'Test: [{0:>4d}/{1}]  '
                        'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        .format(
                            i, len(loader), batch_time=batch_time,
                            rate_avg=input.size(0) / batch_time.avg)
                    )

        # mean_ap = 0.
        if dataset.parser.has_labels:
            metrics = evaluator.evaluate(output_result_file=args.results)
        else:
            evaluator.save(args.results)
        
        aggregated_results[corruption][corruption_severity] = metrics
    # np.save("robust_testres.npy", aggregated_results)
    
    results = np.zeros((len(corruptions), 6, len(metrics)), dtype='float32')
    for corr_i, distortion in enumerate(aggregated_results):
        for severity in aggregated_results[distortion]:
            for metric_j, metric_name in enumerate(metrics):
                mAP = aggregated_results[distortion][severity][metric_name]
                results[corr_i, severity, metric_j] = mAP
                #if verbose > 0:
                #    print(distortion, severity, mAP)
                    
    P = results[0,0,:]
    # benchmark val
    numCorp=len(corruptions)
    mPC = np.mean(results[:numCorp,1:,:], axis=(0,1))
    rPC = mPC/P
    print("Performance on Clean Data [P] (box)")
    print_coco_results(P)
    print("Mean Performance under Corruption [mPC] (box)")
    print_coco_results(mPC)
    print("Realtive Performance under Corruption [rPC] box")
    print_coco_results(rPC)
    
    return aggregated_results


def main():
    args = parser.parse_args()
    mps = validate(args)
    print(mps)


if __name__ == '__main__':
    main()

