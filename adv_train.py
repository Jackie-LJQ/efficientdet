from attacks import *
from timm.utils import *
from contextlib import suppress
from attacks import AttackerBuilder
import time
from utils import get_clip_parameters
import torch
import logging
from collections import OrderedDict
from utils import set_advState
    
def adv_train_epoch(
        epoch, model, loader, optimizer, args, attackTarget,
        lr_scheduler=None, saver=None, output_dir='',  
        amp_autocast=suppress, loss_scaler=None, model_ema=None):

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    attacker = AttackerBuilder(args.attacker)
    clip_params = get_clip_parameters(model, exclude_head='agc' in args.clip_mode)
    end = time.time()
    batch_size = args.batch_size * args.world_size
    last_idx = len(loader) // batch_size * batch_size - 1
    num_updates = epoch * len(loader)
    # _, attackTarget = next(iter(loader))
    
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        # generate adversarial images:
        model.eval()
        img_adv = attacker.attack(model=model, x=input, gtlabels=target, targets=attackTarget)
        model.train()
        with amp_autocast():
            adv_loss = model(img_adv, target)["loss"]
        set_advState(model, 1) # True for clean sample
        clean_loss = model(input, target)["loss"]        
        loss = clean_loss + adv_loss
        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode, parameters=clip_params)
        else:
            loss.backward()
            if args.clip_grad is not None:
                dispatch_clip_grad(clip_params, value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                logging.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        if last_batch: break
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])
