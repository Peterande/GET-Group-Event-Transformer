# ==============================================================================
# Prefetch data during training process.
# ==============================================================================

import torch
from event_based.event_token import E2SRC


def to_cuda(samples, targets, e2s):
    samples = e2s([event.cuda(non_blocking=True) for event in samples])
    targets = targets.cuda(non_blocking=True)
    return samples, targets

class data_prefetcher():
    def __init__(self, loader, device, prefetch=True, bs=1, embed_split=12, patch_size=4):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch and self.device=='cuda':
            self.stream = torch.cuda.Stream()
            self.preload()
        self.e2s = E2SRC([128, 128], batch_size=bs, group_num=embed_split, patch_size=patch_size)

    def preload(self):
        try:
            self.next_samples, self.next_targets = next(self.loader)[0:2]
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_samples, self.next_targets = to_cuda(self.next_samples, self.next_targets, self.device)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:

    def next(self):
        if self.prefetch and self.device=='cuda':
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            if samples is not None:
                samples.record_stream(torch.cuda.current_stream())
            if targets is not None:
                for t in targets:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            try:
                samples, targets = next(self.loader)[0:2]
                samples, targets = to_cuda(samples, targets, self.e2s)
            except StopIteration:
                samples = None
                targets = None
        return samples, targets
