import torch
from torch.optim import Optimizer
from typing import Callable, Iterable, Tuple
from torch.distributions.bernoulli import Bernoulli
import math
import numpy as np

class ChildTuningAdamW(Optimizer):
    def __init__(
        self,
        model,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        reserve_p=1.0,
        mode="ChildTuning-D",  ## 改这个地方确定是任务相关 'ChildTuning-D' 还是任务无关 ChildTuning-F
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1])
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

        self.gradient_mask = dict()
        for name, params in model.named_parameters():
            if 'layer' in name:
                self.gradient_mask[params] = params.new_zeros(params.size())
        self.reserve_p = reserve_p
        self.mode = mode
        self.N = 1
        self.D_start = False
        try:
            gradient_mask = np.load('gradient_mask.npy')
            self.gradient_mask = gradient_mask
            self.D_start = True
        except:
            pass

    def set_gradient_mask(self, gradient_mask):
        self.gradient_mask = gradient_mask

    def epoch_start(self):
        self.N = 1

    def epoch_end(self):

        if self.D_start == False:
            r = None
            for k, v in self.gradient_mask.items():
                v = v.view(-1).cpu().numpy()
                if r is None:
                    r = v
                else:
                    r = np.append(r, v)
            polar = np.percentile(r, (1 - self.reserve_p) * 100)
            for k in self.gradient_mask:
                self.gradient_mask[k] = self.gradient_mask[k] >= polar
            np.save('gradient_mask.npy',self.gradient_mask)
        self.D_start = True
    def step(self, model, closure: Callable = None):
        """
        Performs a single optimization step.
        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        self.N += 1
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                # =================== HACK BEGIN =======================
                if self.mode is not None:
                    if self.mode == "ChildTuning-D":

                        if self.D_start == False:
                            for name, params in model.named_parameters():

                                if 'layer' in name:
                                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                                    self.gradient_mask[params] += (params.grad ** 2) / self.N
                            # grad_mask = Bernoulli(
                            #     grad.new_full(size=grad.size(), fill_value=self.reserve_p)
                            # )
                            # grad *= grad_mask.sample() / self.reserve_p
                        else:

                            if p in self.gradient_mask:
                                grad *= self.gradient_mask[p]
                    else:
                        # ChildTuning-F
                        grad_mask = Bernoulli(
                            grad.new_full(size=grad.size(), fill_value=self.reserve_p)
                        )
                        grad *= grad_mask.sample() / self.reserve_p

                # =================== HACK END =======================

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = (
                        step_size * math.sqrt(bias_correction2) / bias_correction1
                    )

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss


    

   


    # def calculate_fisher(self):
    #     """
    #     Calculate Fisher Information for different parameters
    #     """
    #     gradient_mask = dict()
    #     model = self.model
    #     model.train()

    #     for name, params in model.named_parameters():
    #         if "layer" in name:
    #             gradient_mask[params] = params.new_zeros(params.size())

    #     # Now begin
    #     train_dataloader = DataLoader(
    #         self.train_dataset,
    #         batch_size=self.args.per_device_train_batch_size,
    #         shuffle=True,
    #         collate_fn=self.data_collator,
    #         drop_last=self.args.dataloader_drop_last,
    #         num_workers=self.args.dataloader_num_workers,
    #         pin_memory=self.args.dataloader_pin_memory,
    #     )

    #     N = len(train_dataloader)

    #     for inputs in tqdm(train_dataloader):
    #         inputs.pop("idx")
    #         inputs = self._prepare_inputs(inputs)
    #         outputs = model(**inputs)
    #         loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    #         loss.backward()

    #         for name, params in model.named_parameters():
    #             if "layer" in name:
    #                 torch.nn.utils.clip_grad_norm_(params, self.args.max_grad_norm)
    #                 gradient_mask[params] += (params.grad ** 2) / N
    #         model.zero_grad()

    #     print("Calculate Fisher Information")

    #     # Numpy
    #     r = None
    #     for k, v in gradient_mask.items():
    #         v = v.view(-1).cpu().numpy()
    #         if r is None:
    #             r = v
    #         else:
    #             r = np.append(r, v)
    #     polar = np.percentile(r, (1 - self.reserve_p) * 100)
    #     for k in gradient_mask:
    #         gradient_mask[k] = gradient_mask[k] >= polar
    #     print("Polar => {}".format(polar))

    #     # TODO: pytorch: torch.kthvalue

    #     return gradient_mask
