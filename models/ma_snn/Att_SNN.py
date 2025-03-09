import torch
import torch.nn as nn
from .module.LIF_Module import AttLIF, ConvAttLIF
from torch import optim
from .Config import configs

__all__ = ['att_snn']

def create_net(config):
    # Net
    # define approximate firing function
    class ActFun(torch.autograd.Function):
        # Define approximate firing function

        def __init__(self):
            super(ActFun, self).__init__()

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return input.ge(0.0).float()

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            temp = abs(input) < config.lens
            return grad_output * temp.float() / (2 * config.lens)

    cfg_cnn = [
        (
            2,
            32,
            1,
            1,
            3,
        ),
        (
            32,
            64,
            1,
            1,
            3,
        ),
        (
            64,
            128,
            1,
            1,
            3,
        ),
        (
            128,
            256,
            1,
            1,
            3,
        ),
        (
            256,
            512,
            1,
            1,
            3,
        ),
    ]
    # pooling kernel_size
    cfg_pool = [2, 2, 2, 2, 4]
    # fc layer
    cfg_fc = [cfg_cnn[-1][1] * 3 * 3, 512, config.target_size]

    class Net(nn.Module):
        def __init__(
            self,
        ):
            super(Net, self).__init__()
            h, w = config.im_height, config.im_width
            in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
            pooling_kernel_size = cfg_pool[0]
            h, w = h // cfg_pool[0], h // cfg_pool[0]
            self.convAttLIF0 = ConvAttLIF(
                h=h,
                w=w,
                attention=config.attention,
                inputSize=in_planes,
                hiddenSize=out_planes,
                kernel_size=(kernel_size, kernel_size),
                spikeActFun=ActFun.apply,
                init_method=config.init_method,
                useBatchNorm=True,
                pooling_kernel_size=pooling_kernel_size,
                T=config.T,
                pa_dict={
                    "alpha": config.alpha,
                    "beta": config.beta,
                    "Vreset": config.Vreset,
                    "Vthres": config.Vthres,
                },
                reduction=config.reduction,
                track_running_stats=config.track_running_stats,
                mode_select=config.mode_select,
                mem_act=config.mem_act,
                TR_model=config.TR_model,
                c_ratio=config.c_ratio,
                t_ratio=config.t_ratio
            )
            in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
            pooling_kernel_size = cfg_pool[1]
            h, w = h // cfg_pool[1], h // cfg_pool[1]
            self.convAttLIF1 = ConvAttLIF(
                h=h,
                w=w,
                attention=config.attention,
                inputSize=in_planes,
                hiddenSize=out_planes,
                kernel_size=(kernel_size, kernel_size),
                spikeActFun=ActFun.apply,
                init_method=config.init_method,
                useBatchNorm=True,
                pooling_kernel_size=pooling_kernel_size,
                T=config.T,
                pa_dict={
                    "alpha": config.alpha,
                    "beta": config.beta,
                    "Vreset": config.Vreset,
                    "Vthres": config.Vthres,
                },
                reduction=config.reduction,
                track_running_stats=config.track_running_stats,
                mode_select=config.mode_select,
                mem_act=config.mem_act,
                TR_model=config.TR_model,
                c_ratio=config.c_ratio,
                t_ratio=config.t_ratio
            )
            in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[2]
            pooling_kernel_size = cfg_pool[2]
            h, w = h // cfg_pool[2], h // cfg_pool[2]
            self.convAttLIF2 = ConvAttLIF(
                h=h,
                w=w,
                attention=config.attention,
                inputSize=in_planes,
                hiddenSize=out_planes,
                kernel_size=(kernel_size, kernel_size),
                spikeActFun=ActFun.apply,
                init_method=config.init_method,
                useBatchNorm=True,
                pooling_kernel_size=pooling_kernel_size,
                T=config.T,
                pa_dict={
                    "alpha": config.alpha,
                    "beta": config.beta,
                    "Vreset": config.Vreset,
                    "Vthres": config.Vthres,
                },
                reduction=config.reduction,
                track_running_stats=config.track_running_stats,
                mode_select=config.mode_select,
                mem_act=config.mem_act,
                TR_model=config.TR_model,
                c_ratio=config.c_ratio,
                t_ratio=config.t_ratio
            )
            in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[3]
            pooling_kernel_size = cfg_pool[3]
            h, w = h // cfg_pool[3], h // cfg_pool[3]
            self.convAttLIF3 = ConvAttLIF(
                h=h,
                w=w,
                attention=config.attention,
                inputSize=in_planes,
                hiddenSize=out_planes,
                kernel_size=(kernel_size, kernel_size),
                spikeActFun=ActFun.apply,
                init_method=config.init_method,
                useBatchNorm=True,
                pooling_kernel_size=pooling_kernel_size,
                T=config.T,
                pa_dict={
                    "alpha": config.alpha,
                    "beta": config.beta,
                    "Vreset": config.Vreset,
                    "Vthres": config.Vthres,
                },
                reduction=config.reduction,
                track_running_stats=config.track_running_stats,
                mode_select=config.mode_select,
                mem_act=config.mem_act,
                TR_model=config.TR_model,
                c_ratio=config.c_ratio,
                t_ratio=config.t_ratio
            )
            in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[4]
            pooling_kernel_size = cfg_pool[4]
            h, w = h // cfg_pool[4], h // cfg_pool[4]
            self.convAttLIF4 = ConvAttLIF(
                h=h,
                w=w,
                attention=config.attention,
                inputSize=in_planes,
                hiddenSize=out_planes,
                kernel_size=(kernel_size, kernel_size),
                spikeActFun=ActFun.apply,
                init_method=config.init_method,
                useBatchNorm=True,
                pooling_kernel_size=pooling_kernel_size,
                T=config.T,
                pa_dict={
                    "alpha": config.alpha,
                    "beta": config.beta,
                    "Vreset": config.Vreset,
                    "Vthres": config.Vthres,
                },
                reduction=config.reduction,
                track_running_stats=config.track_running_stats,
                mode_select=config.mode_select,
                mem_act=config.mem_act,
                TR_model=config.TR_model,
                c_ratio=config.c_ratio,
                t_ratio=config.t_ratio
            )
            self.FC0 = AttLIF(
                attention="no"
                if config.attention in ["no", "CA", "SA", "CSA"]
                else "TA",
                inputSize=cfg_fc[0],
                hiddenSize=cfg_fc[1],
                spikeActFun=ActFun.apply,
                useBatchNorm=True,
                T=config.T,
                pa_dict={
                    "alpha": config.alpha,
                    "beta": config.beta,
                    "Vreset": config.Vreset,
                    "Vthres": config.Vthres,
                },
                reduction=config.reduction,
                track_running_stats=config.track_running_stats,
                mode_select=config.mode_select,
                mem_act=config.mem_act,
                TR_model=config.TR_model,
                t_ratio=config.t_ratio
            )
            self.FC1 = AttLIF(
                attention="no"
                if config.attention in ["no", "CA", "SA", "CSA"]
                else "TA",
                inputSize=cfg_fc[1],
                hiddenSize=cfg_fc[2],
                spikeActFun=ActFun.apply,
                useBatchNorm=True,
                T=config.T,
                pa_dict={
                    "alpha": config.alpha,
                    "beta": config.beta,
                    "Vreset": config.Vreset,
                    "Vthres": config.Vthres,
                },
                reduction=config.reduction,
                track_running_stats=config.track_running_stats,
                mode_select=config.mode_select,
                mem_act=config.mem_act,
                TR_model=config.TR_model,
                t_ratio=config.t_ratio
            )

        def forward(self, input):
            torch.cuda.empty_cache()
            input = input.permute(1, 0, 2, 3, 4)
            b, t, c, h, w = input.size()
            if h != 224 or w != 224:
                input = input.reshape(b * t, c, h, w)
                input = nn.functional.interpolate(input, size=(224, 224), mode="bilinear")
                input = input.reshape(b, t, c, 224, 224)
            
            outputs = input

            outputs = self.convAttLIF0(outputs)
            outputs = self.convAttLIF1(outputs)
            outputs = self.convAttLIF2(outputs)
            outputs = self.convAttLIF3(outputs)
            outputs = self.convAttLIF4(outputs)

            outputs = outputs.reshape(b, t, -1)
            outputs = self.FC0(outputs)

            self.feature = outputs.permute(1, 0, 2)  # (b, t, c)

            outputs = self.FC1(outputs)

            outputs = torch.sum(outputs, dim=1)
            outputs = outputs / t

            return outputs

    return Net()


def att_snn(num_classes=1000, T=8):
    config = configs()
    config.target_size = num_classes
    config.T = T
    model = create_net(config)
    return model



if __name__ == '__main__':
    model = att_snn()
    model.cuda()
    x = torch.randn(2, 8, 2, 224, 224)
    x = x.cuda()
    y = model(x)
    loss = y.mean()
    loss.backward()
    print(y.size())
    print('done')