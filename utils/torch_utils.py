# YOLOv5 PyTorch utils

import datetime
import logging
import math
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None
logger = logging.getLogger(__name__)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def date_modified(path=__file__):
    # return human-readable file modification date, i.e. '2021-3-26' 返回可读的文件修改日期
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime) # 根据时间戳获取日期信息
    return f'{t.year}-{t.month}-{t.day}'


def git_describe(path=Path(__file__).parent):  # path must be a directory
    # return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    s = f'git -C {path} describe --tags --long --always' #
    # fatal(致命的): cannot change to 'D:\anaconda3_python37\Github\yolov5-5.0\utils\torch_utils.py': Invalid argument

    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e: # s这个命令出现错误(path路径下可能不是脚本文件而是不可执行文件，比如文件夹)
        return ''  # not a git repository


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'YOLOv5 🚀 {git_describe() or date_modified()} torch {torch.__version__} '  # string
    # date_modified()返回当前的年月日，git_describe()不懂？
    cpu = device.lower() == 'cpu' # 将device所代表的GPU或者CPU设备名称转变为小写,cpu指代的是(dev)
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable (GPU index)
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available() # 若cuda=True,说明指定的是GPU;如果为cuda=False,
    if cuda:
        n = torch.cuda.device_count() # GPU个数，由于调用了torch.cuda,不会对cpu计数
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        # space = ' ' * len(s)
        space=" "
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i) # 获取GPU设备的属性信息
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1024 ** 2)}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    logger.info(s.encode().decode(encoding='utf-8', errors='ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    # errors="ignore"解码过程中忽略非法字符
    # 由于在general.py函数中set_logging的basicConfig已经设置了logging的输出日志的等级为logging.info，
    # set_logging()函数位于select_device之前,设置为logging.INFO等级输出
    return torch.device('cuda:0' if cuda else 'cpu') # 指定了GPU，设备为device指定的第0块GPU


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize() # 保证在gpu上的操作都完成才进行下一步(等待当前设备上所有流中的所有核心完成)
    return time.time()


def profile(x, ops, n=100, device=None):
    # profile a pytorch module or list of modules. Example usage:
    #     x = torch.randn(16, 3, 640, 640)  # input
    #     m1 = lambda x: x * torch.sigmoid(x)
    #     m2 = nn.SiLU()
    #     profile(x, [m1, m2], n=100)  # profile speed over 100 iterations

    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    x.requires_grad = True
    print(torch.__version__, device.type, torch.cuda.get_device_properties(0) if device.type == 'cuda' else '')
    print(f"\n{'Params':>12s}{'GFLOPS':>12s}{'forward (ms)':>16s}{'backward (ms)':>16s}{'input':>24s}{'output':>24s}")
    for m in ops if isinstance(ops, list) else [ops]:
        m = m.to(device) if hasattr(m, 'to') else m  # device
        m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m  # type
        dtf, dtb, t = 0., 0., [0., 0., 0.]  # dt forward, backward
        try:
            flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # GFLOPS
        except:
            flops = 0

        for _ in range(n):
            t[0] = time_synchronized()
            y = m(x)
            t[1] = time_synchronized()
            try:
                _ = y.sum().backward()
                t[2] = time_synchronized()
            except:  # no backward method
                t[2] = float('nan')
            dtf += (t[1] - t[0]) * 1000 / n  # ms per op forward
            dtb += (t[2] - t[1]) * 1000 / n  # ms per op backward

        s_in = tuple(x.shape) if isinstance(x, torch.Tensor) else 'list'
        s_out = tuple(y.shape) if isinstance(y, torch.Tensor) else 'list'
        p = sum(list(x.numel() for x in m.parameters())) if isinstance(m, nn.Module) else 0  # parameters
        print(f'{p:12}{flops:12.4g}{dtf:16.4g}{dtb:16.4g}{str(s_in):>24s}{str(s_out):>24s}')


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    # Finds layer indices matching module class 'mclass'
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def sparsity(model):
    # Return global model sparsity
    a, b = 0., 0.
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune(model, amount=0.3):
    # Prune model to requested global sparsity
    import torch.nn.utils.prune as prune
    print('Pruning model... ', end='')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            prune.remove(m, 'weight')  # make permanent
    print(' %.3g global sparsity' % sparsity(model))


def fuse_conv_and_bn(conv, bn):
    # conv的bias是否为False在yolov5s.yaml文件中已设置
    # 寻找conv
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)
    # conv.weight.device="cuda:0"
    # 设置requires_grad_，冻结某一层的梯度，使其不更新
    # conv.weight.device卷积层的权重所在的位置(cpu or cuda)

    # prepare filters
    ## conv.weight.shape=torch.Size([out_channels,in_channels,kernel_size,kernel_size])
    w_conv = conv.weight.clone().view(conv.out_channels, -1) #
    # Conv的输出通道数=卷积核个数
    # w_conv每一行代表一个卷积核感受野的参数
    # w_conv.shape=torch.Size(out_channels,in_channels*height*width),输出的每一个通道的参数为1行
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    # torch.diag将对象Tensor作为主对角线上的元素，div作用是bn.weight的每个元素除以(eps+var)
    # torch.diag的目的是防止w_bn在与w.bn，不同channel之间的权重做相加运算(相乘后相加)，
    # bn.weight是每个channel一个，
    # bn.running_var维度是(1,out_channels) bn.eps+bn.running_var(80,)使用了广播功能（bn.eps:default=1e-5是1个数）
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    # 本质上而言,BatchNorm2d(channels)在卷积层后作用后，此时融合Conv和BatchNorm2d后的权重shape没有发生变化，
    # 因为BatchaNorm2d只是在Conv作用的图片上各层进行基于均值方差的归一化，对融合后的卷积层和归一化层权重shape是没有影响的
    # fusedconv的权重就是在卷积conv的权重上以channel为单位各乘上一个数
    # torch.mm(a,b)=矩阵a*矩阵b
    # prepare spatial bias
    # 卷积的偏置如果没有,人为设置bias=0矩阵;bias偏置尺寸是(输出通道数,)，和conv.weight.shape第一个数是一样的
    # fusedconv.weight.shape=conv.weight.shape
    # 输出三个通道每个通道有一个bias,每个通道的权重尺寸是k*k的
    # 归一化实在每个通道上进行的，每个通道上的权重(gamma)和偏置(bias)是一样的,
    # bn.bias.shape=bn.weight.shape=torch.Size(1,out_channels)
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    # 在yolov5s.pt模型结构中的卷积层中bias均为False!
    # torch.zeros(n) 返回1行n列的0元素
    # 卷积的偏置如果没有,人为设置bias=0矩阵;bias偏置尺寸是(1,输出通道数)
    # 偏置不会影响尺寸
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn) # 得到的结果为行向量
    # reshape(-1)将得到的列向量转变为行向量
    # a.copy_(b)将b复制给a
    # fusedconv.bias也是行向量
    return fusedconv
    # fusedconv.weight.shape=conv.weight.shape
    # fusedconv.bias.shape=fusedconv.bias.shape
    # 融合Conv层和BatchNorm2d的fusedconv的权重和偏置尺寸是一样的



def model_info(model, verbose=False, img_size=640):
    # 打印模型信息 self.model
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters 获取每一层的参数个数并求和
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients 计算梯度个数
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        # 8,16，32 最大值为32
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # input
        # 依据torch.zeros(1,ch,stride,stride)输入到模型中计算参数量;
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPS # 有接口能够跳转到model.forward,说明这是通过运行模型的，由于主要目的是计算FLOPs每秒的运算次数，一般选用满足网络最小的img作为输入
        # 每秒执行flops次浮点运算 flops是floating-point operations per second
        # GFLOPS=1*10^9 FLOPS 每秒计算多少个10亿的浮点运算数
        # 由于设置了logging。info,所以
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        # 将img_size统一成list类型
        fs = ', %.1f GFLOPS' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPS
        # fs计算的是什么
    except (ImportError, Exception):
        fs = ''

    logger.info(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")
    # 输出信息INFO: Model Summary: 224 layers, 7266973 parameters, 0 gradients, 17.0 GFLOPS
    # 在general.py设置日志输出等级logging.INFO,只输出INFO以上等级的信息
    ## 我觉得len(list(model.modules()))算出来的不是模型层数

def load_classifier(name='resnet101', n=2):
    # Loads a pretrained model reshaped to n-class output
    model = torchvision.models.__dict__[name](pretrained=True) #

    # ResNet model properties
    # input_size = [3, 224, 224]
    # input_space = 'RGB'
    # input_range = [0, 1]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # Reshape output to n classes
    # y=xAT+bias
    filters = model.fc.weight.shape[1]  #
    model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
    model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    model.fc.out_features = n
    return model


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # scales img(batchsize,3,y,x) by ratio constrained to gs-multiple
    # 将img按照比例
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        # 通过双线性插值进行重采样，interpoloate和resize区别有吗？
        # align_corners
        if not same_shape:  # pad/crop img
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)] # (32,32)
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean
        # 对最后两个维度进行扩充,保证重采样后，图像的shape不发生改变
        # value是扩充时指定的填充值，只能在mode="constrant"(pad默认)指定

def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
