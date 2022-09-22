# YOLOv5 YOLO-specific modules

import argparse
import logging
import sys
from copy import deepcopy

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor # 每个边界框由(p,x,y,w,h,c) p是预测概率,c=80,除了输出80这个类总数，还需要输出p,x,y,w,h这5个参数
        self.nl = len(anchors)  # number of detection layers # 3
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        # 采用register_buffer方法定义anchor
        # 在detect阶段不需要使用self.anchors
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        # self.anchor_grid存放的是yolov5l.yaml中的anchor先验框,原始的是三个尺度，每个尺度有预设三个尺寸
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        # 个人感觉ch=(256,256,256)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        # self.training与self.export做或运算符，将结果赋值给self.training
        for i in range(self.nl):
            # x=(1,3,img.shape[0]/8,img.shape[1]/8,85)
            x[i] = self.m[i](x[i])
            # conv # 通过m[i],x[i]实现一个特征层与对应卷积的操作
            bs, _, ny, nx = x[i].shape
            # x(bs,255,20,20) to x(bs,3,20,20,85) # bs是batchsize
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # self.na* # contiguous将Tensor在内存中的分布变成连续的

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    # self.grid初始化网格是80*80*2,40*40*2,20*20*2，每个网格有x,y行列号
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                    # x[i].device="cuda:0"
                # x[i].shape=(1,3,ny,nx,85)
                y = x[i].sigmoid()

                # x[i][...,5]获取目标框的预测概率
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                # xy # y[...,0:2]表示预测框的中心坐标；
                # self.stride分别表示3次，4次，5次下采样的缩小倍数,要从特征图上恢复对原图的大小
                # self.grid在最后输出的特征图
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                # (y[...,2:4]*2)可以理解为缩放系数,均小于1
                # wh
                # self.anchor_grid先验框，通过k-means+Genetic自动生成
                # y.shape=(1,3,nx,ny,4+1+num_classes)
                # *self.anchor_grid[i]在计算过程中使用了矩阵广播(相当于在立方体中广播),self.anchor_grid[i].shape=(1,3,1,1,2)
                z.append(y.view(bs, -1, self.no))
                # -1指代的是每个特征层预测框的个数=ny*nx*3；
                # 每个预测框对应self.no个结果值，4个尺度位置参数（tx,ty,tw,th）+1个obj_conf+80个类别置信度

        return x if self.training else (torch.cat(z, 1), x)
    # x=(不同尺度下的torch.size(1,3,ny,nx,85))

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)]) # 将图像划分为单元网格，和图像坐标系的读取数据的方式相对应
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        # cfg没有指定时采用'yolov5s.yaml'
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
            # 大概率将'yolov5s.yaml'识别为str
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
                # 给Model设置yaml属性

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels,yolov5s中nc=80
        # yoluv5s.pt中没有设置ch属性，我理解是模型的通道数需要和输入图像
        if nc and nc != self.yaml['nc']:
            # 我原来以为nc=None 执行到此时Model类中还没有设置nc属性
            # 这句话是要判断如果nc存在但是与models/yaml模型结果中的nc是否相同，
            # 类实例化的nc一定与models/yaml文件中的nc相同，主要判断nc有没有设置(None)
            # 没有设置，就用具体数字覆盖None
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors: # yolov5s.pt中没有设置anchors(锚框)参数，anchors=None这里默认False
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        # 设置ckpt下Model类对象下的model信息(主要是模型结构信息，各层的属性)，和savelist,这个savelist下是什么
        # Model类下的model属性存放的是yolov5s.yaml模型结构信息
        # 如果是已经生成的模型比如yolov5s.pt(experimental/attempt_load.py)
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # 从这句话可以判断前面的nc已经设置过属性了nc=80，
        # 这样设置：self.names=["0",..,"79"]
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        # yolox5s.pt中是(24): Detect(
        #       (m): ModuleList(
        #         (0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
        #         (1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
        #         (2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
        #       )
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            # s=256是固定的
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward(torch.zeros(1,3,256,256))
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        # x的类型是torch.tensor
        if augment:
            img_size = x.shape[-2:]  # height, width 从倒数第二个元素开始到最后一个元素，即只取最后一个
            s = [1, 0.83, 0.67]  # scales 缩放系数
            f = [None, 3, None]  # flips (2-ud-height方向进行翻转, 3-lr宽度方向进行饭庄)
            # 翻转 2是上下翻转(以第2个维度，start:0)，3是左右翻转,None不翻转
            y = []  # outputs
            for si, fi in zip(s, f): # [(1,None),(0.83,3),(0.67,None)]
                xi = scale_img(x.flip(fi) if fi else x, # 对翻转后的图像进行缩小
                               si, # ratio
                               gs=int(self.stride.max())) # scale+pad
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale 4是间隔
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model: # self.model中的conv(fusedconv)经过了Conv和bn的fuse，并将原来的bn删除
            if m.f != -1:  # if not from previous layer，如果这一层是有来自上一次,一定具有m.f=-1
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                # yolov5l.pt的Concat()将x复制了一份，Concat()的f属性是[-1,6] x=[y[-1],y[6]] y[6]是第第2个C3后的Conv的输出，j=-1对应输出的x是此时最后一个操作Upsample的对应输出
            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run，调用common.py下Focus类的foward
            y.append(x if m.i in self.save else None)  # save output # m.i是layer的索引
            # 保存指定层的输出结果，要输出的层的索引在self.save,不在self.save指定直接输出None
        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.train, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights
    ## Model(nn.Module)
    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():# 首先取出第一个Sequentail
            if type(m) is Conv and hasattr(m, 'bn'): # 判断是否为包含BN功能的卷积层
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                # 将卷积层转变为融合了原有卷积和归一化层的功能层
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
                # type(m.fuseforward)=<class 'method'>
        self.info()  # 打印模型信息
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size) # self就是模型对象


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params_pt', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    # depth_multiple模型深度缩放倍数
    # width_multiple模型宽度缩放倍数
    # yolov5x.yaml和yolov5s.yaml只有上面两个会不一样
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    # 锚框的中一个列表的元素个数的二分之一等于锚框的个数
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    # no = 3*(80+5)=255

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3, C3TR]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params_pt
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params_pt
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
