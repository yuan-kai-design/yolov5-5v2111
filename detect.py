import argparse
import time
from pathlib import Path
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(save_img=False):
    # 支持视频和图像同时进行检测
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    # opt是在if __name__=="__main__"(*)下的opt = parser.parse_args(),在(*)下默认为全局变量
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference train
    # save_img本质上是bool类型
    # save_img是否保存图片，有两点需要判断：1.是否要保存，2.保存的是不是图片
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))  # 对在线资源进行目标检测

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # save_dir="D:\anaconda3_python37\Github\yolov5-5.0\runs\detect\exp14"
    # opt.project=runs\detect
    (save_dir / 'train' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # 指定save_txt=True(生成的box信息，类别数)参数，则创建run/detect/exp{n}/train/,否则创建run/detect/exp{n}/,如果父目录不存在，则上级目录一起创建
    # Initialize
    set_logging() # 设置输出日志(日志输出格式，日志等级)，set_logging只包含logging.BasicConfig(formatter,)
    device = select_device(opt.device) # 选择GPU或者CPU
    half = device.type != 'cpu'  # half precision only supported on CUDA只有CUDA支持半精度

    # Load model
    # 加载权重pt模型
    model = attempt_load(weights, map_location=device)  # load FP32 model
    # 先构建了一个继承nn.Modellist的Ensemble类,并实例化model;通过加载
    # 传入的weights是yolo5.pt模型
    # print(list(model.named_parameters()))
    stride = int(model.stride.max())  # model stride # 三个特征图的缩放比例，log2(stride)为三个特征图对应的下采样次数
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    # 如果imgsz不能被s整除，会弹出警告信息(imgsz不是model.stride.max()的倍数--更新为imgsz) # 向上取整，避免在最后Detect()输出特征
    if half:
        model.half()  # to FP16
        # 将模型中浮点参数和缓冲区强制转换为"半"数据类型，小数点保留位数减少一半
        # 将原来的FP32，half一下变为FP16,但我发现运行过之后，精度还是FP64

    # Second-stage classifier
    # classify = True
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        # 初始化模型分类器参数fc.weight,fc.bias,fc.out_features;n为分类个数
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
# resnet101.pt中好像没有“model”

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names # 分类的类别种类名称
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # 每一个类别设置三种颜色，colors的颜色索引与names中的类别索引号是一一对应的，注意两个"_"指向的对象是不一样的
    # Run inference
    if device.type != 'cpu': # device=0 （cuda:0）
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once；
        # 由于之前的model.half()后model内的所有参数类型都是半精度(weights:yolov5.pt->float16)，
        # 需要先将input转换为model.parameters下相同的数据类型，
        # 考虑到model.paramters是包含参数信息的迭代器，且所有参数类型均为半精度(torch.cuda.Halfcuda)
        # 利用next(model.parameters())提取迭代生成器的第一个元素(第一层参数)
        # 利用type_as将input的数据类型转化为与next(model.parameters())相同的半精度类型,才能输入到参数类型为半精度的model中

        # 但是该语句应该是有返回值的，即必须要有变量去接受它，难道说这句话仅仅只是用来转化类型的，用model只是检验转化的类型能不能传入模型
    t0 = time.time() #
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device) # 将numpy.ndarray转化为tensor
        img = img.half() if half else img.float()  # uint8 to fp16/32
        # 在GPU上运行，img.half()
        # 在CPU上运行，img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0 归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0) # img.shape=torch.Size([1,c=3,h=480,w=640])
            # 在img矩阵位置shape索引为0的位置添加维度为1的维度,添加维度是为了能让图片能够放入model()中运行

        # Inference
        t1 = time_synchronized()  # time.time()
        pred = model(img, augment=opt.augment)[0] # 返回
        # pred.shape=(1,3个不同尺度的特征图生成的预测框总数(每个像元生成3个anchor),3)
        # Apply NMS (into pred) # 对预测的结果应用非极大值抑制，在
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        # opt.classes=False
        # opt.agnostic_nms增强型非极大值抑制
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0) # 赋值im0=im0s 输入进来的检测图像矩阵，p是检测的每一张图片路径

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg # Path(path).name为带扩展名的图片名 save_path=runs/detect/exp*
            # save_path="runs/detect/exp{\d}"
            txt_path = str(save_dir / 'train' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string # %g%gx%g 后面加上一个空格
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det): # 一张图片一个det,所以det[:,:4]=(x_c,y_c,w,h),im0是原图尺寸大小
                # Rescale boxes from img_size to im0 size,
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # img.shape=(1,3,480,640);img.shape[2:]=(480,640)
                # Print results
                for c in det[:, -1].unique(): # 去除重复项，
                    n = (det[:, -1] == c).sum()  # detections per class # 统计一张图片上每个类别出现的个数
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string # 通过(n>1)决定是否要加上's',表示复数

                # Write results
                for *xyxy, conf, cls in reversed(det): # reversed对det进行翻转
                    if save_txt:  # Write to file # xyxy
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('train/*.txt')))} train saved to {save_dir / 'train'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

## 可以实现自动加载pt权重文件(default指定的pt路径下不存在时，可以自动加载)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 创建一个命令行解析器，通过add_argument添加可英语调用的命令行参数，例如weights/source
    parser.add_argument('--weights', nargs='+', type=str,
                        # default=['params_pt/yolov5x.pt','params_pt/yolov5l.pt'],
                        default="params_pt/yolov5l.pt",
                        help='model.pt path(s)')
    # nargs="+"指要求至少输入1个或者多个字,default本质上为一个普通pt文件路径
    parser.add_argument('--source', type=str, default='data/images/IMG_20220306.jpg', help='source')
    # file(image,video)/folder, 0 for webcam（要进行目标检测的资源图片或者视频(由许多帧图像融合)）
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # 任何物体图像传入网络需要被resize成的尺寸：640*640(默认)，输入图片和输出图片的尺寸是一致的，但是网络训练过程中可能会变化
    parser.add_argument('--conf-thres', type=float,
                        # default=0.25,
                        default=0.2, # 如果默认值没有给定，程序中或给定默认值
                        help='object confidence threshold') # 置信度阈值
    # 置信度，只有检测出的置信度大于0.15，才会认为这是一个物体；置信度越大，可以识别出的物体就越少；只有在非常确信的情况下，才能将置信度设置得很大，例如0.8+
    # 置信度阈值的确实不是固定的，需要根据实际情况改变
    parser.add_argument('--iou-thres', type=float, default=0.25, help='IOU threshold for NMS')  # IOU交并比
    # 两个框的IOU如果小于指定的阈值，那么两个框都保留，因为此时会认为这是2个物体；
    # 反之，如果IOU大于指定的阈值，那么从两个框中选择一个框，因为此时认为是1个物体。如何从两个框中进行选择算法需要考虑
    # NMS(Non maximum supression非最大值抑制)
    parser.add_argument('--device', default="0", help='cuda device, i.e. 0 or 0,1,2,3 or cpu') # 指定GPU
    # 这里default设置为NULL，后面的程序会自动对GPU进行检测，也可以对其进行设置为0，或者cpu；指定多个GPU，需要用逗号隔开
    parser.add_argument('--view-img', action='store_true', help='display results')
    # 设定view-img=True,指定是否显示图片(一直显示)，对于action='store_true'那么只要设置了参数view-img，那么就认为view-img=True(bool类型),是否在程序运行过程中显示结果，如果没有设置view-img,那么默认view-img=False
    # 如果以命令行python detect.py --view-img,默认view_img=true,中间的横线"-"在程序运行时转为“_”
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # 保存YOLOV5目标检测的结果
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt train')
    parser.add_argument('--nosave', action='store_true', help='do not save train/videos')
    # 不保存图像images/videos
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    # 只查看指定的类别，假如设置了--classes 0,那么生成的结果中只会出现0类别（第0个类别是人）
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # 增强型NMS(Non maximum supression非极大值抑制)
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # 是否数据结果进行增强，表现：目标物体的置信度增加，具体如何实现需要掌握
    parser.add_argument('--update', action='store_true', help='update all models')
    # update的作用类似于初始化模型，让模型处于最简形式，将模型中的优化器这些附加的东西全部设置为None,只留下模型必要参数
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    # 设置运行结果的保存位置
    parser.add_argument('--name', default='exp', help='save results to project/name')
    # 设置在project参数所指定的保存路径下存放结果的文件夹runs/detect/exp,
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # 一般不设置--exist-ok,即(--exist-ok=False),这样做的目的是为了区分每一次运行得到的结果
    # 如果设置了这个参数，那么程序运行的结果会始终保存在project/name（name指定的文件夹下）,即project/exp
    ## 一般在不给定参数就设为True的变量(必须的)采用default模式 e.g.weights,source,device,project,name.不是必须的采用action模式(添砖加瓦)
    opt = parser.parse_args()
    # 要想查看参数的默认值，只需要在opt前设置断点，点击debug查看相关属性
    print("opt.update=",opt.update)
    print(opt) # 打印Namespace类
    check_requirements(exclude=('pycocotools', 'thop'))
    # 检查requirements.txt下除了pycocotools和thop下其它指定版本的库有没有安装，没有自动安装
    # torch.no_grad()在保证模型参数梯度不更新的情况下进行检测
    with torch.no_grad(): # 以下过程不会进行反向传播计算，只能用于检测现有模型的效果
        if opt.update:  # update all models (to fix SourceChangeWarning)
            # 检测parser设置的参数有没有发生改变
            for opt.weights in ['yolov5m.pt','yolov5s.pt','yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
