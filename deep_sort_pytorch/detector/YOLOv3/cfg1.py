import torch
from yolo_utils1 import convert2cpu


def parse_cfg(cfgfile):
    blocks = []
    fp = open(cfgfile)
    # print(fp)  # <_io.TextIOWrapper name='cfg/yolo.cfg' mode='r' encoding='cp936'>
    block = None
    line = fp.readline()
    # print(line)  # [net]
    while line != '':
        line = line.rstrip()
        # print(line)
        if line == '' or line[0] == '#':
            # print(line)
            line = fp.readline()  # line == '' or line[0] == '#' 的下一行
            # print(line)
            continue  # 继续下一次循环
        elif line[0] == '[':
            if block:
                blocks.append(block)
            block = dict()
            block['type'] = line.lstrip('[').rstrip(']')
            # print(block)
            # set default value
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
                # print(block)
        else:
            # print(line)
            key, value = line.split('=')
            # print(key)
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
        line = fp.readline()
        # print(line)
        # print(block)

    if block:
        """
        block:{'type': 'region', 'anchors': '0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828', 
        'bias_match': '1', 'classes': '80', 'coords': '4', 'num': '5', 'softmax': '1', 'jitter': '.3', 'rescore': '1', 
        'object_scale': '5', 'noobject_scale': '1', 'class_scale': '1', 'coord_scale': '1', 'absolute': '1', 'thresh': '.6', 'random': '1'}
        
        blocks:[{'type': 'net', 'batch': '1', 'subdivisions': '1', 'width': '416', 'height': '416', 'channels': '3', 'momentum': '0.9', 'decay': '0.0005', 'angle': '0', 'saturation': '1.5', 'exposure': '1.5', 'hue': '.1', 'learning_rate': '0.001', 'burn_in': '1000', 'max_batches': '500200', 'policy': 'steps', 'steps': '400000,450000', 'scales': '.1,.1'},
        {'type': 'convolutional', 'batch_normalize': '1', 'filters': '32', 'size': '3', 'stride': '1', 'pad': '1', 'activation': 'leaky'}, 
        {'type': 'maxpool', 'size': '2', 'stride': '2'}, {'type': 'convolutional', 'batch_normalize': '1', 'filters': '64', 'size': '3', 'stride': '1', 'pad': '1', 'activation': 'leaky'},
        {'type': 'maxpool', 'size': '2', 'stride': '2'}, {'type': 'convolutional', 'batch_normalize': '1', 'filters': '128', 'size': '3', 'stride': '1', 'pad': '1', 'activation': 'leaky'}, 
        {'type': 'convolutional', 'batch_normalize': '1', 'filters': '64', 'size': '1', 'stride': '1', 'pad': '1', 'activation': 'leaky'}, 
        {'type': 'convolutional', 'batch_normalize': '1', 'filters': '128', 'size': '3', 'stride': '1', 'pad': '1', 'activation': 'leaky'}, 
        {'type': 'maxpool', 'size': '2', 'stride': '2'}, 
        {'type': 'convolutional', 'batch_normalize': '1', 'filters': '256', 'size': '3', 'stride': '1', 'pad': '1', 'activation': 'leaky'}, 
        {'type': 'convolutional', 'batch_normalize': '1', 'filters': '128', 'size': '1', 'stride': '1', 'pad': '1', 'activation': 'leaky'}, 
        {'type': 'convolutional', 'batch_normalize': '1', 'filters': '256', 'size': '3', 'stride': '1', 'pad': '1', 'activation': 'leaky'}, 
        {'type': 'maxpool', 'size': '2', 'stride': '2'}, 
        {'type': 'convolutional', 'batch_normalize': '1', 'filters': '512', 'size': '3', 'stride': '1', 'pad': '1', 'activation': 'leaky'}, 
        {'type': 'convolutional', 'batch_normalize': '1', 'filters': '256', 'size': '1', 'stride': '1', 'pad': '1', 'activation': 'leaky'}, 
        {'type': 'convolutional', 'batch_normalize': '1', 'filters': '512', 'size': '3', 'stride': '1', 'pad': '1', 'activation': 'leaky'}, 
        {'type': 'convolutional', 'batch_normalize': '1', 'filters': '256', 'size': '1', 'stride': '1', 'pad': '1', 'activation': 'leaky'}, 
        {'type': 'convolutional', 'batch_normalize': '1', 'filters': '512', 'size': '3', 'stride': '1', 'pad': '1', 'activation': 'leaky'}, 
        {'type': 'maxpool', 'size': '2', 'stride': '2'}, 
        {'type': 'convolutional', 'batch_normalize': '1', 'filters': '1024', 'size': '3', 'stride': '1', 'pad': '1', 'activation': 'leaky'}, 
        {'type': 'convolutional', 'batch_normalize': '1', 'filters': '512', 'size': '1', 'stride': '1', 'pad': '1', 'activation': 'leaky'}, 
        {'type': 'convolutional', 'batch_normalize': '1', 'filters': '1024', 'size': '3', 'stride': '1', 'pad': '1', 'activation': 'leaky'},
        {'type': 'convolutional', 'batch_normalize': '1', 'filters': '512', 'size': '1', 'stride': '1', 'pad': '1', 'activation': 'leaky'},
        {'type': 'convolutional', 'batch_normalize': '1', 'filters': '1024', 'size': '3', 'stride': '1', 'pad': '1', 'activation': 'leaky'}, 
        {'type': 'convolutional', 'batch_normalize': '1', 'size': '3', 'stride': '1', 'pad': '1', 'filters': '1024', 'activation': 'leaky'}, 
        {'type': 'convolutional', 'batch_normalize': '1', 'size': '3', 'stride': '1', 'pad': '1', 'filters': '1024', 'activation': 'leaky'}, 
        {'type': 'route', 'layers': '-9'},
        {'type': 'convolutional', 'batch_normalize': '1', 'size': '1', 'stride': '1', 'pad': '1', 'filters': '64', 'activation': 'leaky'},
        {'type': 'reorg', 'stride': '2'}, {'type': 'route', 'layers': '-1,-4'}, {'type': 'convolutional', 'batch_normalize': '1', 'size': '3', 'stride': '1', 'pad': '1', 'filters': '1024', 'activation': 'leaky'},
        {'type': 'convolutional', 'batch_normalize': 0, 'size': '1', 'stride': '1', 'pad': '1', 'filters': '425', 'activation': 'linear'}, 
        {'type': 'region', 'anchors': '0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828', 'bias_match': '1', 'classes': '80', 'coords': '4', 'num': '5', 'softmax': '1', 'jitter': '.3', 'rescore': '1', 'object_scale': '5', 'noobject_scale': '1', 'class_scale': '1', 'coord_scale': '1', 'absolute': '1', 'thresh': '.6', 'random': '1'}]

        """
        # print(block)
        blocks.append(block)
        # print(blocks)
    fp.close()
    return blocks


def print_cfg(blocks):
    print('layer     filters    size              input                output');
    prev_width = 416
    prev_height = 416
    prev_filters = 3
    out_filters = []
    out_widths = []
    out_heights = []
    ind = -2
    for block in blocks:
        # print(block)
        ind += 1
        # print(ind)
        if block['type'] == 'net':
            prev_width = int(block['width'])
            prev_height = int(block['height'])
            continue
        elif block['type'] == 'convolutional':
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            is_pad = int(block['pad'])
            # print(is_pad)
            pad = (kernel_size - 1) // 2 if is_pad else 0
            width = (prev_width + 2 * pad - kernel_size) // stride + 1
            height = (prev_height + 2 * pad - kernel_size) // stride + 1
            print('%5d %-6s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
            ind, 'conv', filters, kernel_size, kernel_size, stride, prev_width, prev_height, prev_filters, width,
            height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'maxpool':
            pool_size = int(block['size'])
            stride = int(block['stride'])
            width = prev_width // stride
            height = prev_height // stride
            print('%5d %-6s       %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
            ind, 'max', pool_size, pool_size, stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'avgpool':
            width = 1
            height = 1
            print('%5d %-6s                   %3d x %3d x%4d   ->  %3d' % (
            ind, 'avg', prev_width, prev_height, prev_filters, prev_filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'softmax':
            print('%5d %-6s                                    ->  %3d' % (ind, 'softmax', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'cost':
            print('%5d %-6s                                     ->  %3d' % (ind, 'cost', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'reorg':
            stride = int(block['stride'])
            filters = stride * stride * prev_filters
            width = prev_width // stride
            height = prev_height // stride
            print('%5d %-6s             / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
            ind, 'reorg', stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            filters = prev_filters
            width = prev_width * stride
            height = prev_height * stride
            print('%5d %-6s           * %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
            ind, 'upsample', stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'route':
            layers = block['layers'].split(',')
            layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
            if len(layers) == 1:
                print('%5d %-6s %d' % (ind, 'route', layers[0]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                prev_filters = out_filters[layers[0]]
            elif len(layers) == 2:
                print('%5d %-6s %d %d' % (ind, 'route', layers[0], layers[1]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                assert (prev_width == out_widths[layers[1]])
                assert (prev_height == out_heights[layers[1]])
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] in ['region', 'yolo']:
            print('%5d %-6s' % (ind, 'detection'))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'shortcut':
            from_id = int(block['from'])
            from_id = from_id if from_id > 0 else from_id + ind
            print('%5d %-6s %d' % (ind, 'shortcut', from_id))
            prev_width = out_widths[from_id]
            prev_height = out_heights[from_id]
            prev_filters = out_filters[from_id]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'connected':
            filters = int(block['output'])
            print('%5d %-6s                            %d  ->  %3d' % (ind, 'connected', prev_filters, filters))
            prev_filters = filters
            out_widths.append(1)
            out_heights.append(1)
            out_filters.append(prev_filters)
        else:
            print('unknown type %s' % (block['type']))


def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    # print("start: {}, num_w: {}, num_b: {}".format(start, num_w, num_b))
    # by ysyun, use .view_as()
    conv_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]).view_as(conv_model.bias.data));
    start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).view_as(conv_model.weight.data));
    start = start + num_w
    return start


def save_conv(fp, conv_model):
    if conv_model.bias.is_cuda:
        convert2cpu(conv_model.bias.data).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        conv_model.bias.data.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)


def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    # conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).view_as(conv_model.weight.data));
    start = start + num_w
    return start


def save_conv_bn(fp, conv_model, bn_model):
    if bn_model.bias.is_cuda:
        convert2cpu(bn_model.bias.data).numpy().tofile(fp)
        convert2cpu(bn_model.weight.data).numpy().tofile(fp)
        convert2cpu(bn_model.running_mean).numpy().tofile(fp)
        convert2cpu(bn_model.running_var).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        bn_model.bias.data.numpy().tofile(fp)
        bn_model.weight.data.numpy().tofile(fp)
        bn_model.running_mean.numpy().tofile(fp)
        bn_model.running_var.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)


def load_fc(buf, start, fc_model):
    num_w = fc_model.weight.numel()
    num_b = fc_model.bias.numel()
    fc_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    fc_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]));
    start = start + num_w
    return start


def save_fc(fp, fc_model):
    fc_model.bias.data.numpy().tofile(fp)
    fc_model.weight.data.numpy().tofile(fp)


if __name__ == '__main__':
    import sys
    # print(sys.argv)

    blocks = parse_cfg('cfg/yolo.cfg')
    if len(sys.argv) == 2:
        blocks = parse_cfg(sys.argv[1])
    print_cfg(blocks)
