import torch.utils.data
import torch

import types
import importlib.machinery
import os
import numpy as np
import h5py
import hyper

from dataset import MyDataset

data_path = "./hdf5_data"
N_PARTS = hyper.N_PARTS
N_CATS = hyper.N_CATS

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

color_map = {
    0: (0.65, 0.95, 0.05),
    1: (0.35, 0.05, 0.35),
    2: (0.65, 0.35, 0.65),
    3: (0.95, 0.95, 0.65),
    4: (0.95, 0.65, 0.05),
    5: (0.35, 0.05, 0.05),
    8: (0.05, 0.05, 0.65),
    9: (0.65, 0.05, 0.35),
    10: (0.05, 0.35, 0.35),
    11: (0.65, 0.65, 0.35),
    12: (0.35, 0.95, 0.05),
    13: (0.05, 0.35, 0.65),
    14: (0.95, 0.95, 0.35),
    15: (0.65, 0.65, 0.65),
    16: (0.95, 0.95, 0.05),
    17: (0.65, 0.35, 0.05),
    18: (0.35, 0.65, 0.05),
    19: (0.95, 0.65, 0.95),
    20: (0.95, 0.35, 0.65),
    21: (0.05, 0.65, 0.95),
    36: (0.05, 0.95, 0.05),
    37: (0.95, 0.65, 0.65),
    38: (0.35, 0.95, 0.95),
    39: (0.05, 0.95, 0.35),
    40: (0.95, 0.35, 0.05),
    47: (0.35, 0.05, 0.95),
    48: (0.35, 0.65, 0.95),
    49: (0.35, 0.05, 0.65)
}


def load_test_set(rand_rot, aug=False):
    f = h5py.File(os.path.join(data_path, 'ply_data_test0.h5'))

    labels = np.asarray(f['label'])
    pts = []
    segs = []
    for i in range(labels.shape[0]):
        pts.append(np.asarray(f['data%d' % i]))
        segs.append(np.asarray(f['pid%d' % i]))
    print(np.array(pts).shape, np.array(segs).shape)

    f.close()

    test_set = MyDataset(pts, labels, segs, rand_rot=rand_rot, aug=aug)

    return test_set


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # prerequisites: install s2cnn and its dependencies from
    # https://github.com/jonas-koehler/s2cnn

    # download weight from
    # https://drive.google.com/open?id=1QnFqQdWmx0cYtYeN9tJNlf-E5ZLawRBv

    # download dataset from
    # https://drive.google.com/drive/folders/1wC-DpeRtxuuEvffubWdhwoGXGeW052Vy?usp=sharing

    # sample usage: python test.py --weight_path ./state.pkl --model_path ./model.py --num_workers 4
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    weight_path = args.weight_path
    model_path = args.model_path

    torch.backends.cudnn.benchmark = True

    # Load the model
    loader = importlib.machinery.SourceFileLoader('model', model_path)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)

    model = mod.Model(N_PARTS)
    model.cuda()
    model.load_state_dict(torch.load(weight_path))

    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    test_set = load_test_set(True)

    batch_size = args.batch_size
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                                              pin_memory=True, drop_last=False)

    model.eval()

    # -------------------------------------------------------------------------------- #
    total_correct = 0
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    for batch_idx, (data, target_index, target, pt_cloud, category) in enumerate(test_loader):

        # Transform category labels to one_hot.
        category_labels = torch.LongTensor(category)
        one_hot_labels = torch.zeros(category.size(0), hyper.N_CATS).scatter_(1, category_labels, 1).cuda()

        data, target_index, target = data.cuda(), target_index.cuda(), target.cuda()

        # print (data.shape)
        with torch.no_grad():
            _, prediction = model(data, target_index, one_hot_labels)

        prediction = prediction.view(-1, hyper.N_PTCLOUD, hyper.N_PARTS)

        target = target.view(-1, hyper.N_PTCLOUD)

        for j in range(target.size(0)):
            cat = seg_label_to_cat[target.cpu().numpy()[j][0]]
            prediction_np = prediction.cpu().numpy()[j][:, seg_classes[cat]].argmax(1) + seg_classes[cat][0]
            target_np = target.cpu().numpy()[j]
            correct = np.mean((prediction_np == target_np).astype(np.float32))

            total_correct += correct

            segp = prediction_np
            segl = target_np
            part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
            for l in seg_classes[cat]:
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    part_ious[l - seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                        np.sum((segl == l) | (segp == l)))
            shape_ious[cat].append(np.mean(part_ious))

        print('acc: ', (batch_idx + 1) * batch_size, total_correct / (batch_idx + 1) / batch_size)

    all_shape_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_shape_ious.append(iou)
        shape_ious[cat] = np.mean(shape_ious[cat])
    print("all shape mIoU: %f, shape mIoU: %f" % (np.mean(all_shape_ious), np.nanmean(list(shape_ious.values()))))
