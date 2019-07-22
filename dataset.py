import os
import numpy as np
import torch.utils.data
from lie_learn.spaces import S2
import hyper
import sys
sys.path.append('./src/build')
from sampling import compute


def rotmat(a, b, c, hom_coord=False):  # apply to mesh using mesh.apply_transform(rotmat(a,b,c, True))
    """
    Create a rotation matrix with an optional fourth homogeneous coordinate

    :param a, b, c: ZYZ-Euler angles
    """

    def z(a):
        return np.array([[np.cos(a), np.sin(a), 0, 0],
                         [-np.sin(a), np.cos(a), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def y(a):
        return np.array([[np.cos(a), 0, -np.sin(a), 0],
                         [0, 1, 0, 0],
                         [np.sin(a), 0, np.cos(a), 0],
                         [0, 0, 0, 1]])

    r = z(a).dot(y(b)).dot(z(c))  # pylint: disable=E1101
    if hom_coord:
        return r
    else:
        return r[:3, :3]


def rnd_rot():
    a = np.random.rand() * 2 * np.pi
    z = np.random.rand() * 2 - 1
    c = np.random.rand() * 2 * np.pi
    rot = rotmat(a, np.arccos(z), c, False)
    return rot


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, pts, labels, segs, rand_rot, aug, cache_dir=None):
        self.rand_rot = rand_rot
        self.aug = aug
        self.pts = pts
        self.labels = labels
        self.segs = segs
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.pts)

    def balanced_indices_sample(self):
        ind = np.zeros([hyper.N_CATS], np.bool)
        results = []
        while not np.all(ind):
            idx = np.random.randint(len(self.pts))
            if ind[self.labels[idx]]:
                continue
            ind[self.labels[idx]] = True
            results.append(idx)
        return results

    def __getitem__(self, index):
        b = hyper.BANDWIDTH_IN
        pts = np.array(self.pts[index])

        # randomly sample points
        # sub_idx = np.random.randint(0, pts.shape[0],  hyper.N_PTCLOUD)
        sub_idx = np.arange(hyper.N_PTCLOUD)
        pts = pts[sub_idx]
        if self.aug:
            rot = rnd_rot()
            pts = np.einsum('ij,nj->ni', rot, pts)
            pts += np.random.rand(3)[None, :] * 0.05
            pts = np.einsum('ij,nj->ni', rot.T, pts)
        segs = np.array(self.segs[index])
        segs = segs[sub_idx]
        labels = self.labels[index]

        pts_norm = np.linalg.norm(pts, axis=1)
        pts_normed = pts / pts_norm[:, None]
        rand_rot = rnd_rot() if self.rand_rot else np.eye(3)
        rotated_pts_normed = np.clip(pts_normed @ rand_rot, -1, 1)

        pts_s2 = S2.change_coordinates(rotated_pts_normed, p_from='C', p_to='S')
        pts_s2[:, 0] *= 2 * b / np.pi  # [0, pi]
        pts_s2[:, 1] *= b / np.pi
        pts_s2[:, 1][pts_s2[:, 1] < 0] += 2 * b

        pts_s2_float = pts_s2
        pts_s2 = (pts_s2 + 0.5).astype(np.int)
        pts_s2[:, 0] = np.clip(pts_s2[:, 0], 0, 2 * b - 1)
        pts_s2[:, 1] = np.clip(pts_s2[:, 1], 0, 2 * b - 1) # [0, 2pi]

        # N * 3
        pts_so3 = np.stack([pts_norm * 2 - 1, pts_s2_float[:, 1] / (2 * b - 1) * 2 - 1, pts_s2_float[:, 0] / (2 * b - 1) * 2 - 1], axis=1)
        pts_so3 = np.clip(pts_so3, -1, 1)

        # one hundred times speed up !
        features = np.asarray(compute(pts_s2_float, np.linalg.norm(pts, axis=1), hyper.R_IN, b, np.sin(np.pi * (2 * np.arange(2 * b) + 1) / 4 / b)))

        return features, pts_so3.astype(np.float32), segs.astype(np.int64), pts @ rand_rot, labels.astype(np.int64)
