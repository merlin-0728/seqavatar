import torch
from pytorch3d.ops import knn_points


class KNN:
    def __init__(self, k=1, transpose_mode=False):
        self.k = k
        self.transpose_mode = transpose_mode

    def __call__(self, ref, query):
        squeeze_batch = False
        if ref.dim() == 2:
            ref = ref.unsqueeze(0)
            query = query.unsqueeze(0)
            squeeze_batch = True

        if not self.transpose_mode:
            ref = ref.transpose(1, 2).contiguous()
            query = query.transpose(1, 2).contiguous()

        result = knn_points(query, ref, K=self.k, return_nn=False)
        dists = torch.sqrt(torch.clamp_min(result.dists, 0.0))
        idx = result.idx

        if squeeze_batch:
            dists = dists.squeeze(0)
            idx = idx.squeeze(0)
        return dists, idx
