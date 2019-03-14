from recon.core.sequence.utils import dist_mat_order, make_cluster_estimator
import numpy as np


if __name__ == "__main__":
    encs = np.load('/home/pushkar/scratchpad/face-rec/face-rec/store_dir/encs.npy')
    _order = dist_mat_order(encs, weight=4.0)
    order = dist_mat_order(encs)

    cluster_estimator = \
        make_cluster_estimator("dbscan", min_samples=2)
    clusters = cluster_estimator.fit_predict(encs)

    print(_order.shape[0], order.shape[0])
