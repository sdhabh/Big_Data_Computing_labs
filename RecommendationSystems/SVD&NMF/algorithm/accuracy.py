from collections import defaultdict

import numpy as np


def rmse(predictions, verbose=True):

    if not predictions:
        raise ValueError("Prediction list is empty.")

    mse = np.mean(
        [float((true_r - est) ** 2) for (_, _, true_r, est, _) in predictions]
    )
    rmse_ = np.sqrt(mse)

    if verbose:
        print(f"RMSE: {rmse_:1.4f}")

    return rmse_


def mse(predictions, verbose=True):

    if not predictions:
        raise ValueError("Prediction list is empty.")

    mse_ = np.mean(
        [float((true_r - est) ** 2) for (_, _, true_r, est, _) in predictions]
    )

    if verbose:
        print(f"MSE: {mse_:1.4f}")

    return mse_


def mae(predictions, verbose=True):

    if not predictions:
        raise ValueError("Prediction list is empty.")

    mae_ = np.mean([float(abs(true_r - est)) for (_, _, true_r, est, _) in predictions])

    if verbose:
        print(f"MAE:  {mae_:1.4f}")

    return mae_


def fcp(predictions, verbose=True):

    if not predictions:
        raise ValueError("Prediction list is empty.")

    predictions_u = defaultdict(list)
    nc_u = defaultdict(int)
    nd_u = defaultdict(int)

    for u0, _, r0, est, _ in predictions:
        predictions_u[u0].append((r0, est))

    for u0, preds in predictions_u.items():
        for r0i, esti in preds:
            for r0j, estj in preds:
                if esti > estj and r0i > r0j:
                    nc_u[u0] += 1
                if esti >= estj and r0i < r0j:
                    nd_u[u0] += 1

    nc = np.mean(list(nc_u.values())) if nc_u else 0
    nd = np.mean(list(nd_u.values())) if nd_u else 0

    try:
        fcp = nc / (nc + nd)
    except ZeroDivisionError:
        raise ValueError(
            "cannot compute fcp on this list of prediction. "
            + "Does every user have at least two predictions?"
        )

    if verbose:
        print(f"FCP:  {fcp:1.4f}")

    return fcp
