import argparse


def args():
    parser = argparse.ArgumentParser(
        description="Train a reservoir based SNN on biosignals"
    )

    parser.add_argument(
        "--method", default="individual", type=str, help="Dataset(BCI3)"
    )
    # Defining the model
    parser.add_argument(
        "--dataset", default="bci3", type=str, help="Dataset(BCI3)"
    )

    parser.add_argument(
        "--tstep", default="500", type=str, help="Dataset(BCI3)"
    )
    parser.add_argument(
        "--classifier", default="RF", type=str, help="Dataset(BCI3)"
    )
    parser.add_argument(
        "--l_feat", default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], type=str, help="Dataset(BCI3)"
    )
    parser.add_argument(
        "--run", default=1, type=int, help="Dataset(BCI3)"
    )
    parser.add_argument(
        "--niter", default=100, type=int, help="Dataset(BCI3)"
    )
    parser.add_argument(
        "--nfeatures", default=17, type=int, help="Dataset(BCI3)"
    )
    parser.add_argument(
        "--max_feat", default=5, type=int, help="Dataset(BCI3)"
    )
    parser.add_argument(
        "--gen", default=2, type=int, help="Dataset(BCI3)"
    )
    



    my_args = parser.parse_args()

    return my_args
