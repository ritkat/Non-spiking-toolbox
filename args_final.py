import argparse


def args():
    parser = argparse.ArgumentParser(
        description="Train a reservoir based SNN on biosignals"
    )

    parser.add_argument(
        "--method", default="individual", type=str, help="Feature selection method.(individual/baseline/genetic)"
    )
    
    # Defining the model
    parser.add_argument(
        "--dataset", default="bci3", type=str, help="Dataset(BCI3)"
    )

    parser.add_argument(
        "--tstep", default="500", type=str, help="time step for segementing the dataset. (200/300/500/1000/1500/3000)"
    )
    parser.add_argument(
        "--classifier", default="RF", type=str, help="Classifier. (SVM/RF)"
    )
    parser.add_argument(
        "--l_feat", default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35], type=str, help="Dataset(BCI3)"
    )
    parser.add_argument(
        "--run", default=1, type=int, help="Dataset(BCI3)"
    )
    parser.add_argument(
        "--niter", default=200, type=int, help="Dataset(BCI3)"
    )
    parser.add_argument(
        "--nfeatures", default=36, type=int, help="Dataset(BCI3)"
    )
    parser.add_argument(
        "--f_split", default=2, type=int, help="Dataset(BCI3)"
    )
    parser.add_argument(
        "--max_feat", default=int(5), help="Select None for no restriction on max_features and specify int for restricting the max features"
    )
    parser.add_argument(
        "--gen", default=20, type=int, help="Dataset(BCI3)"
    )
    parser.add_argument(
        "--seed",
        default=50,
        type=float,
        help="Seed for random number generation",
    )
    



    my_args = parser.parse_args()

    return my_args
