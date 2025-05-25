import pickle


def dump(file_name, predictions=None, algo=None, verbose=0):

    dump_obj = {"predictions": predictions, "algo": algo}
    pickle.dump(dump_obj, open(file_name, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        print("The dump has been saved as file", file_name)


def load(file_name):

    dump_obj = pickle.load(open(file_name, "rb"))

    return dump_obj["predictions"], dump_obj["algo"]
