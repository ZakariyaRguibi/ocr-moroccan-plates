import bz2
import pickle


def compressed_pickle(title, data):
    with compress_file(title + '.pbz2') as f:
        pickle.dump(data, f)


def decompress_pickle(file):
    data = decompress_file(file)
    data = pickle.load(data)
    return data


def compress_file(path):
    return bz2.BZ2File(path, 'w')


def decompress_file(path):
    return bz2.BZ2File(path, 'rb')