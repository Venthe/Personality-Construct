from _utilities.cuda import detect_cuda


def select_device(use_gpu=True):
    return "cuda:0" if use_gpu and detect_cuda() else "cpu"
