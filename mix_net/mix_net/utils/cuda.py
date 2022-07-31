"""This module provides a function to cudanize an artbitrary number of tensors."""


def cudanize(*args):
    """Transform an arbitrary amount of input tensors to cuda tensors.

    Returns:
        (tuple): cuda tensors in the same order as the given inputs
    """
    arglist = []
    for arg in list(args):
        try:
            arglist.append(arg.cuda())
        except Exception:
            arglist.append(None)
    output = arglist[0] if len(arglist) <= 1 else tuple(arglist)
    return output
