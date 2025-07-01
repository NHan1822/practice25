import scipy.io as scio
import numpy as np
from src.ops import curl


class GXBox(object):
    """This class wraps IDL Box structure used in GX Simulator"""

    # noinspection PyTypeChecker
    def __init__(self, filename, order='F'):
        self.order = order
        self._box = scio.readsav(filename).box
        self.refids = [ptr.ID[0].decode('utf-8') for ptr in self._box['REFMAPS'][0].OMAP[0].POINTER[0].PTRS[0]
                       if type(ptr) == np.recarray]

    @property
    def bx(self):
        if self.order == 'C':
            return self._box.bx[0]
        else:
            return np.transpose(self._box.bx[0], (2, 1, 0))

    @property
    def by(self):
        if self.order == 'C':
            return self._box.by[0]
        else:
            return np.transpose(self._box.by[0], (2, 1, 0))

    @property
    def bz(self):
        if self.order == 'C':
            return self._box.bz[0]
        else:
            return np.transpose(self._box.bz[0], (2, 1, 0))

    @property
    def field(self):
        return self.bx, self.by, self.bz

    @property
    def curl(self):
        return curl(*self.field)
