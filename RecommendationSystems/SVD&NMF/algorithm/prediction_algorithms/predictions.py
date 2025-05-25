from collections import namedtuple


class PredictionImpossible(Exception):
    pass


class Prediction(namedtuple("Prediction", ["uid", "iid", "r_ui", "est", "details"])):

    __slots__ = ()

    def __str__(self):
        s = f"user: {self.uid:<10} "
        s += f"item: {self.iid:<10} "
        if self.r_ui is not None:
            s += f"r_ui = {self.r_ui:1.2f}   "
        else:
            s += "r_ui = None   "
        s += f"est = {self.est:1.2f}   "
        s += str(self.details)

        return s
