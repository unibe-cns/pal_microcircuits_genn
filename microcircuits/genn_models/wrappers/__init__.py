from pprint import pformat
import json

INDENT = 4

class PrinterMixin:
    @staticmethod
    def _print_obj(obj):
        print("\n--------------------------------------")
        print(obj.label)
        for k in obj.__dict__:
            try:
                s = json.dumps(obj.__dict__[k], indent=INDENT)
            except:
                s = pformat(obj.__dict__[k], indent=INDENT)
            print("\t{}:\n\t\t{}".format(k, s))

        print("--------------------------------------\n")

    def _str_params(self, level=1):
        s = []
        for k in self.__dict__:
            try:
                ss = json.dumps(self.__dict__[k], indent=INDENT * level)
            except:
                ss = pformat(self.__dict__[k], indent=INDENT * level)

            s.append("\t{}:\n\t\t{}".format(k, ss))
        return "\n".join(s)


from .input import Input
from .inter import Inter
from .output import Output
from .pyramidal import Pyramidal
from .supervisor import Supervisor
from .connect import Connector
from .network import Network
from .recorder import Recorder
