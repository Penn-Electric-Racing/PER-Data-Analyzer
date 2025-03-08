from .csvparser import csvparser
from .dataplotter import dataplotter
from .arrayoperator import arrayoperator

class analyzer:
    def __init__(self):
        self.__csvparser = csvparser()
        self.__dataplotter = dataplotter()
        self.__operator = arrayoperator()
        self.__file_read = False

        print("Analyzer Created")

    def reset(self):
        self.__csvparser.reset()
        self.__dataplotter.reset()
        self.__operator.reset()
        self.__file_read = False

        print("Reset Analyzer")

    def read_csv(self, path: str):
        if self.__file_read:
            print("Call .reset() before reading new csv")
            return
        self.__csvparser.read_csv(path)
        self.__dataplotter.get_csvparser(self.__csvparser)
        self.__operator.get_csvparser(self.__csvparser)
        self.__file_read = True

    def set_plot(self, start_time = 0, end_time = -1, same_graph = True, unit = "s"):
        self.__dataplotter.set_plot(start_time, end_time, same_graph, unit)

    def plot(self, variables: list):
        self.__dataplotter.plot(variables)

    def get_np_array(self, short_name: str):
        return self.__csvparser.get_np_array(short_name)

    def get_compute_arrays(self, op_list: list[str], match_type: str = "extend", start_time = 0, end_time = -1, unit = "s"):
        return self.__operator.get_compute_arrays(op_list, match_type, start_time, end_time, unit)
