import peralyzer  # or from PERalyzer import Analyzer
from peralyzer import Plotter

def main():
    analyzer = peralyzer.Analyzer()
    analyzer.analyze()
    # or analyzer = Analyzer()
    # do something with `analyzer`

if __name__ == "__main__":
    main()