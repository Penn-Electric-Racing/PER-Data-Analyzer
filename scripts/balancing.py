from time import sleep

from perda.live import LiveAnalyzer, ValueType

live = LiveAnalyzer.local()

while True:
    cell_max = live.get("bms.stack.mma.cellV.max", ValueType.FLOAT)
    cell_min = live.get("bms.stack.mma.cellV.min", ValueType.FLOAT)
    print("diff: ", str(cell_max - cell_min))
    sleep(1)
