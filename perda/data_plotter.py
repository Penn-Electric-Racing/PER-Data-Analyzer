import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from .csv_parser import SingleRunData
from .models import DataInstance
from .utils import get_data_slice_by_timestamp


def plot(
    data: SingleRunData,
    left_input,
    right_input=None,
    start_time=0,
    end_time=-1,
    unit="s",
    label=True,
    left_spacing=-1,
    right_spacing=-1,
    left_title="",
    right_title="",
    top_title="",
    figsize=(8, 5),
):
    """
    Plot data from the ParsedData model.
    left_input: name of variable to plot on left y-axis (has to have sth)
    right_input: name of variable to plot on right y-axis (optional)
    start_time: start time in ms (default 0)
    end_time: end time in ms (default -1, means till end)
    label: whether to show label (default True)
    left_spacing: spacing for left y-axis ticks (default -1, means auto)
    right_spacing: spacing for right y-axis ticks (default -1, means auto)
    figsize: tuple of width and height in inches (default (10, 6))
    """
    left_di = []
    right_di = []
    dual_axis = right_input is not None

    # Adjust start_time and end_time if unit is in seconds
    if unit == "s":
        start_time *= 1e3
        end_time *= 1e3

    # Get left DataInstance list and filter time
    if not isinstance(left_input, list):
        di = data[left_input]
        ftr_di = get_data_slice_by_timestamp(di, start_time, end_time)
        left_di = [ftr_di]
    else:
        for left in left_input:
            di = data[left]
            ftr_di = get_data_slice_by_timestamp(di, start_time, end_time)
            left_di.append(ftr_di)

    # Get right DataInstance list if there is any and filter time
    if dual_axis:
        if not isinstance(right_input, list):
            di = data[right_input]
            ftr_di = get_data_slice_by_timestamp(di, start_time, end_time)
            right_di = [ftr_di]
        else:
            for right in right_input:
                di = data[right]
                ftr_di = get_data_slice_by_timestamp(di, start_time, end_time)
                right_di.append(ftr_di)

    # Setup plot and figsize
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = None
    # Setup colors
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    coloridx = 1

    # Plot left axis
    ax1.set_ylabel(left_title)
    ax1.tick_params(axis="y")
    for ldi in left_di:
        ts = ldi.timestamp_np
        val = ldi.value_np
        if unit == "s":
            ts = ts.astype(np.float64) / 1e3
        lb = ldi.label
        ax1.plot(ts, val, label=lb, color=colors[coloridx % len(colors)])
        coloridx += 1
        if left_spacing != -1:
            ax1.yaxis.set_major_locator(MultipleLocator(left_spacing))

    # Plot right axis if any
    if dual_axis:
        ax2 = ax1.twinx()
        ax2.set_ylabel(right_title)
        ax2.tick_params(axis="y")
        for rdi in right_di:
            ts = rdi.timestamp_np
            val = rdi.value_np
            if unit == "s":
                ts = ts.astype(np.float64) / 1e3
            lb = rdi.label
            ax2.plot(
                ts, val, linestyle="-.", label=lb, color=colors[coloridx % len(colors)]
            )
            coloridx += 1
            if right_spacing != -1:
                ax2.yaxis.set_major_locator(MultipleLocator(right_spacing))

    # Calculate proper spacing for legends based on figure size
    if label:
        # Add padding to make room for legends
        fig.tight_layout(pad=2.0)
        if dual_axis:
            ax1.legend(loc="lower left", bbox_to_anchor=(0, 1))
            ax2.legend(loc="lower right", bbox_to_anchor=(1, 1), markerfirst=False)
        else:
            ax1.legend(loc="best")
    else:
        fig.tight_layout()

    # Set x label and title
    xlabel = "Timestamp (ms)" if unit == "ms" else "Timestamp (s)"
    plt.xlabel(xlabel)
    plt.title(f"{top_title}")
    plt.show()
