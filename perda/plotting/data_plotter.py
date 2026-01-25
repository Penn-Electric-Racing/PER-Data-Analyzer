from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from ..analyzer.models import DataInstance
from ..csv_parser import SingleRunData
from ..utils.utils import get_data_slice_by_timestamp


def plot(
    data: SingleRunData,
    left_input: Union[str, int, DataInstance, List[Union[str, int, DataInstance]]],
    right_input: Optional[
        Union[str, int, DataInstance, List[Union[str, int, DataInstance]]]
    ] = None,
    start_time: Union[int, float] = 0,
    end_time: Union[int, float] = -1,
    unit: str = "s",
    label: bool = True,
    left_spacing: Union[int, float] = -1,
    right_spacing: Union[int, float] = -1,
    left_title: str = "",
    right_title: str = "",
    top_title: str = "",
    figsize: Tuple[Union[int, float], Union[int, float]] = (8, 5),
) -> None:
    """
    Plot data from the SingleRunData model.

    Parameters
    ----------
    data : SingleRunData
        Parsed data containing CAN variables
    left_input : Union[str, int, DataInstance, List[Union[str, int, DataInstance]]]
        Variable(s) to plot on left y-axis. Can be name, CAN ID, or DataInstance
    right_input : Optional[Union[str, int, DataInstance, List[Union[str, int, DataInstance]]]], optional
        Variable(s) to plot on right y-axis. Can be name, CAN ID, or DataInstance. Default is None
    start_time : Union[int, float], optional
        Start time for plotting. Default is 0
    end_time : Union[int, float], optional
        End time for plotting. -1 means until end. Default is -1
    unit : str, optional
        Time unit: "ms" or "s". Default is "s"
    label : bool, optional
        Whether to show legends. Default is True
    left_spacing : Union[int, float], optional
        Spacing for left y-axis ticks. -1 means auto. Default is -1
    right_spacing : Union[int, float], optional
        Spacing for right y-axis ticks. -1 means auto. Default is -1
    left_title : str, optional
        Label for left y-axis. Default is ""
    right_title : str, optional
        Label for right y-axis. Default is ""
    top_title : str, optional
        Title for the entire plot. Default is ""
    figsize : Tuple[Union[int, float], Union[int, float]], optional
        Figure size in inches (width, height). Default is (8, 5)

    Returns
    -------
    None
    """
    left_di = []
    right_di = []
    dual_axis = right_input is not None
    ax2 = None  # Initialize ax2 for type checking

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
        if len(ldi) == 0:
            print(f"Warning: No data points in time range for {ldi.label}")
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
            if len(rdi) == 0:
                print(f"Warning: No data points in time range for {rdi.label}")
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
            if ax2 is not None:
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
