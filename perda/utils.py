from .models import DataInstance


def get_data_slice_by_timestamp(
    original_instance: DataInstance, start_time: int = 0, end_time: int = -1
):
    """
    Get a new DataInstance with data in [start_time, end_time).
    If end_time == -1, means till end.
    """
    if end_time < 0:
        mask = original_instance.timestamp_np >= start_time
    else:
        mask = (original_instance.timestamp_np >= start_time) & (
            original_instance.timestamp_np < end_time
        )
    return DataInstance(
        timestamp_np=original_instance.timestamp_np[mask],
        value_np=original_instance.value_np[mask],
        label=original_instance.label,
        canid=original_instance.canid,
    )
