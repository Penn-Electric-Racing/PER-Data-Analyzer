# perda.utils.gps_analysis

### perda.utils.gps_analysis.create_representative_gps_image(lat_raw, lon_raw, vel_raw=None, vel_thresh=0.5, title=None, layout_config=LayoutConfig(width=1200, height=700, margin={'l': 70, 'r': 50, 't': 90, 'b': 70}, plot_bgcolor='white', title_x=0.5, title_xanchor='center', title_yanchor='top'), font_config=FontConfig(large=20, medium=14, small=10), gps_map_config=GpsMapConfig(mapbox_style='carto-positron', mapbox_token='', marker_size=3, line_width=2, line_color='red', zoom_padding=0.4, lat_range=(35.0, 50.0), lon_range=(-125.0, -66.0)))

Generate a GPS trace on an interactive map background.

Points outside the configured lat/lon bounds are filtered out.
Uses CARTO Positron tiles by default (free, no API key required).

* **Parameters:**
  * **lat_raw** ([*DataInstance*](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance)) – Raw latitude data instance, timestamp alignment base.
  * **lon_raw** ([*DataInstance*](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance)) – Raw longitude data instance.
  * **vel_raw** ([*DataInstance*](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance) *|* *None* *,* *optional*) – Raw velocity data instance.
  * **vel_thresh** (*float* *,* *optional*) – Velocity threshold for trimming, by default 0.5, same unit as velocity.
  * **title** (*str* *|* *None* *,* *optional*) – Title for the plot.
  * **layout_config** ([*LayoutConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.LayoutConfig) *,* *optional*)
  * **font_config** ([*FontConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.FontConfig) *,* *optional*)
  * **gps_map_config** ([*GpsMapConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.GpsMapConfig) *,* *optional*) – Map overlay and GPS filtering configuration.
* **Returns:**
  Plotly figure with map background, or None if velocity never exceeds
  the threshold or no points remain after filtering.
* **Return type:**
  go.Figure | None

### Examples

```pycon
>>> fig = create_representative_gps_image(lat_di, lon_di, vel_raw=vel_di, title="Lap Track")
>>> fig.show()
```

### perda.utils.gps_analysis.plot_gps_trimmer(lat, lon, title=None, layout_config=LayoutConfig(width=1200, height=700, margin={'l': 70, 'r': 50, 't': 90, 'b': 70}, plot_bgcolor='white', title_x=0.5, title_xanchor='center', title_yanchor='top'), font_config=FontConfig(large=20, medium=14, small=10), timestamp_unit=Timescale.MS)

Interactive GPS coordinate trimmer with a dual-handle range slider.

Displays a 2D scatter plot of GPS coordinates (longitude vs latitude) alongside
an ipywidgets IntRangeSlider that trims which points are shown. Designed for use
in Jupyter notebooks.

* **Parameters:**
  * **lat** ([*DataInstance*](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance)) – Latitude values. Must have the same length and identical timestamps as lon.
  * **lon** ([*DataInstance*](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance)) – Longitude values. Must have the same length and identical timestamps as lat.
  * **title** (*str* *|* *None* *,* *optional*)
  * **layout_config** ([*LayoutConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.LayoutConfig) *,* *optional*)
  * **font_config** ([*FontConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.FontConfig) *,* *optional*)
  * **timestamp_unit** ([*Timescale*](perda.units.md#perda.units.Timescale) *,* *optional*) – Timestamp unit used for the slider timestamp labels.
* **Return type:**
  ipywidgets.VBox

### Examples

```pycon
>>> widget = plot_gps_trimmer(lat_di, lon_di, title="GPS Track")
>>> display(widget)
```
