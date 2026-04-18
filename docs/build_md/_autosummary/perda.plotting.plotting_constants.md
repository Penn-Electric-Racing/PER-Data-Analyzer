# perda.plotting.plotting_constants

### *pydantic model* perda.plotting.plotting_constants.DiffPlotConfig

Bases: `BaseModel`

Configuration for diff bar chart visualization.

* **Fields:**
  - [`bucket_size_ms (int)`](#perda.plotting.plotting_constants.DiffPlotConfig.bucket_size_ms)
  - [`color_base_extra (str)`](#perda.plotting.plotting_constants.DiffPlotConfig.color_base_extra)
  - [`color_incom_extra (str)`](#perda.plotting.plotting_constants.DiffPlotConfig.color_incom_extra)
  - [`color_total (str)`](#perda.plotting.plotting_constants.DiffPlotConfig.color_total)
  - [`color_value_mismatch (str)`](#perda.plotting.plotting_constants.DiffPlotConfig.color_value_mismatch)

#### *field* bucket_size_ms *: `int`* *= 1000*

#### *field* color_base_extra *: `str`* *= 'blue'*

#### *field* color_incom_extra *: `str`* *= 'darkorange'*

#### *field* color_value_mismatch *: `str`* *= 'crimson'*

#### *field* color_total *: `str`* *= 'gray'*

### *pydantic model* perda.plotting.plotting_constants.FontConfig

Bases: `BaseModel`

Font configuration for plot elements.

Simplified to three sizes:
- Large: Titles
- Medium: Axis labels, legend titles
- Small: Tick labels, legend items

* **Fields:**
  - [`large (int)`](#perda.plotting.plotting_constants.FontConfig.large)
  - [`medium (int)`](#perda.plotting.plotting_constants.FontConfig.medium)
  - [`small (int)`](#perda.plotting.plotting_constants.FontConfig.small)

#### *field* large *: `int`* *= 20*

#### *field* medium *: `int`* *= 14*

#### *field* small *: `int`* *= 10*

### *pydantic model* perda.plotting.plotting_constants.GpsMapConfig

Bases: `BaseModel`

Configuration for GPS map overlay plots.

* **Parameters:**
  * **mapbox_style** (*str*) – Mapbox tile style. `"carto-positron"` is free and requires no token.
  * **mapbox_token** (*str*) – Mapbox access token. Only needed for Mapbox-hosted styles.
  * **marker_size** (*int*) – Marker size on the map trace.
  * **line_width** (*int*) – Line width on the map trace.
  * **line_color** (*str*) – Color of the GPS trace line.
  * **zoom_padding** (*float*) – Extra fraction added around the data extent when auto-zooming.
  * **lat_range** (*tuple* *[**float* *,* *float* *]*) – Valid latitude range. Points outside are filtered.
  * **lon_range** (*tuple* *[**float* *,* *float* *]*) – Valid longitude range. Points outside are filtered.
* **Fields:**
  - [`lat_range (tuple[float, float])`](#perda.plotting.plotting_constants.GpsMapConfig.lat_range)
  - [`line_color (str)`](#perda.plotting.plotting_constants.GpsMapConfig.line_color)
  - [`line_width (int)`](#perda.plotting.plotting_constants.GpsMapConfig.line_width)
  - [`lon_range (tuple[float, float])`](#perda.plotting.plotting_constants.GpsMapConfig.lon_range)
  - [`mapbox_style (str)`](#perda.plotting.plotting_constants.GpsMapConfig.mapbox_style)
  - [`mapbox_token (str)`](#perda.plotting.plotting_constants.GpsMapConfig.mapbox_token)
  - [`marker_size (int)`](#perda.plotting.plotting_constants.GpsMapConfig.marker_size)
  - [`zoom_padding (float)`](#perda.plotting.plotting_constants.GpsMapConfig.zoom_padding)

#### *field* mapbox_style *: `str`* *= 'carto-positron'*

#### *field* mapbox_token *: `str`* *= ''*

#### *field* marker_size *: `int`* *= 3*

#### *field* line_width *: `int`* *= 2*

#### *field* line_color *: `str`* *= 'red'*

#### *field* zoom_padding *: `float`* *= 0.4*

#### *field* lat_range *: `tuple`[`float`, `float`]* *= (35.0, 50.0)*

#### *field* lon_range *: `tuple`[`float`, `float`]* *= (-125.0, -66.0)*

### *pydantic model* perda.plotting.plotting_constants.LayoutConfig

Bases: `BaseModel`

Layout configuration for plot dimensions and spacing.

* **Fields:**
  - [`height (int)`](#perda.plotting.plotting_constants.LayoutConfig.height)
  - [`margin (Dict[str, int])`](#perda.plotting.plotting_constants.LayoutConfig.margin)
  - [`plot_bgcolor (str)`](#perda.plotting.plotting_constants.LayoutConfig.plot_bgcolor)
  - [`title_x (float)`](#perda.plotting.plotting_constants.LayoutConfig.title_x)
  - [`title_xanchor (str)`](#perda.plotting.plotting_constants.LayoutConfig.title_xanchor)
  - [`title_yanchor (str)`](#perda.plotting.plotting_constants.LayoutConfig.title_yanchor)
  - [`width (int)`](#perda.plotting.plotting_constants.LayoutConfig.width)

#### *field* width *: `int`* *= 1200*

#### *field* height *: `int`* *= 700*

#### *field* margin *: `Dict`[`str`, `int`]* *[Optional]*

#### *field* plot_bgcolor *: `str`* *= 'white'*

#### *field* title_x *: `float`* *= 0.5*

#### *field* title_xanchor *: `str`* *= 'center'*

#### *field* title_yanchor *: `str`* *= 'top'*

### *pydantic model* perda.plotting.plotting_constants.ScatterHistogramPlotConfig

Bases: `BaseModel`

Configuration for frequency analysis visualization.

* **Fields:**
  - [`color_histogram (str)`](#perda.plotting.plotting_constants.ScatterHistogramPlotConfig.color_histogram)
  - [`color_line (str)`](#perda.plotting.plotting_constants.ScatterHistogramPlotConfig.color_line)
  - [`color_scatter (str)`](#perda.plotting.plotting_constants.ScatterHistogramPlotConfig.color_scatter)
  - [`histogram_bins (int)`](#perda.plotting.plotting_constants.ScatterHistogramPlotConfig.histogram_bins)

#### *field* color_scatter *: `str`* *= 'blue'*

#### *field* color_line *: `str`* *= 'crimson'*

#### *field* color_histogram *: `str`* *= 'blue'*

#### *field* histogram_bins *: `int`* *= 80*

### *pydantic model* perda.plotting.plotting_constants.VLineConfig

Bases: `BaseModel`

Configuration for vertical marker lines on plots.

* **Fields:**
  - [`color (str)`](#perda.plotting.plotting_constants.VLineConfig.color)
  - [`dash (str)`](#perda.plotting.plotting_constants.VLineConfig.dash)
  - [`opacity (float)`](#perda.plotting.plotting_constants.VLineConfig.opacity)
  - [`width (int)`](#perda.plotting.plotting_constants.VLineConfig.width)

#### *field* color *: `str`* *= 'gray'*

#### *field* width *: `int`* *= 2*

#### *field* dash *: `str`* *= 'dash'*

#### *field* opacity *: `float`* *= 0.7*
