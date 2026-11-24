import plotly.graph_objects as go

# Font sizes (px)
FONT_SIZE_LARGE = 20  # Titles
FONT_SIZE_MEDIUM = 14  # Axis labels, legend titles
FONT_SIZE_SMALL = 10  # Tick labels, legend items

# Figure size (px)
PLOT_WIDTH = 1200
PLOT_HEIGHT = 700

# Stacked subplots: height per subplot row (px) and gap between rows (fraction of figure)
SUBPLOT_HEIGHT = 350
SUBPLOT_VERTICAL_SPACING = 0.12

# Figure layout
MARGIN = go.layout.Margin(l=70, r=50, t=90, b=70)
PLOT_BGCOLOR = "white"
TITLE_X = 0.5
TITLE_XANCHOR = "center"
TITLE_YANCHOR = "top"

# Vertical marker lines (e.g. concat boundaries)
VLINE_COLOR = "gray"
VLINE_WIDTH = 2
VLINE_DASH = "dash"
VLINE_OPACITY = 0.7

# Line style for signals on a right-hand (secondary) y-axis
RIGHT_AXIS_LINE_DASH = "dash"

# Scatter + histogram
SCATTER_COLOR = "blue"
REFERENCE_LINE_COLOR = "crimson"
HISTOGRAM_COLOR = "blue"
HISTOGRAM_BINS = 80

# Parametric curves
PARAMETRIC_MARKER_SIZE = 3
TRIMMER_MARKER_SIZE = 4
TIME_COLORSCALE = "Viridis"  # Colors trimmer points by elapsed time

# Map traces
MAP_STYLE = "open-street-map"  # MapLibre style, needs no API key
MAP_MARKER_SIZE = 3
MAP_LINE_WIDTH = 2
MAP_LINE_COLOR = "red"
MAP_ZOOM_PADDING = 0.4  # Extra fraction added around the data extent when auto-zooming
MAP_MAX_ZOOM = 15.0  # Zoom used when all points coincide

# FFT spectrum
FFT_LINE_COLOR = "steelblue"
