import coloraide
import plotly.graph_objects

COLOR_L = 0.65
COLOR_C = 0.11
COLOR_H_OFFSET = 0
COLOR_NAMES = ["red", "orange", "yellow", "green", "cyan", "blue", "purple"]
COLOR_STEP = 360 / len(COLOR_NAMES)


def get_palette():
    palette = {}

    for i, name in enumerate(COLOR_NAMES):
        h = (i * COLOR_STEP + COLOR_H_OFFSET) % 360
        color = coloraide.Color("oklch", [COLOR_L, COLOR_C, h])

        if not color.in_gamut("srgb"):
            raise ValueError(f"Color '{name}' is out of sRGB gamut")

        palette[name] = color.convert("srgb").to_string(hex=True)

    return palette


PLOT_MINOR_SUBTICKS = 5
PLOT_HEIGHT = 800
PLOT_MARGIN_TOP = 50
PLOT_MARGIN_BOTTOM = 70
PLOT_MARGIN_LEFT = 90
PLOT_MARGIN_RIGHT = 30
PLOT_DEFAULT_TEXT_POS = "top right"
PLOT_COLOR_TEXT = "#000000"
PLOT_COLOR_BORDER = "#000000"
PLOT_COLOR_GRID_MAJOR = "#e5e5e5"
PLOT_COLOR_GRID_MINOR = "#f0f0f0"


def render_chart(
    name,
    datasets,
    x_min,
    x_max,
    x_step,
    y_min,
    y_max,
    y_step,
):
    x_range = x_max - x_min
    y_range = y_max - y_min
    text_x = x_range * 0.005
    text_y = y_range * 0.005
    pos_defaults = {
        "top left": {"x": -text_x},
        "bottom right": {"x":  text_x, "y": -text_y},
    }

    fig = plotly.graph_objects.Figure()

    for dataset in datasets:
        key = dataset["key"]
        color = dataset["color"]
        label = []
        kld = []
        ram = []
        label_pos = []
        label_x = []
        label_y = []

        for d in dataset["data"]:
            ram.append(d["ram"])
            kld.append(d["kld"])
            label.append(d["label"])

            pos = d.get("pos", PLOT_DEFAULT_TEXT_POS)
            offsets = pos_defaults.get(pos, {})
            dx = d.get("x", offsets.get("x", text_x))
            dy = d.get("y", offsets.get("y", text_y))

            label_pos.append(pos)
            label_x.append(d["ram"] + dx)
            label_y.append(d["kld"] + dy)

        marker = plotly.graph_objects.Scatter(
            x=ram,
            y=kld,
            mode="markers",
            name=f"{key}_marker",
            marker={"color": color, "size": 8},
        )
        fig.add_trace(marker)

        label = plotly.graph_objects.Scatter(
            x=label_x,
            y=label_y,
            mode="text",
            name=f"{key}_label",
            text=label,
            textposition=label_pos,
            textfont={"color": color, "size": 13},
        )
        fig.add_trace(label)

    dx_minor = x_step / PLOT_MINOR_SUBTICKS
    dy_minor = y_step / PLOT_MINOR_SUBTICKS
    x_range = x_max - x_min
    y_range = y_max - y_min
    inner_height = PLOT_HEIGHT - PLOT_MARGIN_TOP - PLOT_MARGIN_BOTTOM
    inner_width = inner_height * (x_range / dx_minor) / (y_range / dy_minor)
    width = inner_width + PLOT_MARGIN_LEFT + PLOT_MARGIN_RIGHT

    fig.update_layout(
        title=name,
        title_x=0.5,
        title_y=0.975,
        title_font={"size": 16},
        width=width,
        height=PLOT_HEIGHT,
        showlegend=False,
        font={"color": PLOT_COLOR_TEXT},
        xaxis_title="RAM (GiB)",
        yaxis_title="KL divergence (mean, nats)",
        yaxis={
            "showline": True,
            "mirror": True,
            "linecolor": PLOT_COLOR_BORDER,
            "linewidth": 1,
            "range": [y_min, y_max],
            "showgrid": True,
            "dtick": y_step,
            "tick0": y_min,
            "gridcolor": PLOT_COLOR_GRID_MAJOR,
            "ticklabelposition": "outside",
            "ticklabelstandoff": 10,
            "minor": {
                "showgrid": True,
                "dtick": dy_minor,
                "gridcolor": PLOT_COLOR_GRID_MINOR,
                "gridwidth": 0.5,
            },
        },
        xaxis={
            "showline": True,
            "mirror": True,
            "linecolor": PLOT_COLOR_BORDER,
            "linewidth": 1,
            "range": [x_min, x_max],
            "showgrid": True,
            "dtick": x_step,
            "tick0": x_min,
            "gridcolor": PLOT_COLOR_GRID_MAJOR,
            "ticklabelposition": "outside",
            "ticklabelstandoff": 10,
            "minor": {
                "showgrid": True,
                "dtick": dx_minor,
                "gridcolor": PLOT_COLOR_GRID_MINOR,
                "gridwidth": 0.5,
            },
        },
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin={
            "t": PLOT_MARGIN_TOP,
            "b": PLOT_MARGIN_BOTTOM,
            "l": PLOT_MARGIN_LEFT,
            "r": PLOT_MARGIN_RIGHT,
        },
    )

    # fig.show()
    fig.write_image(f"results/{name}.svg")
