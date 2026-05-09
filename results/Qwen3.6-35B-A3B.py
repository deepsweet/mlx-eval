import plotly.graph_objects

datasets = [
    {
        "name": "Q",
        "color": "#dd8452",
        "data": [
            # {"label": "Q2", "kld": 3.075195, "ram": 10.10},
            {"label": "Q3", "kld": 0.285767, "ram": 14.14},
            {"label": "Q4", "kld": 0.088287, "ram": 18.17},
            {"label": "Q5", "kld": 0.037971, "ram": 22.20},
            {"label": "Q6", "kld": 0.021137, "ram": 26.23, "pos": "top left"},
            {"label": "Q8", "kld": 0.013805, "ram": 34.30, "pos": "top left", "x": -0.15},
        ],
    },
    {
        "name": "oQ",
        "color": "#4c72b0",
        "data": [
            # {"label": "oQ2", "kld": 0.318237, "ram": 11.40},
            {"label": "oQ3", "kld": 0.216858, "ram": 14.77, "pos": "top left", "x": -0.15},
            {"label": "oQ3.5", "kld": 0.216858, "ram": 16.00},
            {"label": "oQ4", "kld": 0.047134, "ram": 18.83},
            {"label": "oQ5", "kld": 0.023533, "ram": 22.76},
            {"label": "oQ6", "kld": 0.019783, "ram": 26.51},
            {"label": "oQ8", "kld": 0.014273, "ram": 34.27},
        ],
    },
    {
        "name": "UD",
        "color": "#55a868",
        "data": [
            {"label": "UD3", "kld": 0.071945, "ram": 15.35},
            {"label": "UD4", "kld": 0.029251, "ram": 19.32},
        ],
    },
]

x_min = 10
x_max = 40
y_min = 0
y_max = 0.3
x_step_fixed = 5
y_step_fixed = 0.05
num_intervals = 5
minor_subdivisions = 5
height = 800
margin_top = 50
margin_bottom = 70
margin_left = 90
margin_right = 30
default_text_pos = "top right"
default_text_x = 0.0
default_text_y = 0.0025
color_text = "#000000"
color_border = "#000000"
color_grid_major = "#d3d3d3"
color_grid_minor = "#f5f5f5"

fig = plotly.graph_objects.Figure()

for dataset in datasets:
    name = dataset["name"]
    color = dataset["color"]
    label = []
    kld = []
    ram = []
    text_pos = []
    text_x = []
    text_y = []

    for d in dataset["data"]:
        ram.append(d["ram"])
        kld.append(d["kld"])
        label.append(d["label"])
        text_pos.append(d.get("pos", default_text_pos))
        text_x.append(d["ram"] + d.get("x", default_text_x))
        text_y.append(d["kld"] + d.get("y", default_text_y))

    marker = plotly.graph_objects.Scatter(
        x=ram,
        y=kld,
        mode="markers",
        name=f"{name}_marker",
        marker={"color": color, "size": 8},
    )
    fig.add_trace(marker)

    label = plotly.graph_objects.Scatter(
        x=text_x,
        y=text_y,
        mode="text",
        name=f"{name}_label",
        text=label,
        textposition=text_pos,
        textfont={"color": color, "size": 14},
    )
    fig.add_trace(label)


dx_minor = x_step_fixed / minor_subdivisions
dy_minor = y_step_fixed / minor_subdivisions
x_range = x_max - x_min
y_range = y_max - y_min
inner_height = height - margin_top - margin_bottom
inner_width = inner_height * (x_range / dx_minor) / (y_range / dy_minor)
width = inner_width + margin_left + margin_right

fig.update_layout(
    title="Qwen3.6-35B-A3B",
    title_x=0.5,
    title_y=0.975,
    title_font={"size": 16},
    width=width,
    height=height,
    showlegend=False,
    font={"color": color_text},
    xaxis_title="RAM (GiB)",
    yaxis_title="KL divergence (mean, nats)",
    yaxis={
        "showline": True,
        "mirror": True,
        "linecolor": color_border,
        "linewidth": 1,
        "range": [y_min, y_max],
        "showgrid": True,
        "dtick": y_step_fixed,
        "tick0": y_min,
        "gridcolor": color_grid_major,
        "ticklabelposition": "outside",
        "ticklabelstandoff": 10,
        "minor": {
            "showgrid": True,
            "dtick": dy_minor,
            "gridcolor": color_grid_minor,
            "gridwidth": 0.5,
        },
    },
    xaxis={
        "showline": True,
        "mirror": True,
        "linecolor": color_border,
        "linewidth": 1,
        "range": [x_min, x_max],
        "showgrid": True,
        "dtick": x_step_fixed,
        "tick0": x_min,
        "gridcolor": color_grid_major,
        "ticklabelposition": "outside",
        "ticklabelstandoff": 10,
        "minor": {
            "showgrid": True,
            "dtick": dx_minor,
            "gridcolor": color_grid_minor,
            "gridwidth": 0.5,
        },
    },
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    margin={
        "t": margin_top,
        "b": margin_bottom,
        "l": margin_left,
        "r": margin_right,
    },
)

# fig.show()
fig.write_image("results/Qwen3.6-35B-A3B.svg")
