import plotly.graph_objects

datasets = [
    {
        "name": "Q",
        "color": "#dd8452",
        "data": [
            # {"label": "Q2", "kld": 2.200684, "ram": 7.83},
            {"label": "Q3", "kld": 0.603271, "ram": 10.96},
            {"label": "Q4", "kld": 0.299316, "ram": 14.09},
            {"label": "Q5", "kld": 0.203751, "ram": 17.23, "pos": "top left", "x": -0.15},
            {"label": "Q6", "kld": 0.115662, "ram": 20.36, "pos": "top left", "x": -0.15},
            {"label": "Q8", "kld": 0.057327, "ram": 26.62},
        ],
    },
    {
        "name": "oQ",
        "color": "#4c72b0",
        "data": [
            {"label": "oQ4", "kld": 0.275848, "ram": 14.72},
            {"label": "oQ5", "kld": 0.200607, "ram": 17.69},
            {"label": "oQ6", "kld": 0.118469, "ram": 20.65},
        ],
    },
    {
        "name": "UD",
        "color": "#55a868",
        "data": [
            {"label": "UD4", "kld": 0.168304, "ram": 23.53},
        ],
    },
    {
        "name": "PARO",
        "color": "#8c6bb1",
        "data": [
            {"label": "PARO", "kld": 0.235654, "ram": 13.42},
        ],
    },
]

x_min = 10
x_max = 30
y_min = 0
y_max = 1
x_step_fixed = 5
y_step_fixed = 0.25
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
    title="Qwen3.6-27B",
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
fig.write_image("results/Qwen3.6-27B.svg")
