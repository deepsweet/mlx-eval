import utils

palette = utils.get_palette()

datasets = [
    {
        "key": "Q",
        "color": palette["orange"],
        "data": [
            # {"label": "Q2", "kld": 3.075195, "ram": 10.10},
            {"label": "Q3", "kld": 0.285767, "ram": 14.14},
            {"label": "Q4", "kld": 0.088287, "ram": 18.17},
            {"label": "Q5", "kld": 0.037971, "ram": 22.20},
            {"label": "Q6", "kld": 0.021137, "ram": 26.23, "pos": "top left"},
            {"label": "Q8", "kld": 0.013805, "ram": 34.30, "pos": "top left"},
        ],
    },
    {
        "key": "oQ",
        "color": palette["blue"],
        "data": [
            {"label": "oQ2", "kld": 0.318237, "ram": 11.40},
            {"label": "oQ3", "kld": 0.216858, "ram": 14.77, "pos": "top left"},
            {"label": "oQ3.5", "kld": 0.216858, "ram": 16.00},
            {"label": "oQ4", "kld": 0.047134, "ram": 18.83},
            {"label": "oQ5", "kld": 0.023533, "ram": 22.76},
            {"label": "oQ6", "kld": 0.019783, "ram": 26.51},
            {"label": "oQ8", "kld": 0.014273, "ram": 34.27},
        ],
    },
    {
        "key": "UD",
        "color": palette["green"],
        "data": [
            {"label": "UD3", "kld": 0.071945, "ram": 15.35},
            {"label": "UD4", "kld": 0.029251, "ram": 19.32},
        ],
    },
    {
        "key": "PARO",
        "color": palette["purple"],
        "data": [
            {"label": "PARO", "kld": 0.059202, "ram": 17.37},
        ],
    },
    {
        "key": "JANG",
        "color": palette["cyan"],
        "data": [
            {"label": "JTQ2", "kld": 0.240434, "ram": 10.00},
            {"label": "JTQ4", "kld": 0.034605, "ram": 17.50, "pos": "top left"},
        ],
    },
    {
        "key": "OptiQ",
        "color": palette["red"],
        "data": [
            {"label": "OptiQ", "kld": 0.028477, "ram": 19.67, "pos": "bottom right"},
        ],
    },
]

utils.render_chart(
    name="Qwen3.6-35B-A3B",
    datasets=datasets,
    x_min=5,
    x_max=40,
    x_step=5,
    y_min=0,
    y_max=0.35,
    y_step=0.05,
)
