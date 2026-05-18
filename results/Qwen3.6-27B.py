import utils

palette = utils.get_palette()

datasets = [
    {
        "key": "Q",
        "color": palette["orange"],
        "data": [
            # {"label": "Q2", "kld": 2.200684, "ram": 7.83},
            {"label": "Q3", "kld": 0.603271, "ram": 10.96, "pos": "top left"},
            {"label": "Q4", "kld": 0.299316, "ram": 14.09},
            {"label": "Q5", "kld": 0.203751, "ram": 17.23, "pos": "top left"},
            {"label": "Q6", "kld": 0.115662, "ram": 20.36, "pos": "top left"},
            {"label": "Q8", "kld": 0.057327, "ram": 26.62, "pos": "top left"},
        ],
    },
    {
        "key": "oQ",
        "color": palette["blue"],
        "data": [
            # {"label": "oQ2", "kld": 1.691895, "ram": 9.74},
            {"label": "oQ3", "kld": 0.614136, "ram": 11.75},
            {"label": "oQ3.5", "kld": 0.576782, "ram": 12.70},
            {"label": "oQ4", "kld": 0.275848, "ram": 14.72},
            {"label": "oQ5", "kld": 0.200607, "ram": 17.69},
            {"label": "oQ6", "kld": 0.118469, "ram": 20.65},
            {"label": "oQ8", "kld": 0.057327, "ram": 26.62},
        ],
    },
    {
        "key": "UD",
        "color": palette["green"],
        "data": [
            {"label": "UD3", "kld": 0.315521, "ram": 21.54},
            {"label": "UD4", "kld": 0.168304, "ram": 23.53},
        ],
    },
    {
        "key": "PARO",
        "color": palette["purple"],
        "data": [
            {"label": "PARO", "kld": 0.235654, "ram": 13.42},
        ],
    },
    {
        "key": "OptiQ",
        "color": palette["red"],
        "data": [
            {"label": "OptiQ", "kld": 0.277649, "ram": 15.34, "pos": "bottom right"},
        ],
    },
]

utils.render_chart(
    name="Qwen3.6-27B",
    datasets=datasets,
    x_min=5,
    x_max=30,
    x_step=5,
    y_min=0,
    y_max=0.75,
    y_step=0.15,
)
