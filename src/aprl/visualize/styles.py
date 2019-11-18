PRETTY_ENV = {
    "multicomp/KickAndDefend-v0": "Kick and Defend",
    "multicomp/SumoAntsAutoContact-v0": "Sumo Ants",
    "multicomp/SumoAnts-v0": "Sumo Ants",
    "multicomp/SumoHumansAutoContact-v0": "Sumo Humans",
    "multicomp/SumoHumans-v0": "Sumo Humans",
    "multicomp/YouShallNotPassHumans-v0": "You Shall Not Pass",
}

PRETTY_LABELS = {
    "Adv": "Adversary (Adv)",
    "Zoo": "Normal (Zoo)",
    "Rand": "Random (Rand)",
    "Zero": "Zero",
}

STYLES = {
    "paper": {
        "figure.figsize": (5.5, 7.5),
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "font.size": 9,
        "legend.fontsize": 9,
        "axes.unicode_minus": False,  # workaround bug with Unicode minus signs not appearing
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    },
    "slides": {"figure.figsize": (9.32, 3)},
    "slides_density": {"figure.figsize": (5, 3)},
    "poster": {
        "font.family": "sans-serif",
        "font.sans-serif": "Arial",
        "font.weight": "bold",
        "font.size": 14,
        "legend.fontsize": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    },
    "monolithic": {"figure.figsize": (5.5, 2.0625)},
    "twocol": {"figure.figsize": (2.75, 2.0625)},
    "threecol": {"figure.figsize": (1.83, 1.7)},
    "scores_monolithic": {"figure.figsize": (5.5, 1.4)},
    "scores_twocol": {"figure.figsize": (2.7, 1.6), "font.size": 8, "ytick.labelsize": 8},
    "scores_threecol": {"figure.figsize": (1.76, 1.6)},
    "density_twocol": {"figure.figsize": (2.7, 2.0625), "legend.fontsize": 8},
    "scores_poster_threecol": {"figure.figsize": (5.15, 3.1)},
    "a4": {"figure.figsize": (8.27, 11.69)},
}
