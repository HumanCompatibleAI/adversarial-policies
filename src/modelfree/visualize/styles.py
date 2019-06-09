PRETTY_LABELS = {
    'Adv': 'Adversary (Adv)',
    'Zoo': 'Normal (Zoo)',
    'Rand': 'Random (Rand)',
    'Zero': 'Zero',
}

STYLES = {
    'paper': {
        'figure.figsize': (5.5, 7.5),
        'font.serif': 'Times New Roman',
        'font.family': 'serif',
        'font.size': 9,
        'legend.fontsize': 9,
        'axes.unicode_minus': False,  # workaround bug with Unicode minus signs not appearing
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
    },
    'slides': {
        'figure.figsize': (9.32, 3),
    },
    'monolithic': {
        'figure.figsize': (5.5, 2.0625),
    },
    'twocol': {
        'figure.figsize': (2.75, 2.0625),
    },
    'threecol': {
        'figure.figsize': (1.83, 1.7),
    },
    'scores_monolithic': {
        'figure.figsize': (5.5, 1.4),
    },
    'scores_twocol': {
        'figure.figsize': (2.7, 1.6),
        'ytick.labelsize': 8,
    },
    'scores_threecol': {
        'figure.figsize': (1.76, 1.6),
    },
    'density_twocol': {
        'figure.figsize': (2.7, 2.0625),
        'legend.fontsize': 8,
    },
    'a4': {
        'figure.figsize': (8.27, 11.69),
    },
}
