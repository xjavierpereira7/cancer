# config_visual.py

# Parámetros globales de escalado
config = {
    "general": {
        "escala": 1.0,
        "figsize": lambda esc: (8 * esc, 6 * esc),
    },
    "barras": {
        "escala": 0.55,
        "font_scale": 2.0,
        "figsize": lambda esc: (6 * esc, 3 * esc),
        "fontsize_title": lambda esc, fs: int(14 * fs * esc),
        "fontsize_labels": lambda esc, fs: int(10 * fs * esc),
        "fontsize_ticks": lambda esc, fs: int(8 * fs * esc),
        "height": lambda esc, fs: 200 * fs,
        "width": lambda esc, fs: 100 * fs,
    },
    "heatmap": {
        "escala_heatmap": 0.75,
        "font_scale_heatmap": 0.5,
        "figsize_heatmap": lambda esc: (5 * esc, 2 * esc),
        "fontsize_title_heatmap": lambda esc, fs: int(14 * fs * esc),
        "fontsize_labels_heatmap": lambda esc, fs: int(10 * fs * esc),
        "fontsize_ticks_heatmap": lambda esc, fs: int(8 * fs * esc),
        "height_heatmap": lambda esc, fs: 200 * fs,
        "width_heatmap": lambda esc, fs: 100 * fs,
    },
    "mapa": {
        "escala": 0.5,
        "font_scale": 0.55,
        "fontsize_title": lambda esc, fs: int(14 * fs * esc),
        "fontsize_labels": lambda esc, fs: int(10 * fs * esc),
        "fontsize_ticks": lambda esc, fs: int(8 * fs * esc),
        "height": lambda esc: 400 * esc,
        "width": lambda esc: 500 * esc,
    },
    "dispersion": {
        "escala_d": 0.5,
        "font_scale_d": 0.75,
        "fontsize_title_d": lambda esc, fs: int(14 * fs * esc),
        "fontsize_labels_d": lambda esc, fs: int(10 * fs * esc),
        "height": lambda esc: 250 * esc,
        "width": lambda esc: 250 * esc,
    },
    "mapa_j": {
        "escala_j": 0.75,
        "font_scale_j": 0.5,
        "figsize_j": lambda esc: (6 * esc, 3 * esc),
        "fontsize_title_j": lambda esc, fs: int(14 * fs * esc),
        "fontsize_labels_j": lambda esc, fs: int(10 * fs * esc),
        "fontsize_ticks_j": lambda esc, fs: int(8 * fs * esc),
        "height_j": lambda esc: 200 * esc,
        "width_j": lambda esc: 400 * esc,
    },
    "lineas": {
        "escala_lineas": 1.0,
        "font_scale_lineas": 1.0,
        "figsize_lineas": lambda esc: (7 * esc, 5 * esc),
        "fontsize_title_lineas": lambda esc, fs: int(16 * fs * esc),
        "fontsize_labels_lineas": lambda esc, fs: int(12 * fs * esc),
        "fontsize_ticks_lineas": lambda esc, fs: int(10 * fs * esc),
        "line_width_lineas": lambda esc: 4 * esc,
        "marker_size_lineas": lambda esc: 6 * esc,
        "height_lineas": lambda esc: 500 * esc,
        "width_lineas": lambda esc: 500 * esc,
    }
}
