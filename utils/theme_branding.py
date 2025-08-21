import os
from cycler import cycler
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from fontTools.ttLib import TTFont


# Add all TTF/OTF fonts from the directory
def load_fonts(font_dir: str) -> None:
    """Loads all .tff, .otf fonts from a folder"""
    fonts = []
    for fname in os.listdir(font_dir):
        if fname.lower().endswith((".ttf", ".otf")):
            font_path = os.path.join(font_dir, fname)
            fm.fontManager.addfont(font_path)
            fonts.append(font_path)
    return fonts


fonts = load_fonts("utils/fonts/brisbane-font-family")

# Get the actual font family name from one of the files

font_name = None
if fonts:
    ttfont = TTFont(fonts[0])
    name_record = ttfont["name"].getName(1, 3, 1)  # Font Family name
    font_name = name_record.toStr()
    print(f"Using font family: {font_name}")


def set_figure_style(axes_colour: str, prop_cycle: list[str]):
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": axes_colour,
            "axes.labelcolor": axes_colour,
            "xtick.color": axes_colour,
            "ytick.color": axes_colour,
            "text.color": axes_colour,
            "grid.color": "#F2F2F0",
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "axes.grid": True,
            "axes.prop_cycle": cycler("color", prop_cycle),
        }
    )
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

 




axes_colour = "#000000"
BRAND_COLOURS = [
    "#FF417B",
    "#00C8FF",
    "#8AFF7B",
    "#FF8B00",
    "#FFF75C",
    "#D6FF36",
]
set_figure_style(axes_colour, BRAND_COLOURS)


def set_font_to_Brisbane() -> None:
    # Use the detected font family name
    plt.rcParams.update(
        {
            "font.family": font_name if font_name else "Brisbane",
            "font.serif": [font_name if font_name else "Brisbane"],
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        }
    )
set_font_to_Brisbane()

if __name__ == "__main__":
    plt.figure()
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.title("Sample Plot")
    plt.xlabel("X-axis Label")
    plt.ylabel("Y-axis Label")
    plt.show()
