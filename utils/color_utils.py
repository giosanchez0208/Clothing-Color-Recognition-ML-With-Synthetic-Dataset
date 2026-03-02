import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Optional, Tuple

class ColorLibrary:
    """
    A utility class for loading, categorizing, and visualizing colors from the ISCC-NBS dataset.
    """

    def __init__(
        self,
        csv_path: str,
        color_categories: Dict[str, List[str]],
        name_col: str = 'ISCC-NBS Level 3 Name',
        hex_col: str = 'hex',
    ):
        """
        Parameters
        ----------
        csv_path : str
            Path to the CSV file containing color data.
        color_categories : dict
            Dictionary mapping category names (e.g., 'red') to lists of subwords
            that indicate that category (e.g., ['red', 'reddish']).
        name_col : str, default 'ISCC-NBS Level 3 Name'
            Name of the column containing the full color name.
        hex_col : str, default 'hex'
            Name of the column containing the hex color code.
        """
        self.df = pd.read_csv(csv_path)
        self.name_col = name_col
        self.hex_col = hex_col
        self.color_categories = color_categories

        # Build reverse mapping from subword -> category
        self.subword_to_category = {}
        for category, subwords in color_categories.items():
            for sw in subwords:
                self.subword_to_category[sw.lower()] = category

        # Build the color library (category -> list of hex codes)
        self.color_library = self._build_color_library()

    @classmethod
    def from_categorized_csv(cls, csv_path: str, category_col: str = 'category', hex_col: str = 'hex'):
        """
        Create a ColorLibrary instance from a CSV that already contains 'category' and 'hex' columns.
        This is useful for reloading data previously saved with `save_as_csv()`.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file.
        category_col : str, default 'category'
            Name of the column containing category labels.
        hex_col : str, default 'hex'
            Name of the column containing hex color codes.
        """
        df = pd.read_csv(csv_path)
        # Ensure required columns exist
        if category_col not in df.columns or hex_col not in df.columns:
            raise ValueError(f"CSV must contain columns '{category_col}' and '{hex_col}'.")

        # Create a new instance without calling __init__
        obj = cls.__new__(cls)
        # Set minimal attributes needed for the class to function
        obj.df = df  # store for potential future use
        obj.name_col = None  # not used
        obj.hex_col = hex_col
        obj.color_categories = {}  # not used for categorization now, but we'll populate from data
        obj.subword_to_category = {}  # not used, but kept for compatibility

        # Build the color library directly from the DataFrame
        categories = df[category_col].unique()
        obj.color_library = {}
        for cat in categories:
            hex_list = df[df[category_col] == cat][hex_col].tolist()
            obj.color_library[cat] = hex_list
            # Also rebuild a dummy color_categories for consistency
            obj.color_categories[cat] = []  # no subwords needed

        return obj
    
    def _build_color_library(self) -> Dict[str, List[str]]:
        """
        Iterate over all rows, determine the category for each color,
        and collect hex codes per category.
        """
        library = {cat: [] for cat in self.color_categories.keys()}

        for idx, row in self.df.iterrows():
            full_name = str(row[self.name_col]).lower()
            raw_color = None
            if self.hex_col in row.index:
                raw_color = row[self.hex_col]
            elif 'sRGB' in row.index:
                raw_color = row['sRGB']

            hex_code = self._normalize_color_code(raw_color)

            # Skip if hex is missing or invalid (optional)
            if not hex_code:
                continue

            # Split name into words (simple split on whitespace)
            words = full_name.split()

            # Find all words that are known subwords
            matched_subwords = [(i, w) for i, w in enumerate(words) if w in self.subword_to_category]

            if not matched_subwords:
                # No category found – optionally assign to 'unknown' or skip
                # Here we skip silently, but you could add a warning
                continue

            # Put in both dominant and secondary categories (if multiple subwords match, we take the last one as dominant)
            for _, subword in matched_subwords:
                category = self.subword_to_category[subword]
                library[category].append(hex_code)

        return library

    @staticmethod
    def _normalize_color_code(raw_color: object) -> Optional[str]:
        if raw_color is None or pd.isna(raw_color):
            return None

        color_text = str(raw_color).strip()
        if not color_text:
            return None

        if color_text.startswith('#'):
            return color_text

        parts = [p.strip() for p in color_text.split(',')]
        if len(parts) != 3:
            return None

        try:
            rgb_values = [max(0, min(255, int(float(p)))) for p in parts]
        except ValueError:
            return None

        return '#{:02x}{:02x}{:02x}'.format(*rgb_values)

    # ----------------------------------------------------------------------
    # Visualization methods (for Jupyter notebooks, not exported)
    # ----------------------------------------------------------------------

    def show_category_colors(self, category: str, ncols: int = 15, figsize: Tuple[int, int] = (15, 5)):
        """
        Display a grid of color swatches for all hex codes in a given category.
        """
        hex_list = self.color_library.get(category, [])
        if not hex_list:
            print(f"No colors found for category '{category}'.")
            return

        n = len(hex_list)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if nrows * ncols > 1 else [axes]

        for i, ax in enumerate(axes):
            if i < n:
                ax.add_patch(patches.Rectangle((0, 0), 1, 1, color=hex_list[i]))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(hex_list[i], fontsize=8)
            else:
                ax.axis('off')

        plt.suptitle(f"Category: {category} ({n} colors)", fontsize=14)
        plt.tight_layout()
        plt.show()

    def show_all_categories(self, max_per_category: int = 5, ncols: int = 5, figsize: Tuple[int, int] = (15, 10)):
        """
        Display a grid where each row shows a few representative colors from each category.
        """
        categories = list(self.color_library.keys())
        ncats = len(categories)
        nrows = (ncats + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if nrows * ncols > 1 else [axes]

        for idx, cat in enumerate(categories):
            ax = axes[idx]
            hex_list = self.color_library[cat][:max_per_category]
            # Draw a small strip of color patches
            for j, h in enumerate(hex_list):
                rect = patches.Rectangle((j, 0), 1, 1, color=h)
                ax.add_patch(rect)
            ax.set_xlim(0, max_per_category)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{cat} ({len(self.color_library[cat])})", fontsize=10)

        # Hide any unused subplots
        for idx in range(len(categories), len(axes)):
            axes[idx].axis('off')

        plt.suptitle("Color Categories Overview", fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_category_counts(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Bar chart showing the number of colors in each category.
        """
        cats = list(self.color_library.keys())
        counts = [len(self.color_library[cat]) for cat in cats]

        plt.figure(figsize=figsize)
        bars = plt.bar(cats, counts, color='skyblue')
        plt.xlabel('Category')
        plt.ylabel('Number of colors')
        plt.title('Color count per category')
        plt.xticks(rotation=45, ha='right')

        # Add value labels on top of bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(count), ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def plot_category_hue_lightness(
        self,
        category: str,
        size_by_saturation: bool = True,
        figsize: Tuple[int, int] = (8, 4),
        alpha: float = 0.8,
        show_centroid: bool = False,
        centroid_marker: str = '*',
        centroid_size: int = 200,
        centroid_color: str = 'black',
        draw_lines_to_centroid: bool = False,
        line_style: str = '--',
        line_alpha: float = 0.3,
        centroid_space: str = 'hsl'
    ):
        """
        Plot all colors in a given category on a 2D hue–lightness plane.
        The x-axis is hue (0-360°), y-axis is lightness (0 = black, 1 = white).
        Each point is colored with its actual RGB color.
        Optionally, point size can represent saturation (larger = more saturated).

        Parameters
        ----------
        category : str
            Name of the category (e.g., 'red', 'blue').
        size_by_saturation : bool, default True
            If True, point size is proportional to saturation (range 0–1 scaled to 20–200).
        figsize : tuple, optional
            Figure size (width, height) in inches.
        alpha : float, default 0.8
            Transparency of points.
        show_centroid : bool, default False
            If True, compute and display the centroid (average color) of the category.
        centroid_marker : str, default '*'
            Marker style for the centroid.
        centroid_size : int, default 200
            Size of the centroid marker.
        centroid_color : str, default 'black'
            Outline color of the centroid marker (the face will be the centroid's own color).
        draw_lines_to_centroid : bool, default False
            If True, draw faint lines from every point to the centroid.
        line_style : str, default '--'
            Style of the connector lines.
        line_alpha : float, default 0.3
            Transparency of the connector lines.
        centroid_space : {'rgb', 'hsl'}, default 'hsl'
            Color space in which to compute the centroid. Use 'hsl' to respect hue circularity.
        """
        import colorsys
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        hex_list = self.color_library.get(category, [])
        if not hex_list:
            print(f"No colors found for category '{category}'.")
            return

        # Prepare data
        hues = []
        saturations = []
        lightnesses = []
        rgb_colors = []

        for hex_code in hex_list:
            hex_code = hex_code.lstrip('#')
            r = int(hex_code[0:2], 16) / 255.0
            g = int(hex_code[2:4], 16) / 255.0
            b = int(hex_code[4:6], 16) / 255.0
            rgb_colors.append((r, g, b))

            h, l, s = colorsys.rgb_to_hls(r, g, b)   # returns (hue, lightness, saturation)
            hues.append(h * 360.0)
            lightnesses.append(l)
            saturations.append(s)

        if not hues:
            print("No valid HSL values could be computed.")
            return

        # Determine point sizes
        if size_by_saturation:
            sizes = [20 + 180 * s for s in saturations]  # scale to 20–200
        else:
            sizes = 50

        # Create plot
        plt.figure(figsize=figsize)

        # Optionally draw lines from points to centroid (before points so lines are behind)
        if show_centroid and draw_lines_to_centroid:
            # Get centroid RGB and compute its HSL
            centroid_rgb, centroid_hex = self.get_category_centroid(category, space=centroid_space)
            h_cent, l_cent, s_cent = colorsys.rgb_to_hls(*centroid_rgb)
            h_cent_deg = h_cent * 360.0

            for h, l in zip(hues, lightnesses):
                plt.plot([h, h_cent_deg], [l, l_cent],
                         color='gray', linestyle=line_style, alpha=line_alpha, linewidth=0.5)

        # Scatter plot of the points
        scatter = plt.scatter(
            hues, lightnesses,
            c=rgb_colors,          # color each point with its actual RGB
            s=sizes,
            alpha=alpha,
            edgecolors='none'
        )

        # Optionally add the centroid
        if show_centroid:
            centroid_rgb, centroid_hex = self.get_category_centroid(category, space=centroid_space)
            h_cent, l_cent, s_cent = colorsys.rgb_to_hls(*centroid_rgb)
            h_cent_deg = h_cent * 360.0

            plt.scatter(
                [h_cent_deg], [l_cent],
                marker=centroid_marker,
                s=centroid_size,
                color=centroid_hex,
                edgecolors=centroid_color,
                linewidths=1.5,
                zorder=5,           # ensure centroid is on top
                label='Centroid'
            )

        # Labels and limits
        plt.xlabel('Hue (degrees)')
        plt.ylabel('Lightness (0 = black, 1 = white)')
        plt.title(f'Hue–Lightness Distribution: {category} ({len(hex_list)} colors)')
        plt.xlim(0, 360)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.3)

        # Legend for saturation (if used)
        if size_by_saturation and not show_centroid:
            legend_elements = [
                Line2D([0], [0], marker='o', color='gray', label='Low saturation',
                       markerfacecolor='gray', markersize=5, linestyle='None'),
                Line2D([0], [0], marker='o', color='gray', label='High saturation',
                       markerfacecolor='gray', markersize=15, linestyle='None')
            ]
            plt.legend(handles=legend_elements, loc='upper right', title="Saturation")
        elif show_centroid:
            # If centroid is shown, include it in the legend automatically
            plt.legend(loc='upper right')

        plt.tight_layout()
        plt.show()
        
    # ----------------------------------------------------------------------
    # Additional utilities
    # ----------------------------------------------------------------------

    def save_as_csv(self, output_path: str):
        """
        Save the color library as a CSV file with columns 'category' and 'hex'.
        Each row corresponds to a single color sample and its assigned category.
        """
        rows = []
        for category, hex_list in self.color_library.items():
            for hex_code in hex_list:
                rows.append({'category': category, 'hex': hex_code})
        df_out = pd.DataFrame(rows)
        df_out.to_csv(output_path, index=False)
        print(f"Saved {len(rows)} color entries to {output_path}")
    
    def get_category_centroid(self, category: str, space: str = 'rgb') -> Tuple[Tuple[float, float, float], str]:
        """
        Compute the centroid (average) color of all samples in a category.
        
        Parameters
        ----------
        category : str
            Name of the category (e.g., 'red', 'blue').
        space : {'rgb', 'hsl'}, default 'rgb'
            Color space in which to average:
            - 'rgb' : simple linear average of R, G, B (fast, intuitive)
            - 'hsl' : average in HSL with circular mean for hue (handles hue wrap-around)
        
        Returns
        -------
        (r, g, b) : tuple of floats in [0, 1]
            The average RGB values.
        hex_code : str
            Hex representation of the average color (#rrggbb).
        """
        import colorsys
        import numpy as np
        from math import pi, sin, cos, atan2

        hex_list = self.color_library.get(category, [])
        if not hex_list:
            raise ValueError(f"No colors found for category '{category}'.")

        if space == 'rgb':
            # Average RGB components
            r_sum = g_sum = b_sum = 0.0
            for hex_code in hex_list:
                hex_code = hex_code.lstrip('#')
                r = int(hex_code[0:2], 16) / 255.0
                g = int(hex_code[2:4], 16) / 255.0
                b = int(hex_code[4:6], 16) / 255.0
                r_sum += r
                g_sum += g
                b_sum += b
            n = len(hex_list)
            r_avg = r_sum / n
            g_avg = g_sum / n
            b_avg = b_sum / n
            rgb_avg = (r_avg, g_avg, b_avg)

        elif space == 'hsl':
            # Collect HSL values
            hues = []
            saturations = []
            lightnesses = []
            for hex_code in hex_list:
                hex_code = hex_code.lstrip('#')
                r = int(hex_code[0:2], 16) / 255.0
                g = int(hex_code[2:4], 16) / 255.0
                b = int(hex_code[4:6], 16) / 255.0
                h, l, s = colorsys.rgb_to_hls(r, g, b)   # h in [0,1]
                hues.append(h * 2 * pi)                  # convert to radians for circular mean
                saturations.append(s)
                lightnesses.append(l)

            # Circular mean of hue
            sin_sum = sum(sin(h) for h in hues)
            cos_sum = sum(cos(h) for h in hues)
            n = len(hues)
            hue_avg_rad = atan2(sin_sum / n, cos_sum / n)
            if hue_avg_rad < 0:
                hue_avg_rad += 2 * pi
            hue_avg = hue_avg_rad / (2 * pi)              # back to [0,1]

            # Linear averages for saturation and lightness
            sat_avg = sum(saturations) / n
            light_avg = sum(lightnesses) / n

            # Convert back to RGB
            r_avg, g_avg, b_avg = colorsys.hls_to_rgb(hue_avg, light_avg, sat_avg)
            rgb_avg = (r_avg, g_avg, b_avg)

        else:
            raise ValueError("space must be 'rgb' or 'hsl'")

        # Convert to hex
        hex_avg = '#{:02x}{:02x}{:02x}'.format(
            int(round(r_avg * 255)),
            int(round(g_avg * 255)),
            int(round(b_avg * 255))
        )

        return rgb_avg, hex_avg
    
    def get_category_of_color(self, color_name: str) -> Optional[str]:
        """
        Given a full color name (e.g., 'dark red'), return the category it belongs to.
        Uses the same logic as the library building (last matching subword).
        """
        words = color_name.lower().split()
        matched = [(i, w) for i, w in enumerate(words) if w in self.subword_to_category]
        if not matched:
            return None
        last_subword = max(matched, key=lambda x: x[0])[1]
        return self.subword_to_category[last_subword]

    def get_hex_list(self, category: str) -> List[str]:
        """Return all hex codes for a given category."""
        return self.color_library.get(category, [])

    def summary(self):
        """Print a quick summary of the color library."""
        for cat, hexes in self.color_library.items():
            print(f"{cat}: {len(hexes)} colors")