# =============================================================================
# color_utils.py — ColorLibrary
# =============================================================================
# Construction
#   __init__(csv_path, color_categories, name_col='ISCC-NBS Level 3 Name', hex_col='hex')
#     Build library from raw ISCC-NBS CSV via subword matching on color names.
#   from_categorized_csv(csv_path, category_col='category')  [classmethod]
#     Reload from a CSV previously saved by save_as_csv() (expects L, a, b columns).
#
# Visualization
#   plot_category_hue_lightness(category, size_by_chroma=True, figsize=(8,4), alpha=0.8,
#                               show_centroid=False, centroid_marker='*', centroid_size=200,
#                               centroid_color='black', draw_lines_to_centroid=False,
#                               line_style='--', line_alpha=0.3)
#     CIELAB h_ab vs L* scatter for one category. Point size encodes chroma C*_ab.
#   plot_category_lab_3d(category, size_by_chroma=True, figsize=(8,6), alpha=0.8,
#                        show_centroid=False, centroid_marker='*', centroid_size=200,
#                        centroid_color='black', draw_lines_to_centroid=False,
#                        line_style='--', line_alpha=0.3, elev=20.0, azim=45.0)
#     Interactive CIELAB 3-D scatter (a*, b*, L* axes). elev/azim set initial viewing angle.
#   show_all_categories(max_per_category=5, ncols=5, figsize=(15,10))
#     Swatch overview grid across all categories.
#   show_color_group(colors, space='auto', ncols=15, figsize=(15,3))
#     Swatch grid for a category name (str) or an explicit list of colors (hex/lab/srgb).
#   show_single_color(color, space='auto', figsize=(2,2))
#     Display a single color swatch. space: 'auto' | 'hex' | 'lab' | 'srgb' (normalized 0-1).
#     'auto' infers: string -> hex; first value > 1 -> lab; else -> srgb.
#
# Properties
#   categories  [read-only]  -> List[str]  list of category names
#   color_library             -> Dict[str, List[str]]  category -> hex list; settable
#   color_categories          -> Dict[str, List[str]]  subword map; setter rebuilds subword index
#
# Analysis
#   get_category_centroid(category) -> (L*, a*, b*)
#     Centroid averaged in CIELAB space; perceptually uniform.
#   get_category_colors(category) -> List[Tuple[L*, a*, b*]]
#     All colors in a category as CIELAB tuples.
#   get_category_mahalanobis_distances(category) -> List[Tuple[str, float]]
#     All colors sorted by Mahalanobis distance from the category centroid.
#     Returns [(hex, distance), ...] ascending. Falls back to Euclidean if covariance
#     is singular (e.g. fewer than 4 distinct points).
#   get_color_mahalanobis_distance(category, L, a, b) -> float
#     Mahalanobis distance of a single CIELAB point from its category's distribution.
#     Useful for assigning/scoring an unknown color against a trained category.
#   summary()
#     Print color count per category.
#
# Distance from centroid — Mahalanobis in CIELAB
#   Colors in each category are represented as points in the 3-D CIELAB space
#   (L*, a*, b*).  The category's covariance matrix S is computed over all member
#   points, capturing both per-axis variance and the axis correlations within that
#   hue family.  The Mahalanobis distance for a point x from centroid μ is:
#
#       d_M(x, μ) = sqrt( (x − μ)ᵀ · S⁻¹ · (x − μ) )
#
#   Why Mahalanobis over Euclidean ΔE?  CIELAB is perceptually uniform but a
#   category's natural spread is never isotropic: a warm-red group may vary widely
#   in lightness yet be tightly clustered in hue angle.  S⁻¹ normalises each
#   direction by its actual spread, so d_M = 1 means "one standard deviation away
#   along the principal axis of that category", regardless of how elongated or
#   tilted the cloud is.  This makes distances comparable across categories and
#   usable as a likelihood proxy (assuming a multivariate Gaussian distribution).
#
#   Fallback: if S is singular (fewer than 4 points, or degenerate geometry) the
#   ordinary Euclidean distance sqrt(ΔL*² + Δa*² + Δb*²) is returned instead,
#   and a warning is printed.
#
# I/O
#   save_as_csv(output_path)
#     Save library as CSV with 'category', 'L', 'a', 'b' columns (CIELAB).
#
# Internal CIELAB helpers (sRGB/D65, IEC 61966-2-1)
#   _hex_to_lab(hex_code) -> (L*, a*, b*)
#   _lab_to_hex(L*, a*, b*) -> ((r,g,b), '#rrggbb')
# =============================================================================

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from typing import Dict, List, Optional, Tuple


class ColorLibrary:

    _D65 = (0.95047, 1.00000, 1.08883)  # D65 reference white in XYZ

    def __init__(
        self,
        csv_path: str,
        color_categories: Dict[str, List[str]],
        name_col: str = 'ISCC-NBS Level 3 Name',
        hex_col: str = 'hex',
    ):
        self.df = pd.read_csv(csv_path)
        self.name_col = name_col
        self.hex_col = hex_col
        self.color_categories = color_categories  # uses setter, builds subword index
        self._color_library = self._build_color_library()

    @classmethod
    def from_categorized_csv(cls, csv_path: str, category_col: str = 'category'):
        df = pd.read_csv(csv_path)
        for col in (category_col, 'L', 'a', 'b'):
            if col not in df.columns:
                raise ValueError(f"CSV must contain '{col}' column.")
        obj = cls.__new__(cls)
        obj.df = df
        obj.name_col = None
        obj.hex_col = None
        obj.subword_to_category = {}
        obj._color_library = {}
        for cat in df[category_col].unique():
            rows = df[df[category_col] == cat]
            obj._color_library[cat] = [
                cls._lab_to_hex(r['L'], r['a'], r['b'])[1]
                for _, r in rows.iterrows()
            ]
        obj.color_categories = {cat: [] for cat in obj._color_library}  # uses setter
        return obj

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def categories(self) -> List[str]:
        return list(self._color_library.keys())

    @property
    def color_library(self) -> Dict[str, List[str]]:
        return self._color_library

    @color_library.setter
    def color_library(self, value: Dict[str, List[str]]):
        if not isinstance(value, dict):
            raise TypeError("color_library must be a dict.")
        self._color_library = value

    @property
    def color_categories(self) -> Dict[str, List[str]]:
        return self._color_categories

    @color_categories.setter
    def color_categories(self, value: Dict[str, List[str]]):
        if not isinstance(value, dict):
            raise TypeError("color_categories must be a dict.")
        self._color_categories = value
        self.subword_to_category = {
            sw.lower(): cat
            for cat, subwords in value.items()
            for sw in subwords
        }

    def _build_color_library(self) -> Dict[str, List[str]]:
        library = {cat: [] for cat in self.color_categories}
        for _, row in self.df.iterrows():
            full_name = str(row[self.name_col]).lower()
            raw = row[self.hex_col] if self.hex_col in row.index else row.get('sRGB')
            hex_code = self._normalize_hex(raw)
            if not hex_code:
                continue
            for _, subword in [(i, w) for i, w in enumerate(full_name.split()) if w in self.subword_to_category]:
                library[self.subword_to_category[subword]].append(hex_code)
        return library

    @staticmethod
    def _normalize_hex(raw) -> Optional[str]:
        if raw is None or pd.isna(raw):
            return None
        raw = str(raw).strip()
        if raw.startswith('#'):
            return raw
        parts = [p.strip() for p in raw.split(',')]
        if len(parts) != 3:
            return None
        try:
            return '#{:02x}{:02x}{:02x}'.format(*[max(0, min(255, int(float(p)))) for p in parts])
        except ValueError:
            return None

    # -------------------------------------------------------------------------
    # CIELAB conversion — sRGB/D65, IEC 61966-2-1
    # -------------------------------------------------------------------------

    @classmethod
    def _hex_to_lab(cls, hex_code: str) -> Tuple[float, float, float]:
        h = hex_code.lstrip('#')
        r, g, b = (int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        def expand(c): return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        r, g, b = expand(r), expand(g), expand(b)
        X = 0.4124564*r + 0.3575761*g + 0.1804375*b
        Y = 0.2126729*r + 0.7151522*g + 0.0721750*b
        Z = 0.0193339*r + 0.1191920*g + 0.9503041*b
        Xn, Yn, Zn = cls._D65
        def f(t): return t ** (1/3) if t > (6/29)**3 else (29/6)**2 * t / 3 + 4/29
        fx, fy, fz = f(X/Xn), f(Y/Yn), f(Z/Zn)
        return 116*fy - 16, 500*(fx - fy), 200*(fy - fz)

    @classmethod
    def _lab_to_hex(cls, L: float, a: float, b: float) -> Tuple[Tuple[float, float, float], str]:
        Xn, Yn, Zn = cls._D65
        fy = (L + 16) / 116
        fx, fz = a / 500 + fy, fy - b / 200
        def finv(t): return t**3 if t > 6/29 else 3*(6/29)**2*(t - 4/29)
        X, Y, Z = Xn*finv(fx), Yn*finv(fy), Zn*finv(fz)
        rl =  3.2404542*X - 1.5371385*Y - 0.4985314*Z
        gl = -0.9692660*X + 1.8760108*Y + 0.0415560*Z
        bl =  0.0556434*X - 0.2040259*Y + 1.0572252*Z
        def compress(c): return max(0.0, min(1.0, 12.92*c if c <= 0.0031308 else 1.055*c**(1/2.4) - 0.055))
        r, g, b = compress(rl), compress(gl), compress(bl)
        return (r, g, b), '#{:02x}{:02x}{:02x}'.format(int(round(r*255)), int(round(g*255)), int(round(b*255)))

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def plot_category_hue_lightness(
        self,
        category: str,
        size_by_chroma: bool = True,
        figsize: Tuple[int, int] = (8, 4),
        alpha: float = 0.8,
        show_centroid: bool = False,
        centroid_marker: str = '*',
        centroid_size: int = 200,
        centroid_color: str = 'black',
        draw_lines_to_centroid: bool = False,
        line_style: str = '--',
        line_alpha: float = 0.3,
    ):
        """CIELAB h_ab vs L* scatter. Point size = chroma C*_ab (when size_by_chroma=True)."""
        hex_list = self.color_library.get(category, [])
        if not hex_list:
            print(f"No colors found for '{category}'.")
            return

        hues, lightnesses, chromas, rgb_colors = [], [], [], []
        for hx in hex_list:
            s = hx.lstrip('#')
            rgb_colors.append(tuple(int(s[i:i+2], 16) / 255.0 for i in (0, 2, 4)))
            L, a_, b_ = self._hex_to_lab(hx)
            hues.append(math.degrees(math.atan2(b_, a_)) % 360.0)
            lightnesses.append(L)
            chromas.append(math.sqrt(a_**2 + b_**2))

        max_c = max(chromas) or 1.0
        sizes = [20 + 180 * (c / max_c) for c in chromas] if size_by_chroma else 50

        plt.figure(figsize=figsize)

        if show_centroid and draw_lines_to_centroid:
            c_lab = self.get_category_centroid(category)
            _, c_hex = self._lab_to_hex(*c_lab)
            L_c, a_c, b_c = c_lab
            h_c = math.degrees(math.atan2(b_c, a_c)) % 360.0
            for h, l in zip(hues, lightnesses):
                plt.plot([h, h_c], [l, L_c], color='gray', linestyle=line_style,
                         alpha=line_alpha, linewidth=0.5)

        plt.scatter(hues, lightnesses, c=rgb_colors, s=sizes, alpha=alpha, edgecolors='none')

        if show_centroid:
            c_lab = self.get_category_centroid(category)
            L_c, a_c, b_c = c_lab
            _, c_hex = self._lab_to_hex(*c_lab)
            h_c = math.degrees(math.atan2(b_c, a_c)) % 360.0
            plt.scatter([h_c], [L_c], marker=centroid_marker, s=centroid_size,
                        color=c_hex, edgecolors=centroid_color, linewidths=1.5, zorder=5, label='Centroid')
            plt.legend(loc='upper right')
        elif size_by_chroma:
            plt.legend(handles=[
                Line2D([0], [0], marker='o', color='gray', markersize=5,  linestyle='None', label='Low chroma'),
                Line2D([0], [0], marker='o', color='gray', markersize=15, linestyle='None', label='High chroma'),
            ], loc='upper right', title='Chroma (C*_ab)')

        plt.xlabel('CIELAB hue angle h_ab (°)')
        plt.ylabel('CIELAB lightness L*')
        plt.title(f'CIELAB h_ab vs L*: {category} ({len(hex_list)} colors)')
        plt.xlim(0, 360)
        plt.ylim(0, 100)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_category_lab_3d(
        self,
        category: str,
        size_by_chroma: bool = True,
        figsize: Tuple[int, int] = (8, 6),
        alpha: float = 0.8,
        show_centroid: bool = False,
        centroid_marker: str = '*',
        centroid_size: int = 200,
        centroid_color: str = 'black',
        draw_lines_to_centroid: bool = False,
        line_style: str = '--',
        line_alpha: float = 0.3,
        elev: float = 20.0,
        azim: float = 45.0,
    ):
        """CIELAB 3-D scatter (L*, a*, b* axes). Point colour = actual sRGB; size = chroma C*_ab."""
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3-D projection

        hex_list = self.color_library.get(category, [])
        if not hex_list:
            print(f"No colors found for '{category}'.")
            return

        Ls, As, Bs, chromas, rgb_colors = [], [], [], [], []
        for hx in hex_list:
            s = hx.lstrip('#')
            rgb_colors.append(tuple(int(s[i:i+2], 16) / 255.0 for i in (0, 2, 4)))
            L, a_, b_ = self._hex_to_lab(hx)
            Ls.append(L)
            As.append(a_)
            Bs.append(b_)
            chromas.append(math.sqrt(a_**2 + b_**2))

        max_c = max(chromas) or 1.0
        sizes = [20 + 180 * (c / max_c) for c in chromas] if size_by_chroma else 50

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=elev, azim=azim)

        if show_centroid and draw_lines_to_centroid:
            L_c, a_c, b_c = self.get_category_centroid(category)
            for L, a, b in zip(Ls, As, Bs):
                ax.plot([a, a_c], [b, b_c], [L, L_c], color='gray',
                        linestyle=line_style, alpha=line_alpha, linewidth=0.5)

        ax.scatter(As, Bs, Ls, c=rgb_colors, s=sizes, alpha=alpha, edgecolors='none', depthshade=True)

        if show_centroid:
            L_c, a_c, b_c = self.get_category_centroid(category)
            _, c_hex = self._lab_to_hex(L_c, a_c, b_c)

            # Orthogonal drop-lines from centroid to each coordinate plane
            proj_kw = dict(color=centroid_color, linestyle=':', linewidth=1.2, alpha=0.7)
            ax.plot([a_c, a_c], [b_c, b_c], [0,   L_c], **proj_kw)   # → a*-b* floor (L*=0)
            ax.plot([a_c, a_c], [-128, b_c], [L_c, L_c], **proj_kw)  # → b*-L* wall  (a*=-128)
            ax.plot([-128, a_c], [b_c, b_c], [L_c, L_c], **proj_kw)  # → a*-L* wall  (b*=-128)

            # Small projected dots on each plane
            ax.scatter([a_c], [b_c], [0],    color=centroid_color, s=30, marker='x', zorder=4)
            ax.scatter([a_c], [-128], [L_c], color=centroid_color, s=30, marker='x', zorder=4)
            ax.scatter([-128], [b_c], [L_c], color=centroid_color, s=30, marker='x', zorder=4)

            ax.scatter([a_c], [b_c], [L_c], marker=centroid_marker, s=centroid_size,
                       color=c_hex, edgecolors=centroid_color, linewidths=1.5, zorder=5, label='Centroid')
            ax.legend(loc='upper right')
        elif size_by_chroma:
            ax.legend(handles=[
                Line2D([0], [0], marker='o', color='gray', markersize=5,  linestyle='None', label='Low chroma'),
                Line2D([0], [0], marker='o', color='gray', markersize=15, linestyle='None', label='High chroma'),
            ], loc='upper right', title='Chroma (C*_ab)')

        ax.set_xlabel('a* (green–red)')
        ax.set_ylabel('b* (blue–yellow)')
        ax.set_zlabel('L* (lightness)')
        ax.set_title(f'CIELAB 3-D: {category} ({len(hex_list)} colors)')
        ax.set_xlim(-128, 128)
        ax.set_ylim(-128, 128)
        ax.set_zlim(0, 100)
        plt.tight_layout()
        plt.show()

    def show_color_group(self, colors, space: str = 'auto', ncols: int = 15, figsize: Tuple[int, int] = (15, 3)):
        """Swatch grid for a category name (str) or an explicit list of colors (hex/lab/srgb)."""
        if isinstance(colors, str):
            title = colors
            hex_list = self._color_library.get(colors, [])
            if not hex_list:
                print(f"No colors found for '{colors}'.")
                return
        else:
            title = f"{len(colors)} colors"
            hex_list = []
            for c in colors:
                sp = space
                if sp == 'auto':
                    sp = 'hex' if isinstance(c, str) else ('lab' if c[0] > 1.0 else 'srgb')
                if sp == 'hex':
                    hex_list.append(c)
                elif sp == 'lab':
                    hex_list.append(self._lab_to_hex(*c)[1])
                elif sp == 'srgb':
                    r, g, b = c
                    hex_list.append('#{:02x}{:02x}{:02x}'.format(int(round(r*255)), int(round(g*255)), int(round(b*255))))
        n = len(hex_list)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if nrows * ncols > 1 else [axes]
        for i, ax in enumerate(axes):
            if i < n:
                h = hex_list[i].lstrip('#')
                rgb = [int(h[j:j+2], 16) / 255.0 for j in (0, 2, 4)]
                ax.imshow([[rgb]])
            ax.axis('off')
        plt.suptitle(f"{title}", fontsize=12)
        plt.tight_layout()
        plt.show()

    def show_all_categories(self, max_per_category: int = 5, ncols: int = 5, figsize: Tuple[int, int] = (15, 10)):
        """Swatch overview grid — each cell shows up to max_per_category sample colors per category."""
        categories = list(self.color_library.keys())
        nrows = (len(categories) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if nrows * ncols > 1 else [axes]
        for idx, cat in enumerate(categories):
            ax = axes[idx]
            for j, h in enumerate(self.color_library[cat][:max_per_category]):
                ax.add_patch(patches.Rectangle((j, 0), 1, 1, color=h))
            ax.set_xlim(0, max_per_category)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{cat} ({len(self.color_library[cat])})", fontsize=10)
        for idx in range(len(categories), len(axes)):
            axes[idx].axis('off')
        plt.suptitle("Color Categories Overview", fontsize=14)
        plt.tight_layout()
        plt.show()

    def show_single_color(self, color, space: str = 'auto', figsize: Tuple[int, int] = (2, 2)):
        """Display a single color swatch. space: 'auto' | 'hex' | 'lab' | 'srgb' (normalized 0-1)."""
        if space == 'auto':
            if isinstance(color, str):
                space = 'hex'
            elif color[0] > 1.0:
                space = 'lab'
            else:
                space = 'srgb'

        if space == 'hex':
            hex_code = color
        elif space == 'lab':
            _, hex_code = self._lab_to_hex(*color)
        elif space == 'srgb':
            r, g, b = color
            hex_code = '#{:02x}{:02x}{:02x}'.format(int(round(r*255)), int(round(g*255)), int(round(b*255)))
        else:
            raise ValueError("space must be 'auto', 'hex', 'lab', or 'srgb'")

        h = hex_code.lstrip('#')
        rgb = [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
        plt.figure(figsize=figsize)
        plt.imshow([[rgb]])
        plt.axis('off')
        plt.title(f"{space.upper()}: {color}", fontsize=10)
        plt.tight_layout()
        plt.show()
        
    # -------------------------------------------------------------------------
    # Analysis & I/O
    # -------------------------------------------------------------------------

    def get_category_centroid(self, category: str) -> Tuple[float, float, float]:
        """Perceptual centroid of a category as (L*, a*, b*), averaged in CIELAB space."""
        hex_list = self.color_library.get(category, [])
        if not hex_list:
            raise ValueError(f"No colors found for '{category}'.")
        n = len(hex_list)
        L_sum, a_sum, b_sum = 0.0, 0.0, 0.0
        for hx in hex_list:
            L, a, b = self._hex_to_lab(hx)
            L_sum += L; a_sum += a; b_sum += b
        return (L_sum / n, a_sum / n, b_sum / n)

    def get_category_colors(self, category: str) -> List[Tuple[float, float, float]]:
        """All colors in a category as a list of (L*, a*, b*) CIELAB tuples."""
        hex_list = self.color_library.get(category, [])
        if not hex_list:
            raise ValueError(f"No colors found for '{category}'.")
        return [self._hex_to_lab(hx) for hx in hex_list]

    def _category_inv_cov(self, category: str):
        """Return (centroid_array, inv_cov_matrix) for a category, or None if singular."""
        hex_list = self.color_library.get(category, [])
        points = np.array([self._hex_to_lab(hx) for hx in hex_list])  # (n, 3)
        centroid = points.mean(axis=0)
        if len(points) < 4:
            return centroid, None
        cov = np.cov(points.T)  # (3, 3)
        try:
            inv_cov = np.linalg.inv(cov)
            # Sanity-check: reject near-singular matrices
            if not np.all(np.isfinite(inv_cov)):
                raise np.linalg.LinAlgError("non-finite inverse")
        except np.linalg.LinAlgError:
            inv_cov = None
        return centroid, inv_cov

    def get_category_mahalanobis_distances(self, category: str) -> List[Tuple[str, float]]:
        """Return [(hex, mahalanobis_distance), ...] sorted ascending by distance from centroid.

        Uses the category's own covariance matrix so that the spread and axis
        correlations of the colour cloud are taken into account (see module header
        for a full explanation).  Falls back to Euclidean distance when the
        covariance matrix is singular.
        """
        hex_list = self.color_library.get(category, [])
        if not hex_list:
            raise ValueError(f"No colors found for '{category}'.")
        centroid, inv_cov = self._category_inv_cov(category)
        results = []
        for hx in hex_list:
            p = np.array(self._hex_to_lab(hx))
            diff = p - centroid
            if inv_cov is not None:
                dist = float(math.sqrt(max(0.0, diff @ inv_cov @ diff)))
            else:
                dist = float(np.linalg.norm(diff))  # Euclidean fallback
            results.append((hx, dist))
        return sorted(results, key=lambda x: x[1])

    def get_color_mahalanobis_distance(self, category: str, L: float, a: float, b: float) -> float:
        """Mahalanobis distance of a single CIELAB point from the named category's distribution.

        Useful for scoring / classifying an unknown colour against a trained
        category (lower = more typical of that category).
        Falls back to Euclidean when the covariance matrix is singular.
        """
        centroid, inv_cov = self._category_inv_cov(category)
        diff = np.array([L, a, b]) - centroid
        if inv_cov is not None:
            return float(math.sqrt(max(0.0, diff @ inv_cov @ diff)))
        return float(np.linalg.norm(diff))

    def save_as_csv(self, output_path: str):
        """Save the library as a CSV with 'category', 'L', 'a', 'b' columns (CIELAB)."""
        rows = [
            {'category': cat, 'L': L, 'a': a, 'b': b}
            for cat, hexes in self.color_library.items()
            for hx in hexes
            for L, a, b in [self._hex_to_lab(hx)]
        ]
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"Saved {len(rows)} entries to {output_path}")

    def summary(self):
        """Print color count per category."""
        for cat, hexes in self.color_library.items():
            print(f"{cat}: {len(hexes)} colors")
