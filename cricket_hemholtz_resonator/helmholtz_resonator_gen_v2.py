import math
from solid import *
from solid import scad_render_to_file, sphere, translate, cylinder, rotate
from solid.utils import *
from stl import mesh
import numpy as np
import subprocess
import pdb
import mingus.core.chords as chords
from mingus.core import notes
import sympy as sp
import plotly.graph_objects as go

# %%
CHORD = "Emin"


# %%
def note_to_frequency(note, octave, A4_key_number=49, A4_frequency=440):
    """
    A4 is the reference note with frequency 440 Hz
    and is the 49th key on the standard piano
    """
    key_number = (
        notes.note_to_int(note)
        - notes.note_to_int("A")
        + 12 * (octave - 4)
        + A4_key_number
    )
    # returns frequency in Hertz
    return A4_frequency * (2 ** ((key_number - A4_key_number) / 12))


def chord_frequencies(chord_name, octave=4):
    chord_notes = chords.from_shorthand(chord_name)
    return [
        (note + str(octave), note_to_frequency(note, octave)) for note in chord_notes
    ]


# %% [markdown]
# Quick unit test (https://mixbutton.com/mixing-articles/music-note-to-frequency-chart/)

# %%
assert round(note_to_frequency("C", 8), 2) == 4186.01
assert round(note_to_frequency("C", 4), 2) == 261.63
assert round(note_to_frequency("D", 1), 2) == 36.71

# %% [markdown]
# Lets start with E minor since we're roughly shooting for 6.5 kHz...

# %%
octave = 8
chord_frequencies_defined = chord_frequencies(CHORD, octave=octave)
print(f"{CHORD} in {octave} octave is: ", chord_frequencies_defined)

# %% [markdown]
# ## Helmholtz dimensions to suit specific frequencies
#


# %%
def calculate_L(
    frequency,
    c_value=343,  # Speed of sound in air (m/s)
    end_correction=lambda L, d: (L + 0.3 * d),
):
    """
    https://en.wikipedia.org/wiki/Helmholtz_resonance
    f = (c / 2pi) sqrt(A / V(L + 0.3d))

    f is the resonant frequency (Hertz),
    c is the speed of sound in air (approximately 343 meters per second at room temperature) (m/s)
    A is the cross-sectional area of the neck (m^2)
    V is the volume of the cavity (m^3)
    L is the physical length of the neck (m)
    d is the diameter of the neck (m)
    """
    f, c, L, d, r_cavity = sp.symbols("f c L d r_cavity")

    # Area of the neck (A) and Volume of the cavity (V)
    r_neck = d / 2
    A = sp.pi * r_neck**2  # Cross-sectional area of the neck
    V = (4 / 3) * sp.pi * r_cavity**3  # Volume of the cavity

    # Helmholtz resonator formula
    equation = sp.Eq(f, (c / (2 * sp.pi)) * sp.sqrt(A / (V * end_correction(L, d))))
    return (
        sp.solve(equation.subs({c: c_value, f: frequency, A: A, V: V}), L)[0],
        r_cavity,
        d,
    )


def good_results_metric(r_cav, neck_d, neck_l):
    if neck_d < 0 or r_cav < 0 or neck_l < 0:
        # Unphysical, lower than 0 measurements
        return -2
    if neck_d > (1.5 * r_cav):
        # Unphysical, the neck is wider than the body (give it 1.5 for a bit of margin)
        return -1
    if neck_l > 1.5 * r_cav:
        # Purely aesthetic (and some issue with manufacturing) but the neck is too long!
        return 0
    # Some weird homebrewed metric for "what is a good resonator form"
    # I really like long necks, I like big bodies, I don't like thick necks.
    return 10 * neck_l + r_cav + neck_d / 2


def parameter_surface(
    frequency,
    r_cavity_range=np.linspace(0.01, 0.2, 20),  # Cavity radius range (in meters)
    d_range=np.linspace(0.05, 0.2, 20),  # Neck diameter range (in meters)
    title="Helmholtz Resonator Dimensions Grid Search",
    arbitrary_ranking_metric=good_results_metric,
):
    results = []
    solution_for_L, r_cavity, d = calculate_L(frequency)
    print(solution_for_L)

    # Perform grid search
    for r_cav in r_cavity_range:
        for diam in d_range:
            L_value = solution_for_L.subs({r_cavity: r_cav, d: diam}).evalf()
            results.append(
                (
                    (float(r_cav), float(diam), float(L_value)),
                    arbitrary_ranking_metric(r_cav, diam, L_value),
                )
            )

    r_cavity_vals, d_vals, L_vals = zip(*[r[0] for r in results])

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=r_cavity_vals,
                y=d_vals,
                z=L_vals,
                mode="markers",
                marker=dict(
                    size=5,
                    color=L_vals,  # set color to neck length
                    colorscale="Viridis",  # choose a colorscale
                    opacity=0.8,
                ),
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Cavity Radius (m)",
            yaxis_title="Neck Diameter (m)",
            zaxis=dict(
                title="Neck Length (m)", type="log"  # Set z-axis to logarithmic scale
            ),
        ),
    )
    fig.show()
    return sorted(results, key=lambda x: -x[1])


# %%
def calculate_f(
    length,
    diameter,
    radius,
    c_value=343,
    end_correction=lambda L, d: (L + 0.3 * d),
):
    """
    This is for a sanity check
    """
    f, c, L, d, r_cavity = sp.symbols("f c L d r_cavity")

    # Area of the neck (A) and Volume of the cavity (V)
    r_neck = d / 2
    A = sp.pi * r_neck**2  # Cross-sectional area of the neck
    V = (4 / 3) * sp.pi * r_cavity**3  # Volume of the cavity

    # Helmholtz resonator formula
    equation = sp.Eq(f, (c / (2 * sp.pi)) * sp.sqrt(A / (V * end_correction(L, d))))
    return (
        sp.solve(
            equation.subs(
                {c: c_value, L: length, r_cavity: radius, d: diameter, A: A, V: V}
            ),
            f,
        ),
    )


# %%
def parameter_sweep(target_frequency, note_name):
    parameter_sweep = parameter_surface(
        target_frequency,
        r_cavity_range=np.linspace(0.001, 0.03, 30),  # Cavity radius range (in meters)
        d_range=np.linspace(0.001, 0.02, 30),  # Neck diameter range (in meters)
        title="Helmholtz Resonator Dimensions Grid Search"
        + f"\nfor {note_name} ({round(target_frequency, 2)} Hz)",
    )
    top_parameter_set = parameter_sweep[0][0]

    r = top_parameter_set[0]
    d = top_parameter_set[1]
    L = top_parameter_set[2]

    assert round(calculate_f(length=L, diameter=d, radius=r)[0][0]) == round(
        target_frequency
    )
    return r, d, L


# %%
note_idx = 0

e8_r_cavity, e8_d, e8_L = parameter_sweep(
    chord_frequencies_defined[note_idx][1],
    chord_frequencies_defined[note_idx][0],
)

# %%
note_idx = 1

g8_r_cavity, g8_d, g8_L = parameter_sweep(
    chord_frequencies_defined[note_idx][1],
    chord_frequencies_defined[note_idx][0],
)

# %%
note_idx = 2

b8_r_cavity, b8_d, b8_L = parameter_sweep(
    chord_frequencies_defined[note_idx][1],
    chord_frequencies_defined[note_idx][0],
)

# %% [markdown]
# ## Let's generate the STL objects now!


# %%
def create_resonator_model(radius, neck_length, neck_radius, shell_thickness=1):
    sphere_diameter = radius * 2
    neck_diameter = neck_radius * 2

    sphere_model = sphere(sphere_diameter)
    neck_model = translate([0, 0, radius])(cylinder(h=neck_length, d=neck_diameter))
    resonator_model = sphere_model + neck_model

    # Hollow out the inside of the resonator...
    negative_sphere_model = sphere(sphere_diameter - (2 * shell_thickness))
    negative_neck_model = translate([0, 0, radius])(
        cylinder(
            h=neck_length + 2 * shell_thickness, d=neck_diameter - (2 * shell_thickness)
        )
    )
    negative_model = negative_sphere_model + negative_neck_model

    # TODO: Poke some holes in the bottom to allow sound to escape
    return resonator_model - negative_model


# %%
# create_resonator_model() expects everything in millimeters...
model_1 = create_resonator_model(b8_r_cavity * 1000, b8_L * 1000, b8_d * 1000 / 2)

# pdb.set_trace()

print("whats up")


def arrange_resonators(semisphere_radius=5):
    """
    It's going to be angled around a semisphere
    """
    # TODO: stop hardcoding this...
    b8_model = create_resonator_model(b8_r_cavity * 1000, b8_L * 1000, b8_d * 1000 / 2)
    e8_model = create_resonator_model(e8_r_cavity * 1000, e8_L * 1000, e8_d * 1000 / 2)
    g8_model = create_resonator_model(g8_r_cavity * 1000, g8_L * 1000, g8_d * 1000 / 2)

    # Rotate and translate b8_model
    b8_model_transformed = rotate([180, 0, 0])(b8_model)
    b8_height = b8_r_cavity * 1000 + b8_L * 1000
    b8_model_transformed = translate([0, 0, b8_height / 2])(b8_model_transformed)

    # Rotate and translate e8_model
    e8_model_transformed = rotate([135, 0, 0])(e8_model)
    e8_height = e8_r_cavity * 1000 + e8_L * 1000
    e8_model_transformed = translate([0, e8_height, e8_height / 2])(
        e8_model_transformed
    )

    # Rotate and translate a8_model
    g8_model_transformed = rotate([225, 0, 0])(g8_model)
    g8_height = g8_r_cavity * 1000 + g8_L * 1000
    g8_model_transformed = translate([0, -g8_height / 2, g8_height / 2])(
        g8_model_transformed
    )

    full_sphere = sphere(semisphere_radius)
    cutting_cube = cube(
        [2 * semisphere_radius, 2 * semisphere_radius, semisphere_radius], center=True
    )
    cutting_cube = translate([0, 0, -semisphere_radius])(cutting_cube)
    semi_sphere = difference()(full_sphere, cutting_cube)

    # Combine all models
    combined_model = (
        b8_model_transformed + e8_model_transformed + g8_model_transformed + semi_sphere
    )

    return combined_model


# Create individual models TODO:!!

# Arrange models
combined_model = arrange_resonators()
# Export the model
scad_render_to_file(combined_model, "combined_resonators.scad")