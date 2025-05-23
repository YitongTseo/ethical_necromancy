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
CHORD = "Gdim"


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
def create_resonator_model(
    radius,
    neck_length,
    neck_radius,
    shell_thickness=1,
    num_sound_holes=1,
    min_sound_hole_size=3,
):
    sphere_diameter = radius * 2
    neck_diameter = neck_radius * 2

    sphere_model = sphere(d=sphere_diameter + (2 * shell_thickness))
    little_gap = np.sqrt(radius**2 + (shell_thickness + neck_radius) ** 2)
    neck_model = translate([0, 0, radius - little_gap])(
        cylinder(h=neck_length + little_gap, d=neck_diameter + (2 * shell_thickness))
    )
    resonator_model = sphere_model + neck_model

    # Hollow out the inside of the resonator...
    negative_sphere_model = sphere(d=sphere_diameter)
    negative_neck_model = translate([0, 0, radius - little_gap])(
        cylinder(h=neck_length + little_gap + 2 * shell_thickness, d=neck_diameter)
    )
    resonator_model = resonator_model - negative_sphere_model
    resonator_model = resonator_model - negative_neck_model

    for idx in range(num_sound_holes):
        # TODO: add math here to allow for multiple sound holes...
        # arbitrarily the sound hole diameter is going to be 3 times smaller than diameter
        negative_soundhole = sphere(d=max(sphere_diameter / 3, min_sound_hole_size))
        negative_soundhole = translate([0, 0, -radius])(negative_soundhole)
        resonator_model = resonator_model - negative_soundhole

    return resonator_model


def arrange_resonators(
    resonator_models, parameters, semisphere_radius=18, hemisphere_thickness=1
):
    """
    Three helmholtz resonators are angled around a hollow semisphere
    """
    # Cut and outfit the hemisphere
    full_sphere = sphere(r=semisphere_radius)
    cutting_sphere = sphere(r=semisphere_radius - hemisphere_thickness)
    cutting_cube = cube(
        [2 * semisphere_radius, 2 * semisphere_radius, semisphere_radius], center=True
    )
    cutting_cube = translate([0, 0, -semisphere_radius])(cutting_cube)
    hemi_sphere = difference()(full_sphere - cutting_sphere, cutting_cube)

    # Arbitrarily the entrance will be 1/2 the dimension of the sphere...
    entrance = sphere(r=semisphere_radius / 2)
    entrance = translate([semisphere_radius, 0, 0])(entrance)
    hemi_sphere = hemi_sphere - entrance

    rotations = [
        [180, 0, 0],
        [135, 0, 0],
        [225, 0, 0],
    ]
    translations = [
        [0, 0, semisphere_radius],
        [0, semisphere_radius / np.sqrt(2), semisphere_radius / np.sqrt(2)],
        [0, -semisphere_radius / np.sqrt(2), semisphere_radius / np.sqrt(2)],
    ]

    sorted_models_and_params = sorted(
        zip(resonator_models, parameters), key=lambda x: x[1][0]
    )
    for resonator_idx, (resonator_model, params) in enumerate(sorted_models_and_params):
        resonator = rotate(rotations[resonator_idx])(resonator_model)
        resonator = translate(translations[resonator_idx])(resonator)
        hemi_sphere = hemi_sphere + resonator

        resonator_cutout = sphere(r=params[0])
        resonator_cutout = translate(translations[resonator_idx])(resonator_cutout)
        hemi_sphere = hemi_sphere - resonator_cutout

    return hemi_sphere


octave = 8
chord_frequencies_defined = chord_frequencies(CHORD, octave=octave)
print(f"{CHORD} in {octave} octave is: ", chord_frequencies_defined)

assert len(chord_frequencies_defined) == 3, "for now we can only handle 3 note chords"
resonator_models = []
parameters = []

for note_idx in range(len(chord_frequencies_defined)):
    r_cavity, d, L = parameter_sweep(
        chord_frequencies_defined[note_idx][1],
        chord_frequencies_defined[note_idx][0],
    )
    model = create_resonator_model(r_cavity * 1000, L * 1000, d * 1000 / 2)
    parameters.append((1000 * r_cavity, 1000 * d, 1000 * L))
    resonator_models.append(model)

combined_model = arrange_resonators(resonator_models, parameters)
# Export the model
scad_render_to_file(combined_model, "combined_resonators.scad")
