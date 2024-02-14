import math
from solid import *
from solid import scad_render_to_file, sphere, translate, cylinder
from solid.utils import *
from stl import mesh
import numpy as np
import subprocess
import pdb
import mingus.core.chords as chords
from mingus.core import notes


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
    return [note_to_frequency(note, octave) for note in chord_notes]

# https://mixbutton.com/mixing-articles/music-note-to-frequency-chart/
# Quick test
assert round(note_to_frequency("C", 8), 2) == 4186.01
assert round(note_to_frequency("C", 4), 2) == 261.63
assert round(note_to_frequency("D", 1), 2) == 36.71

def convert_scad_to_stl(scad_file, stl_file):
    try:
        # Command to convert SCAD to STL using OpenSCAD
        cmd = ["openscad", "-o", stl_file, scad_file]

        # Run the command
        subprocess.run(cmd, check=True)
        print(f"Successfully converted {scad_file} to {stl_file}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while converting SCAD to STL: {e}")


# Function to calculate dimensions of a Helmholtz resonator


def calculate_resonator_dimensions(frequency):
    """
    f = (c / 2pi) sqrt(A / V(L + 0.3d))

    f is the resonant frequency (Hertz),
    c is the speed of sound in air (approximately 343 meters per second at room temperature) (m/s)
    A is the cross-sectional area of the neck (m^2)
    V is the volume of the cavity (m^3)
    L is the physical length of the neck (m)
    d is the diameter of the neck (m)

    Note: The 0.3d is an end correction to account for the radiation of sound from the open end of the neck.
    """
    # frequency / (343 / (2 * math.pi)) = math.sqrt(A / )

    # These are placeholder values. You'll need to replace these with the actual calculations.
    radius = 10  # in mm
    neck_length = 20  # in mm
    neck_radius = 5  # in mm
    shell_thickness = 0.5

    return radius, neck_length, neck_radius, shell_thickness





# Function to create a 3D model of the resonator
def create_resonator_model(radius, neck_length, neck_radius, shell_thickness):
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


# Function to save the model as an STL file
def save_stl(model, filename):
    model_mesh = mesh.Mesh(np.zeros(model.faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(model.faces):
        for j in range(3):
            model_mesh.vectors[i][j] = model.points[f[j], :]

    model_mesh.save(filename)


# Main execution
if __name__ == "__main__":
    target_frequency = 440  # A4 note, for example
    radius, neck_length, neck_radius, shell_thickness = calculate_resonator_dimensions(
        target_frequency
    )
    resonator_model = create_resonator_model(
        radius, neck_length, neck_radius, shell_thickness
    )
    scad_render_to_file(resonator_model, "resonator.scad")
    pdb.set_trace()
    save_stl(resonator_model, "resonator.stl")

    # convert_scad_to_stl("resonator.scad", "resonator.stl")

    # Convert SCAD to STL (you might need to do this step outside of Python using OpenSCAD software)
    # Alternatively, you can use a Python library that can directly convert OpenSCAD scripts to STL, if available.

    # Placeholder for STL conversion

    print("Model generated. Check the 'resonator.stl' file.")
