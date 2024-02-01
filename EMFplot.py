import matplotlib.pyplot as plt
import subprocess
import tempfile
import os

class EMFplot:
    @staticmethod
    def savefig(filename, dpi=600, inkscape_path='C:\\Program Files\\Inkscape\\bin\\inkscape.exe'):
        # Use a temporary file to save the plot as SVG first
        temp_svg_file = tempfile.NamedTemporaryFile(delete=False, suffix='.svg')
        plt.savefig(temp_svg_file.name, format='svg', dpi=dpi)

        # Convert the SVG to EMF using Inkscape
        command = [inkscape_path, temp_svg_file.name, '--export-filename', filename]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)







