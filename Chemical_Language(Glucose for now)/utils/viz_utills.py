# STOLEN FROM: https://practicalcheminformatics.blogspot.com/2019/11/interactive-plots-with-chemical.html

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
import base64
from io import BytesIO


def display_selected_data(selectedData):
    max_structs = 12
    structs_per_row = 6
    empty_plot = "data:image/gif;base64,R0lGODlhAQABAAAAACwAAAAAAQABAAA="
    if selectedData:
        if len(selectedData['points']) == 0:
            return empty_plot
        match_idx = [x['pointIndex'] for x in selectedData['points']]
        smiles_list = [Chem.MolFromSmiles(x) for x in list(df.iloc[match_idx].SMILES)]
        name_list = list(df.iloc[match_idx].Name)
        active_list = list(df.iloc[match_idx].is_active)
        name_list = [x + " " + str(y) for (x, y) in zip(name_list, active_list)]
        img = MolsToGridImage(smiles_list[0:max_structs], molsPerRow=structs_per_row, legends=name_list)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue())
        src_str = 'data:image/png;base64,{}'.format(encoded_image.decode())
    else:
        return empty_plot
    return src_str
