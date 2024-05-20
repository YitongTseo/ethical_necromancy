# STOLEN FROM: https://practicalcheminformatics.blogspot.com/2019/11/interactive-plots-with-chemical.html
# And edited with ChatGPT

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute().parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute().parent))

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import MolsToGridImage
import base64
from io import BytesIO
import pandas as pd
import numpy as np
import umap
from sklearn.decomposition import PCA
import nbformat
import plotly.express as px
import umap

def create_interactive_plot(df, smiles_col, x_col, y_col, hover_cols, words_col=None, cluster_col=None):
    # Initialize the Dash app
    app = dash.Dash(__name__)

    # Define marker colors
    if cluster_col and cluster_col in df.columns:
        unique_clusters = df[cluster_col].unique()
        color_map = {cluster: i for i, cluster in enumerate(unique_clusters)}
        df["color"] = df[cluster_col].map(color_map)
        colors = df["color"]
    else:
        colors = "blue"

    # Generate the scatter plot component
    graph_component = dcc.Graph(
        id="scatter-plot",
        config={"displayModeBar": True, "scrollZoom": True},  # Enable zooming
        figure={
            "data": [
                go.Scattergl(
                    x=df[x_col],
                    y=df[y_col],
                    mode="markers",
                    opacity=0.7,
                    marker={
                        "size": 5,
                        "color": colors,
                        "colorscale": "Viridis",
                        "line": {"width": 0.5, "color": "white"},
                    },
                    customdata=df[hover_cols].values,
                    hoverinfo="text",
                    hovertext=df.apply(
                        lambda row: "<br>".join(
                            [f"{col}: {row[col]}" for col in hover_cols]
                        ),
                        axis=1,
                    ),
                )
            ],
            "layout": go.Layout(
                height=600,
                xaxis={"title": x_col},
                yaxis={"title": y_col},
                margin={"l": 40, "b": 40, "t": 10, "r": 10},
                hovermode="closest",
                dragmode="select",
            ),
        },
    )

    # Image component to display selected molecules
    image_component = html.Img(id="structure-image")

    # Layout of the Dash app
    app.layout = html.Div([html.Div([graph_component]), html.Div([image_component])])

    # Callback to update the image based on selected points
    @app.callback(
        Output("structure-image", "src"), [Input("scatter-plot", "selectedData")]
    )
    def display_selected_data(selectedData):
        max_structs = 12
        structs_per_row = 6
        empty_plot = "data:image/gif;base64,R0lGODlhAQABAAAAACwAAAAAAQABAAA="
        if selectedData:
            if len(selectedData["points"]) == 0:
                return empty_plot
            match_idx = [point["pointIndex"] for point in selectedData["points"]]
            smiles_list = [
                Chem.MolFromSmiles(smile) for smile in df.iloc[match_idx][smiles_col]
            ]
            if words_col is not None:
                words_list = df.iloc[match_idx][words_col].tolist()
            else:
                words_list = smiles_list
            img = MolsToGridImage(
                smiles_list[0:max_structs], molsPerRow=structs_per_row, legends=words_list[0:max_structs], returnPNG=False
            )
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue())
            src_str = "data:image/png;base64,{}".format(encoded_image.decode())
        else:
            return empty_plot
        return src_str

    # Run the app
    app.run_server(debug=True)



# def create_interactive_plot(df, smiles_col, x_col, y_col, hover_cols, cluster_col=None):
#     # Initialize the Dash app
#     app = dash.Dash(__name__)

#     # Define marker colors
#     if cluster_col and cluster_col in df.columns:
#         unique_clusters = df[cluster_col].unique()
#         color_map = {cluster: i for i, cluster in enumerate(unique_clusters)}
#         df["color"] = df[cluster_col].map(color_map)
#         colors = df["color"]
#     else:
#         colors = "blue"

#     # Generate the scatter plot component
#     graph_component = dcc.Graph(
#         id="scatter-plot",
#         config={"displayModeBar": True, "scrollZoom": True},  # Enable zooming
#         figure={
#             "data": [
#                 go.Scattergl(
#                     x=df[x_col],
#                     y=df[y_col],
#                     mode="markers",
#                     opacity=0.7,
#                     marker={
#                         "size": 5,
#                         "color": colors,
#                         "colorscale": "Viridis",
#                         "line": {"width": 0.5, "color": "white"},
#                     },
#                     customdata=df[hover_cols].values,
#                     hoverinfo="text",
#                     hovertext=df.apply(
#                         lambda row: "<br>".join(
#                             [f"{col}: {row[col]}" for col in hover_cols]
#                         ),
#                         axis=1,
#                     ),
#                 )
#             ],
#             "layout": go.Layout(
#                 height=600,
#                 xaxis={"title": x_col},
#                 yaxis={"title": y_col},
#                 margin={"l": 40, "b": 40, "t": 10, "r": 10},
#                 hovermode="closest",
#                 dragmode="select",
#             ),
#         },
#     )

#     # Image component to display selected molecules
#     image_component = html.Img(id="structure-image")

#     # Layout of the Dash app
#     app.layout = html.Div([html.Div([graph_component]), html.Div([image_component])])

#     # Callback to update the image based on selected points
#     @app.callback(
#         Output("structure-image", "src"), [Input("scatter-plot", "selectedData")]
#     )
#     def display_selected_data(selectedData):
#         max_structs = 12
#         structs_per_row = 6
#         empty_plot = "data:image/gif;base64,R0lGODlhAQABAAAAACwAAAAAAQABAAA="
#         if selectedData:
#             if len(selectedData["points"]) == 0:
#                 return empty_plot
#             match_idx = [point["pointIndex"] for point in selectedData["points"]]
#             smiles_list = [
#                 Chem.MolFromSmiles(smile) for smile in df.iloc[match_idx][smiles_col]
#             ]
#             img = MolsToGridImage(
#                 smiles_list[0:max_structs], molsPerRow=structs_per_row, returnPNG=False
#             )
#             buffered = BytesIO()
#             img.save(buffered, format="PNG")
#             encoded_image = base64.b64encode(buffered.getvalue())
#             src_str = "data:image/png;base64,{}".format(encoded_image.decode())
#         else:
#             return empty_plot
#         return src_str

#     # Run the app
#     app.run_server(debug=True)


def visualize_chem_embeddings(
    df, hover_cols=["ZINC_ID", "SMILES"], cluster_col=None, words_col=None, use_pca=False
):
    fingerprints = np.array(df["Morgan_fingerprint"].tolist())
    pca = PCA(n_components=2)
    if use_pca:
        embedding = pca.fit_transform(fingerprints)
    else:
        pca = PCA(n_components=300)  # Reduce to 100 dimensions
        pcaed = pca.fit_transform(fingerprints)
        reducer = umap.UMAP(
            n_neighbors=5, min_dist=0.1, n_components=2, random_state=42
        )
        embedding = reducer.fit_transform(pcaed)

    df["PCA1"] = embedding[:, 0]
    df["PCA2"] = embedding[:, 1]
    create_interactive_plot(
        df,
        smiles_col="SMILES",
        x_col="PCA1",
        y_col="PCA2",
        hover_cols=hover_cols + ([cluster_col] if cluster_col is not None else []),
        cluster_col=cluster_col,
        words_col=words_col
    )



def visualize_word_embeddings(df, use_pca=False):
    embeddings = np.array(df["BERT_Embedding"].tolist())

    if use_pca:
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(embeddings)
    else:
        pca = PCA(n_components=300)  # Reduce to 100 dimensions
        pcaed = pca.fit_transform(embeddings)
        reducer = umap.UMAP(
            n_neighbors=5, min_dist=0.1, n_components=2, random_state=42
        )
        pca_results = reducer.fit_transform(pcaed)

    df["PCA1"] = pca_results[:, 0]
    df["PCA2"] = pca_results[:, 1]
    fig = px.scatter(
        df,
        x="PCA1",
        y="PCA2",
        title="PCA projection of BERT Embeddings",
        labels={"PCA1": "PCA1", "PCA2": "PCA2"},
        hover_name="Word",
        hover_data=["Cluster"],
        color_continuous_scale="plasma",
        color="Cluster",
        render_mode="webgl",
    )
    fig.show()


# # Example usage
# if __name__ == '__main__':
#     # Assuming enamine_fingerprints_df is your DataFrame
#     df = enamine_fingerprints_df

#     # Define the columns
#     smiles_col = 'SMILES'
#     x_col = 'UMAP1'
#     y_col = 'UMAP2'
#     hover_cols = ['ZINC_ID', 'SMILES']

#     # Run the interactive plot
#     create_interactive_plot(df, smiles_col, x_col, y_col, hover_cols)
