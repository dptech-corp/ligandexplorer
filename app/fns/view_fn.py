import gradio as gr
import random
import os
import string

def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choices(characters, k=length))
    return random_string

def show_protein(input_file_protein):
    with open(input_file_protein, "r") as f:
        protein = f.read()
    keyword = generate_random_string(5)
    html = f"""<!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, user-scalable=no, minimum-scale=1.0, maximum-scale=1.0">
            <title>Mol* Viewer - Protein</title>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                #uni-view-iframe-protein-{keyword} {{
                    width: 100%;
                    height: 500px;
                }}
            </style>
        </head>
        <body>
            <iframe id="uni-view-iframe-protein-{keyword}" src="https://cdn.dp.tech/hermite/web/static/uni-view/uni-view-20250402.html" onload="document.getElementById('uni-view-iframe-protein-{keyword}').contentWindow.postMessage({{ key: ['protein'], protein: `{protein}`}}, 'https://cdn.dp.tech')"></iframe>
        </body>
    </html>"""
    return gr.HTML(html, visible=True)

def show_ligand(input_file_ligand):
    file_extension = os.path.splitext(input_file_ligand)[1]
    pure_extension = file_extension[1:]
    with open(input_file_ligand, "r") as f:
        ligand = f.read()
    keyword = generate_random_string(5)
    html = f"""<!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, user-scalable=no, minimum-scale=1.0, maximum-scale=1.0">
            <title>Mol* Viewer - Ligand</title>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                #uni-view-iframe-ligand-{keyword} {{
                    width: 100%;
                    height: 500px;
                }}
            </style>
        </head>
        <body>
            <iframe id="uni-view-iframe-ligand-{keyword}" src="https://cdn.dp.tech/hermite/web/static/uni-view/uni-view-20250402.html" onload="document.getElementById('uni-view-iframe-ligand-{keyword}').contentWindow.postMessage({{ key: ['ligand'], ligand: `{ligand}`, ligandFormat: `{pure_extension}`}}, 'https://cdn.dp.tech')"></iframe>
        </body>
    </html>"""
    return gr.HTML(html, visible=True)
