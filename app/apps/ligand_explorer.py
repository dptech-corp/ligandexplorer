import gradio as gr
import os
import subprocess
import zipfile
from pathlib import Path
from fns.view_fn import show_protein, show_ligand
from fns.app_utils import create_dir


def ligand_explorer_app():
    
    def process_upload(complex_file):
        if complex_file is None:
            return "<div>Error: File not found</div>", gr.update(interactive=False)
        return show_protein(complex_file), gr.update(interactive=True)

    def submit_task(complex_file):
        if complex_file is None:
            return "no file", None, None, None, None

        complex_name = Path(complex_file).stem

        source_dir = Path(create_dir(f"data/{complex_name}"))
        input_dir = source_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir = source_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        complex_zip_path = input_dir / "input.zip"
        with zipfile.ZipFile(complex_zip_path, 'w') as zipf:
            zipf.write(complex_file, os.path.basename(complex_file))
        
        # cmd = f"ligandexplorer -i {complex_zip_path} -o {output_dir}"
        # os.system(cmd)

        process = subprocess.Popen(
            ["ligandexplorer", "-i", str(complex_zip_path), "-o", str(output_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        stdout_output, _ = process.communicate()

        target_dir = output_dir / complex_name
        temps = []
        if target_dir.exists():
            for _, _, temps in os.walk(target_dir):
                break

        if not temps:
            return (
                gr.update(value=stdout_output or "No output", visible=True),
                gr.update(choices=[], visible=False),
                [],
                gr.update(visible=False),
                gr.update(visible=False),
            )

        result_zip_path = output_dir / f"{complex_name}_ligandexplorer_result.zip"
        with zipfile.ZipFile(result_zip_path, 'w') as zipf:
            for temp in temps:
                zipf.write(output_dir / complex_name / temp, temp)

        names = [name for name in temps if name.endswith(".pdb")]
        names = [name for name in names if name != Path(complex_file).name]
        files = [str(output_dir / complex_name / f) for f in names]

        if not names:
            return (
                gr.update(value=stdout_output or "No ligands found", visible=True),
                gr.update(choices=[], visible=False),
                [],
                gr.update(value=str(result_zip_path), visible=True),
                gr.update(visible=False),
            )

        return (
            gr.update(value=stdout_output, visible=True),
            gr.update(choices=names, visible=True, value=names[0]),
            files, 
            gr.update(value=str(result_zip_path), visible=True),
            gr.update(value=select_structure(names[0], files), visible=True),
        )


    def select_structure(selected_name, files):
        for file in files:
            file = Path(file)
            if file.name == selected_name:
                if file.name == "protein.pdb":
                    structure_display = show_protein(file)
                else:
                    structure_display = show_ligand(file)
                return structure_display
        return "<div>Error: File not found</div>"


    with gr.Blocks() as block:

        with gr.Row():
            gr.Markdown("# Ligand Explorer")

        with gr.Row():
            with gr.Accordion("Introduction", open=True):
                gr.Markdown("**LigandExplorer** is a workflow that combines *cheminformatics tools* and *machine learning methods* to automatically extract and classify ligands from PDB structures. It applies graph theory to identify covalent and non-covalent ligands based on molecular connectivity and uses machine learning models to filter out irrelevant molecules, ensuring that only biologically significant ligands are retained.")
                gr.Markdown("**Notice:** The current interface is a trial version. Please do not upload any content containing sensitive information. For more information, please contact chenx@dp.tech.")

        with gr.Row():
            
            with gr.Column():
                gr.Markdown("### Input File:")
                complex_file = gr.File(label="Upload complex file in .pdb format", file_types=[".pdb"])

            with gr.Column():
                gr.Markdown("### Output File:")
                download_zip_file = gr.File(label="Download results (Please copy the link address below and paste it on a new tab to download the zip file.)", interactive=False, visible=False)

        with gr.Row():

            with gr.Column():
                complex_display = gr.HTML(label="Input Display", visible=True)
                submit_btn = gr.Button("Run LigandExplorer", interactive=False)

            with gr.Column():
                structure_display = gr.HTML(label="Structure Display", visible=False)
                structures_dropdown = gr.Dropdown(label="Select structure to display", choices=[], interactive=True, visible=False)

        with gr.Row():
            log_display = gr.Textbox(label="Log Display", value="", visible=False, lines=15)

        complex_file.upload(fn=process_upload, inputs=[complex_file], outputs=[complex_display, submit_btn]).then(
            lambda: gr.update(value="Run LigandExplorer", interactive=True),
            inputs=None,
            outputs=[submit_btn]
        )

        files_state = gr.State([])

        submit_btn.click(
            lambda: (gr.update(value="⏳ Running... Please wait.", visible=True, interactive=False)),
            inputs=None,
            outputs=[submit_btn]
        ).then(
            fn=submit_task, 
            inputs=complex_file, 
            outputs=[
                log_display,
                structures_dropdown, 
                files_state, 
                download_zip_file, 
                structure_display, 
            ]).then(
            lambda: (gr.update(value="✅ Task completed.", visible=True, interactive=False)),
            inputs=None,
            outputs=[submit_btn]
        )

        structures_dropdown.change(fn=select_structure, inputs=[structures_dropdown, files_state], outputs=[structure_display])

    return block

if __name__ == "__main__":
    ligand_explorer_app(None).launch(server_port=9999)