import gradio as gr
from apps import ligand_explorer_app
import argparse

def main():

    parser = argparse.ArgumentParser(description="Launch the Ligand Explorer Gradio APP.")
    parser.add_argument("--port", "-p", type=int, default=23003, help="Port number to run the Gradio app on.")
    args = parser.parse_args()

    with gr.Blocks() as app:
        ligand_explorer_app()
    
    app.launch(
        server_port=args.port,
        server_name="0.0.0.0"
    )
    
if __name__ == "__main__":
    main()
