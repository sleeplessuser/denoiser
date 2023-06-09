import gradio as gr

from ml.pipeline import Pipeline

import warnings
warnings.simplefilter('ignore')

def run_app(pipeline: Pipeline, js, css, queue_size=5):
    with gr.Blocks(css=css) as app:          
        upload_input = gr.Audio(
            source="upload", label="Audio Upload", type="filepath"
        )
        denoised_uploaded_output = gr.Audio(label="Denoised")
        denoise_uploaded_button = gr.Button("Denoise")
        denoise_uploaded_button.click(
            pipeline.denoise,
            inputs=[upload_input],
            outputs=[denoised_uploaded_output],
        )
        app.load(_js=js)

    app.queue(queue_size)
    app.launch(server_name='0.0.0.0', server_port=7860)
