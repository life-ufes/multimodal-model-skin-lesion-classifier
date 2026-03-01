import gradio as gr
from models.inference import run_inference
from metadata_builder import build_metadata_csv


# ==========================================================
# Prediction wrapper
# ==========================================================
def gradio_predict(
    image,
    enabled_groups,
    age,
    gender,
    region,
    diameter1,
    diameter2,
    itch,
    grew,
    hurt,
    changed,
    bleed,
    elevation
):

    values = dict(
        age=age,
        gender=gender,
        region=region,
        diameter_1=diameter1,
        diameter_2=diameter2,
        itch=itch,
        grew=grew,
        hurt=hurt,
        changed=changed,
        bleed=bleed,
        elevation=elevation,
    )

    metadata_csv = build_metadata_csv(values, enabled_groups)
    return run_inference(image, metadata_csv)


# ==========================================================
# UI
# ==========================================================
with gr.Blocks(title="Multimodal Skin Lesion Explainability") as demo:

    gr.Markdown(
        "# 🔬 Multimodal Skin Lesion Explainability\n"
        "Interactive GradCAM++ for multimodal diagnosis"
    )

    # ======================================================
    # INPUT TAB
    # ======================================================
    with gr.Tab("Input"):

        with gr.Row():

            # ---------- IMAGE ----------
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Dermoscopic Image",
                    height=400
                )

            # ---------- METADATA ----------
            with gr.Column(scale=1):

                group_selector = gr.CheckboxGroup(
                    choices=[
                        ("Demographics", "demographics"),
                        ("Clinical History", "history"),
                        ("Symptoms", "symptoms"),
                        ("Lesion Geometry", "lesion_geometry"),
                    ],
                    value=["demographics", "symptoms", "lesion_geometry"],
                    label="Active Metadata Groups"
                )

                # -------- DEMOGRAPHICS --------
                with gr.Accordion("Demographics", open=True):

                    age = gr.Number(label="Age", value=55)

                    gender = gr.Dropdown(
                        ["MALE", "FEMALE"],
                        value="FEMALE",
                        label="Gender"
                    )

                    region = gr.Dropdown(
                        ["HEAD","NECK","BACK","ARM","LEG","TORSO"],
                        value="NECK",
                        label="Region"
                    )

                # -------- GEOMETRY --------
                with gr.Accordion("Lesion Geometry", open=False):

                    diameter1 = gr.Number(label="Diameter 1", value=6)
                    diameter2 = gr.Number(label="Diameter 2", value=5)

                # -------- SYMPTOMS --------
                with gr.Accordion("Symptoms", open=False):

                    itch = gr.Checkbox(label="Itch")
                    grew = gr.Checkbox(label="Grew")
                    hurt = gr.Checkbox(label="Hurt")
                    changed = gr.Checkbox(label="Changed")
                    bleed = gr.Checkbox(label="Bleed")
                    elevation = gr.Checkbox(label="Elevation")

        run_btn = gr.Button(
            "Generate GradCAM++",
            variant="primary",
            size="lg"
        )

    # ======================================================
    # RESULT TAB
    # ======================================================
    with gr.Tab("Results"):

        with gr.Row():
            output_img = gr.Image(
                label="GradCAM++ Attention Map",
                height=500
            )

        output_text = gr.Textbox(
            label="Prediction",
            interactive=False
        )

    # ======================================================
    # EVENT
    # ======================================================
    run_btn.click(
        gradio_predict,
        inputs=[
            image_input,
            group_selector,
            age,
            gender,
            region,
            diameter1,
            diameter2,
            itch,
            grew,
            hurt,
            changed,
            bleed,
            elevation
        ],
        outputs=[output_img, output_text]
    )


if __name__ == "__main__":
    demo.launch()