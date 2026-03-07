import gradio as gr
from models.inference import run_inference
from metadata_builder import build_metadata_csv


# ==========================================================
# CONSTANTS
# ==========================================================
GROUP_CHOICES = [
    ("Demographics", "demographics"),
    ("Clinical History", "history"),
    ("Symptoms", "symptoms"),
    ("Lesion Geometry", "lesion_geometry"),
]

DEFAULT_GROUPS = ["demographics", "symptoms", "lesion_geometry"]

REGION_CHOICES = ["HEAD", "NECK", "BACK", "ARM", "LEG", "TORSO"]
GENDER_CHOICES = ["MALE", "FEMALE"]


# ==========================================================
# HELPERS
# ==========================================================
def format_groups(enabled_groups):
    if not enabled_groups:
        return "No metadata group selected."

    label_map = {
        "demographics": "Demographics",
        "history": "Clinical History",
        "symptoms": "Symptoms",
        "lesion_geometry": "Lesion Geometry",
    }
    names = [label_map.get(g, g) for g in enabled_groups]
    return " | ".join(names)


def safe_bool(value):
    return bool(value)


def build_values_dict(
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
    return dict(
        age=age,
        gender=gender,
        region=region,
        diameter_1=diameter1,
        diameter_2=diameter2,
        itch=safe_bool(itch),
        grew=safe_bool(grew),
        hurt=safe_bool(hurt),
        changed=safe_bool(changed),
        bleed=safe_bool(bleed),
        elevation=safe_bool(elevation),
    )


def build_metadata_preview(
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
    values = build_values_dict(
        age, gender, region, diameter1, diameter2,
        itch, grew, hurt, changed, bleed, elevation
    )

    metadata_csv = build_metadata_csv(values, enabled_groups)
    groups_text = format_groups(enabled_groups)

    return metadata_csv, groups_text


def validate_inputs(image, enabled_groups, age, diameter1, diameter2):
    if image is None:
        raise gr.Error("Please upload a dermoscopic image first.")

    if not enabled_groups:
        raise gr.Error("Please select at least one metadata group.")

    if age is None or age < 0:
        raise gr.Error("Age must be a valid non-negative number.")

    if diameter1 is None or diameter1 < 0:
        raise gr.Error("Diameter 1 must be a valid non-negative number.")

    if diameter2 is None or diameter2 < 0:
        raise gr.Error("Diameter 2 must be a valid non-negative number.")


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
    validate_inputs(image, enabled_groups, age, diameter1, diameter2)

    values = build_values_dict(
        age, gender, region, diameter1, diameter2,
        itch, grew, hurt, changed, bleed, elevation
    )

    metadata_csv = build_metadata_csv(values, enabled_groups)
    heatmap_img, prediction_text = run_inference(image, metadata_csv)

    groups_text = format_groups(enabled_groups)

    pretty_prediction = (
        f"### Prediction Result\n\n"
        f"**Active groups:** {groups_text}\n\n"
        f"**Model output:**\n{prediction_text}"
    )

    return image, heatmap_img, pretty_prediction, metadata_csv, groups_text


def clear_all():
    return (
        None,                              # image_input
        DEFAULT_GROUPS,                    # group_selector
        55,                                # age
        "FEMALE",                          # gender
        "NECK",                            # region
        6,                                 # diameter1
        5,                                 # diameter2
        False,                             # itch
        False,                             # grew
        False,                             # hurt
        False,                             # changed
        False,                             # bleed
        False,                             # elevation
        None,                              # original_img_out
        None,                              # heatmap_out
        "### Prediction Result\n\nRun the model to see the output here.",
        "",                                # metadata_preview
        format_groups(DEFAULT_GROUPS),     # active_groups_text
    )


# ==========================================================
# UI
# ==========================================================
with gr.Blocks(title="Multimodal Skin Lesion Explainability", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# 🔬 Multimodal Skin Lesion Explainability
Interactive GradCAM++ visualization for multimodal diagnosis.

Upload a dermoscopic image, adjust the metadata, and generate the heatmap to observe how the model's attention changes.
"""
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=380):
            gr.Markdown("## Input Controls")

            image_input = gr.Image(
                type="pil",
                label="Dermoscopic Image",
                height=360
            )

            group_selector = gr.CheckboxGroup(
                choices=GROUP_CHOICES,
                value=DEFAULT_GROUPS,
                label="Active Metadata Groups"
            )

            active_groups_text = gr.Textbox(
                label="Selected Groups",
                value=format_groups(DEFAULT_GROUPS),
                interactive=False
            )

            with gr.Accordion("Demographics", open=True):
                age = gr.Number(label="Age", value=55, precision=0)
                gender = gr.Dropdown(
                    GENDER_CHOICES,
                    value="FEMALE",
                    label="Gender"
                )
                region = gr.Dropdown(
                    REGION_CHOICES,
                    value="NECK",
                    label="Region"
                )

            with gr.Accordion("Lesion Geometry", open=False):
                diameter1 = gr.Number(label="Diameter 1", value=6)
                diameter2 = gr.Number(label="Diameter 2", value=5)

            with gr.Accordion("Symptoms", open=False):
                with gr.Row():
                    itch = gr.Checkbox(label="Itch")
                    grew = gr.Checkbox(label="Grew")
                    hurt = gr.Checkbox(label="Hurt")

                with gr.Row():
                    changed = gr.Checkbox(label="Changed")
                    bleed = gr.Checkbox(label="Bleed")
                    elevation = gr.Checkbox(label="Elevation")

            with gr.Row():
                run_btn = gr.Button("Generate GradCAM++", variant="primary", size="lg")
                clear_btn = gr.Button("Clear", variant="secondary")

        with gr.Column(scale=2):
            gr.Markdown("## Visualization")

            with gr.Row():
                original_img_out = gr.Image(
                    label="Original Lesion Image",
                    height=360,
                    interactive=False
                )
                heatmap_out = gr.Image(
                    label="GradCAM++ Attention Map",
                    height=360,
                    interactive=False
                )

            prediction_out = gr.Markdown(
                "### Prediction Result\n\nRun the model to see the output here."
            )

            with gr.Accordion("Generated Metadata CSV", open=False):
                metadata_preview = gr.Textbox(
                    label="Metadata CSV Preview",
                    lines=8,
                    interactive=False
                )

    # ======================================================
    # EXAMPLES
    # ======================================================
    gr.Markdown("## Quick Interaction Tips")
    gr.Markdown(
        """
- Change one metadata field at a time to better observe how attention shifts.
- Keep the image fixed while modifying symptoms or lesion geometry.
- Use the metadata preview to confirm exactly what is being fed to the model.
"""
    )

    # ======================================================
    # LIVE METADATA PREVIEW
    # ======================================================
    preview_inputs = [
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
    ]

    for component in preview_inputs:
        component.change(
            fn=build_metadata_preview,
            inputs=preview_inputs,
            outputs=[metadata_preview, active_groups_text]
        )

    # ======================================================
    # RUN EVENT
    # ======================================================
    run_btn.click(
        fn=gradio_predict,
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
        outputs=[
            original_img_out,
            heatmap_out,
            prediction_out,
            metadata_preview,
            active_groups_text
        ]
    )

    # ======================================================
    # CLEAR EVENT
    # ======================================================
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[
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
            elevation,
            original_img_out,
            heatmap_out,
            prediction_out,
            metadata_preview,
            active_groups_text
        ]
    )


if __name__ == "__main__":
    demo.launch()