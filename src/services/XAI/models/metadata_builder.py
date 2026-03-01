from metadata_schema import PAD_SCHEMA
from metadata_groups import METADATA_GROUPS


def bool_to_str(v):
    if v is None:
        return ""
    return "True" if v else "False"


def build_metadata_csv(values_dict, enabled_groups):

    # inicia vazio
    data = {k: "" for k in PAD_SCHEMA}

    # campos sempre presentes
    data["patient_id"] = "PAT_DEMO"
    data["lesion_id"] = "000"
    data["img_id"] = "demo.png"

    # descobrir quais colunas estão habilitadas
    enabled_columns = set()

    for group in enabled_groups:
        enabled_columns.update(METADATA_GROUPS[group])

    # preencher apenas se grupo ativo
    for key, value in values_dict.items():

        if key not in enabled_columns:
            continue

        if isinstance(value, bool):
            data[key] = bool_to_str(value)
        else:
            data[key] = "" if value is None else str(value)

    return ",".join(str(data[col]) for col in PAD_SCHEMA)