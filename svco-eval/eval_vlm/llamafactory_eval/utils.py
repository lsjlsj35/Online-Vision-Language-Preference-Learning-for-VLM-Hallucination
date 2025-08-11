def init_dataclass_from_dict(data_class, input_dict):
    fields = data_class.__dataclass_fields__.keys()
    filtered_dict = {key: input_dict[key] for key in fields if key in input_dict}
    return data_class(**filtered_dict)