def clean_dictionary(input_dict: input) -> dict:
    to_return = dict()
    for k, v in input_dict.items():
        if v is not None:
            to_return[k] = v
    return to_return