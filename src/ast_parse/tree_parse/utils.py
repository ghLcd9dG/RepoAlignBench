def get_function_name(function_node):
    name_node = function_node.child_by_field_name("name")
    return name_node.text.decode("utf8") if name_node else "Unnamed"


def get_class_name(class_node):
    name_node = class_node.child_by_field_name("name")
    return name_node.text.decode("utf8") if name_node else "Unnamed"


def get_node_range(node):
    start_pos = node.start_point
    end_pos = node.end_point
    return {"start": start_pos, "end": end_pos}


def get_node_code(source_code, node):
    start_byte = node.start_byte
    end_byte = node.end_byte
    return source_code[start_byte:end_byte]
