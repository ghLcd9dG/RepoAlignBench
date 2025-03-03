import os
import warnings

warnings.filterwarnings("ignore")
from tree_sitter import Language, Parser


Language.build_library(
    "build/my-languages.so",
    [
        "vendor/tree-sitter-python",
        "vendor/tree-sitter-go",
        "vendor/tree-sitter-java",
        "vendor/tree-sitter-php/php",
        "vendor/tree-sitter-ruby",
        "vendor/tree-sitter-javascript",
    ],
)


PYTHON_LANG = Language("build/my-languages.so", "python")

file_path = "src/code_snippet/graph_code_bert.py"
with open(file_path, "r", encoding="utf-8") as file:
    src = file.read().encode("utf-8")


parser = Parser()
parser.set_language(PYTHON_LANG)

tree = parser.parse(src)

root_node = tree.root_node


functions = []
classes = []


def extract_function_and_class(node, parent_is_class=False):
    if node.type == "function_definition":

        if parent_is_class:
            return

        function_name_node = node.child_by_field_name("name")
        function_name = src[
            function_name_node.start_byte : function_name_node.end_byte
        ].decode("utf8")

        function_body_node = node.child_by_field_name("body")
        function_body = src[
            function_body_node.start_byte : function_body_node.end_byte
        ].decode("utf8")

        functions.append({"name": function_name, "body": function_body})

    elif node.type == "class_definition":

        class_name_node = node.child_by_field_name("name")
        class_name = src[class_name_node.start_byte : class_name_node.end_byte].decode(
            "utf8"
        )

        class_body_node = node.child_by_field_name("body")
        class_body = src[class_body_node.start_byte : class_body_node.end_byte].decode(
            "utf8"
        )

        classes.append({"name": class_name, "body": class_body})

        for child in node.children:
            extract_function_and_class(child, parent_is_class=True)
    else:

        for child in node.children:
            extract_function_and_class(child, parent_is_class=parent_is_class)


extract_function_and_class(root_node)


print("Functions found:")
for function in functions:
    print(f"Function Name: {function['name']}")
    print(f"Function Body: {function['body']}\n")

print("Classes found:")
for cls in classes:
    print(f"Class Name: {cls['name']}")
    print(f"Class Body: {cls['body']}\n")
