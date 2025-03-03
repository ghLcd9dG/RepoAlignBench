import os
import ast
import random
import warnings
import pandas as pd
from tree_sitter import Language, Parser
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")

# Language.build_library
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

LANG_map = {
    "python": ".py",
    "go": ".go",
    "java": ".java",
    "php": ".php",
    "ruby": ".rb",
    "javascript": ".js",
}

# Cache the languages
LANGUAGES = {
    "python": Language("build/my-languages.so", "python"),
    "go": Language("build/my-languages.so", "go"),
    "java": Language("build/my-languages.so", "java"),
    "php": Language("build/my-languages.so", "php"),
    "ruby": Language("build/my-languages.so", "ruby"),
    "javascript": Language("build/my-languages.so", "javascript"),
}


def extract_functions_and_classes_from_file(
    file_path, parser, language, ignore_class_methods=False
):
    """
    解析文件并提取其中的函数和类定义
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            src = file.read().encode("utf-8")

        tree = parser.parse(src)
        root_node = tree.root_node

        functions = []
        classes = []

        def extract_function_and_class(node, parent_is_class=False):
            if node.type == "function_definition":
                if parent_is_class and ignore_class_methods:
                    return  # 忽略类中的函数

                # 提取函数名称
                function_name_node = node.child_by_field_name("name")
                function_name = src[
                    function_name_node.start_byte : function_name_node.end_byte
                ].decode("utf8")

                # 提取函数体
                function_body_node = node.child_by_field_name("body")
                function_body = src[
                    function_body_node.start_byte : function_body_node.end_byte
                ].decode("utf8")

                functions.append({"name": function_name, "body": function_body})

            elif node.type == "class_definition":
                # 提取类名称
                class_name_node = node.child_by_field_name("name")
                class_name = src[
                    class_name_node.start_byte : class_name_node.end_byte
                ].decode("utf8")

                # 提取类体
                class_body_node = node.child_by_field_name("body")
                class_body = src[
                    class_body_node.start_byte : class_body_node.end_byte
                ].decode("utf8")

                classes.append({"name": class_name, "body": class_body})

                # 递归处理类内部的节点，设置 parent_is_class 为 True
                for child in node.children:
                    extract_function_and_class(child, parent_is_class=True)
            else:
                for child in node.children:
                    extract_function_and_class(child, parent_is_class=parent_is_class)

        # 从根节点开始遍历，提取函数和类
        extract_function_and_class(root_node)

        return functions, classes

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return [], []


def process_file(file_path, parser, LANG):
    """处理单个文件，提取函数和类定义"""
    try:
        functions, classes = extract_functions_and_classes_from_file(
            file_path, parser, LANG
        )
        return functions, classes
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return [], []


def process_multiple_files(file_paths, LANG):
    """
    批量处理多个文件，提取函数和类定义
    """
    parser = Parser()
    parser.set_language(LANG)

    all_functions = []
    all_classes = []

    # 使用并行化加速文件处理
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_file, file_path, parser, LANG): file_path
            for file_path in file_paths
        }
        for future in futures:
            try:
                functions, classes = future.result()
                all_functions.extend(functions)
                all_classes.extend(classes)
            except Exception as e:
                print(f"Error processing file in thread: {e}")

    return all_functions, all_classes


def extract_from_project(file_path, language="python"):
    # 获取对应语言的解析器
    LANG = LANGUAGES[language]

    # 获取文件路径
    file_paths = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith(LANG_map[language]):
                file_paths.append(os.path.join(root, file))

    print(f"Using language: {language}")
    print(f"Extracting functions and classes from {file_path}")

    all_functions, all_classes = process_multiple_files(file_paths, LANG)

    return all_functions, all_classes


if __name__ == "__main__":
    try:
        project_base_path = "dataset/ds_content"
        output_data = []
        output_columns = [
            "Language",
            "Query",
            "Code",
            "GitHubUrl",
            "Relevance",
            "Notes",
        ]

        ds = pd.read_csv("dataset/ds_label/own_dataset.csv")
        for index, row in ds.iterrows():
            func_label, cls_label = ast.literal_eval(row["label"])  # 解包标签
            if func_label == [] and cls_label == []:
                continue

            query = row["problem_statement"]
            save_path = os.path.join(project_base_path, row["save_path"])

            print(f"Processing {save_path}")
            # print(f"Function Label: {func_label}")
            # print(f"Class Label: {cls_label}")
            # print(f"Query: {query}")
            print(f"*" * 50)

            # 提取函数和类
            func_lst, cls_lst = extract_from_project(save_path)

            for _func in func_lst:
                url = save_path + "#" + _func["name"]
                if _func["name"] in func_label:
                    output_data.append(["Python", query, _func["body"], url, 1, ""])
                else:
                    if random.random() < 0.1:
                        output_data.append(["Python", query, _func["body"], url, 0, ""])

            for _class in cls_lst:
                url = save_path + "#" + _class["name"]
                if _class["name"] in cls_label:
                    output_data.append(["Python", query, _class["body"], url, 1, ""])
                else:
                    if random.random() < 0.1:
                        output_data.append(
                            ["Python", query, _class["body"], url, 0, ""]
                        )

        # 保存结果到 CSV 文件
        output_path = "dataset/ds_label/own_dataset_label_1.jsonl"
        output_df = pd.DataFrame(output_data, columns=output_columns)
        output_df.to_json(output_path, orient='records', lines=True)

    except Exception as e:
        print(f"An error occurred during processing: {e}")
