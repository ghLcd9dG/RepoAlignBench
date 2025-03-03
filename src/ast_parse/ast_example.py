from tree_sitter import Language, Parser

###############################################
PY_LANGUAGE = Language('build/my-languages.so', 'python')

py_parser = Parser()
py_parser.set_language(PY_LANGUAGE)

py_code_snippet = '''
def foo():
    print("Hello, World!")
    return 42
'''

tree = py_parser.parse(bytes(py_code_snippet, "utf8"))
root_node = tree.root_node
print(f"python code snippet: {py_code_snippet}")

###############################################
PHP_LANGUAGE = Language('build/my-languages.so', 'php')


php_parser = Parser()
php_parser.set_language(PHP_LANGUAGE)

php_code_snippet = '''
<?php
echo "Hello, World!";
?>
'''

tree = php_parser.parse(bytes(php_code_snippet, "utf8"))
root_node = tree.root_node
print(f"php code snippet: {php_code_snippet}")


