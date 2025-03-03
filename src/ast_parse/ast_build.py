import warnings
warnings.filterwarnings("ignore")
from tree_sitter import Language, Parser

Language.build_library(
    # Store the library in the `build` directory
    "build/my-languages.so",
    # Include one or more languages
    [
        "vendor/tree-sitter-python",
        "vendor/tree-sitter-go",
        "vendor/tree-sitter-java",
        "vendor/tree-sitter-php/php",
        "vendor/tree-sitter-ruby",
        "vendor/tree-sitter-javascript",
    ],
)
