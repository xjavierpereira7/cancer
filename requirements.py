# guardar como generar_requirements.py
import os
import ast

libs = set()

for root, _, files in os.walk("."):
    for file in files:
        if file.endswith(".py"):
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                try:
                    tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for n in node.names:
                                libs.add(n.name.split(".")[0])
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                libs.add(node.module.split(".")[0])
                except:
                    pass

with open("requirements.txt", "w") as out:
    for lib in sorted(libs):
        if lib not in ("__future__", "typing"):  # librerías estándar comunes
            out.write(lib + "\n")

print("requirements.txt generado.")
