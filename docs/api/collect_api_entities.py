import importlib
import inspect
import pkgutil
from pathlib import Path

import nncf

modules = {}
skipped_modules = []
for importer, modname, ispkg in pkgutil.walk_packages(path=nncf.__path__,
                                                      prefix=nncf.__name__+'.',
                                                      onerror=lambda x: None):
    try:
        modules[modname] = importlib.import_module(modname)
    except:
        skipped_modules.append(modname)

api_fqns = []
for modname, module in modules.items():
    print(f"{modname}")
    for obj_name, obj in inspect.getmembers(module):
        objects_module = getattr(obj, '__module__', None)
        if objects_module == modname:
            if inspect.isclass(obj) or inspect.isfunction(obj):
                if hasattr(obj, "_nncf_api_marker"):
                    print(f"\t{obj_name}")
                    api_fqns.append(f"{modname}.{obj_name}")

print()
skipped_str = '\n'.join(skipped_modules)
print(f"Skipped: {skipped_str}\n")

print("API entities:")
for api_fqn in api_fqns:
    print(api_fqn)

DOC_ROOT = Path(__file__).parent
template_file = DOC_ROOT / 'source' / 'index_template.rst'
target_file = DOC_ROOT / 'source' / 'index.rst'

with open(template_file, encoding='utf-8', mode='r') as f:
    old_lines = f.readlines()
    for idx, line in enumerate(old_lines):
        anchor_line = idx
        if line == '.. API_ENTITIES_TEMPLATE_ANCHOR' + '\n':
            break
    api_section = ""
    for api_fqn in api_fqns:
        api_section += f"  {api_fqn}\n"
    content = ''.join(old_lines[:anchor_line]) + api_section + ''.join(old_lines[anchor_line + 1:])

with open(target_file, encoding='utf-8', mode='w+') as f:
    f.write(content)
