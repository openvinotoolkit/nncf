import importlib
import inspect
import pkgutil

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


