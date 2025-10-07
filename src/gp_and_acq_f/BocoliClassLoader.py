"""Discover classes with `bocoli_name` in custom GP / acquisition modules.

Provides:
- acquisition_function_class_dict: Dict[bocoli_name] = class
- gaussian_process_class_dict: Dict[bocoli_name] = class

"""

from __future__ import annotations

import inspect

from typing import Dict, Type, Optional, Any


import src.gp_and_acq_f.custom_gp_and_acq_f as custom_gp_and_acq_f

class BocoliClassLoader:
    """Loader exposing a small public API while sharing inspection helpers."""
    def __init__(self, source_module = custom_gp_and_acq_f):
        """allows passing a different module to inspect than the default one."""
        self.source_module = source_module
        self.all_classes = self._inspect_classes()
        self.load_bocoli_classes()
        


    def _inspect_classes(self) -> Dict[str, Dict[str, Any]]:
        """Return mapping bocoli_name -> info dict with keys: class, type, description.
        """
        all_classes = {}
        for _, cls in inspect.getmembers(self.source_module, inspect.isclass):
            # only consider classes exposing bocoli_name
            name = getattr(cls, "bocoli_name", None)
            if not isinstance(name, str):
                continue

            btype = getattr(cls, "bocoli_type", "default") or "default"
            desc = getattr(cls, "bocoli_description", None) or "no description"
            all_classes[name] = {"class": cls, "type": btype, "description": desc}
            

        return all_classes

    def load_bocoli_classes(self) -> Dict[str, Type]:
        """Return mapping bocoli_name -> class."""
        return {name: entry["class"] for name, entry in self.all_classes.items()}

    def group_by_type(self, mapping: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Group mapping (bocoli_name -> info) by the info['type'] value."""
        groups: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for name, entry in mapping.items():
            btype = entry.get("type", "default") or "default"
            groups.setdefault(btype, {})[name] = entry
        return groups

    @property
    def gaussian_process_class_dict(self) -> Dict[str, Type]:
        classes = self.load_bocoli_classes()
        info = {k: {"class": v, "type": getattr(v, "bocoli_type", "default") or "default", "description": getattr(v, "bocoli_description", None) or "no description"} for k, v in classes.items()}
        return self.group_by_type(info).get("gp", {})

    @property
    def acquisition_function_class_dict(self) -> Dict[str, Type]:
        classes = self.load_bocoli_classes()
        info = {k: {"class": v, "type": getattr(v, "bocoli_type", "default") or "default", "description": getattr(v, "bocoli_description", None) or "no description"} for k, v in classes.items()}
        return self.group_by_type(info).get("acqf", {})

    @property
    def gaussian_process_info(self) -> Dict[str, Dict[str, Any]]:
        return self.group_by_type(self.all_classes).get("gp", {})

    @property
    def acquisition_function_info(self) -> Dict[str, Dict[str, Any]]:

        return self.group_by_type(self.all_classes).get("acqf", {})


if __name__ == "__main__":
    # Quick manual test when running the file directly
    loader = BocoliClassLoader()
    print(loader.gaussian_process_class_dict)
    print(loader.acquisition_function_class_dict)

