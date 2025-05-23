import os
import glob

from PyQt6.QtWidgets import QGroupBox, QHBoxLayout, QCheckBox, QComboBox


def load_scenarios():
    scenario_dropdown = QComboBox()
    scenario_files = glob.glob("data_scenarios/*.yaml")
    scenario_names = [os.path.splitext(os.path.basename(f))[0] for f in scenario_files]
    scenario_names.sort(reverse=True)
    scenario_dropdown.addItems(scenario_names)
    return scenario_dropdown


def load_scenario_checkboxes():
    scenario_group = QGroupBox("Select Scenarios:")
    layout = QHBoxLayout()

    scenario_files = glob.glob("data_scenarios/*.yaml")
    scenario_names = [os.path.splitext(os.path.basename(f))[0] for f in scenario_files]
    scenario_names.sort(reverse=True)
    scenario_names.insert(0, "training")

    checkboxes = {}

    for name in scenario_names:
        checkbox = QCheckBox(name)
        layout.addWidget(checkbox)
        checkboxes[name] = checkbox

    checkboxes["training"].setChecked(True)

    scenario_group.setLayout(layout)
    return scenario_group, checkboxes
