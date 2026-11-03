
"""
DUWE DISASTER INTELLIGENCE
MODULE 10: HEC-RAS PROJECT INSPECTOR

Purpose:
1. Verify RAS Commander installation.
2. Locate HEC-RAS 7.0.1.
3. Inspect existing DUWE HEC-RAS projects.
4. Prepare for automated hydraulic modelling.
"""

from pathlib import Path
import importlib.metadata
import sys

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

DUWE = Path(
    r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"
)

HECRAS_EXE = Path(
    r"C:\Program Files (x86)\HEC\HEC-RAS\7.0.1\Ras.exe"
)

# --------------------------------------------------
# 2. DISPLAY SYSTEM INFORMATION
# --------------------------------------------------

print("=" * 65)
print("DUWE DISASTER INTELLIGENCE")
print("HEC-RAS 7.0.1 PROJECT INSPECTOR")
print("=" * 65)

print("\nPython version:")
print(sys.version)

# --------------------------------------------------
# 3. CHECK RAS COMMANDER
# --------------------------------------------------

print("\nCHECKING RAS COMMANDER")

try:
    version = importlib.metadata.version("ras-commander")

    print("RAS Commander installed.")
    print("Version:", version)

except importlib.metadata.PackageNotFoundError:

    print("RAS Commander installation not detected.")

# --------------------------------------------------
# 4. CHECK HEC-RAS
# --------------------------------------------------

print("\nCHECKING HEC-RAS")

if HECRAS_EXE.exists():

    print("HEC-RAS executable found.")
    print(HECRAS_EXE)

else:

    print("HEC-RAS executable not found.")

# --------------------------------------------------
# 5. CHECK DUWE PROJECT
# --------------------------------------------------

print("\nCHECKING DUWE PROJECT")

if not DUWE.exists():

    raise FileNotFoundError(
        f"DUWE project directory not found: {DUWE}"
    )

print("DUWE project found.")

# --------------------------------------------------
# 6. SEARCH FOR EXISTING HEC-RAS PROJECTS
# --------------------------------------------------

print("\nSEARCHING FOR HEC-RAS PROJECT FILES")

projects = list(DUWE.rglob("*.prj"))

if projects:

    for project in projects:

        print("\nProject:", project)
        print("Folder:", project.parent)

else:

    print("No HEC-RAS project files found in DUWE.")

# --------------------------------------------------
# 7. CHECK HYDRAULIC INPUT DATA
# --------------------------------------------------

print("\nCHECKING DUWE TERRAIN DATA")

datasets = [
    DUWE / "01_data" / "processed" / "dem_conditioned_duwe.tif",
    DUWE / "01_data" / "processed" / "catchment_duwe.tif"
]

for dataset in datasets:

    if dataset.exists():

        print("FOUND:", dataset.name)

    else:

        print("MISSING:", dataset.name)

# --------------------------------------------------
# 8. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 65)
print("INSPECTION COMPLETED")
print("=" * 65)
