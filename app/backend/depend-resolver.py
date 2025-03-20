#!/usr/bin/env python3
"""
Dependency Resolver for AI Image Processing Project
This script attempts to find compatible versions of all required packages.
"""

import subprocess
import sys
import tempfile
import os
from itertools import product

# Base dependencies that must work together
CORE_DEPS = {
    "torch": ["2.0.0", "2.1.0"],
    "flask": ["2.2.0", "2.3.0"],
    "numpy": ["1.24.0", "1.24.3"],
    "Pillow": ["10.0.0"],
    "requests": ["2.31.0"],
}

# Hugging Face dependencies with version ranges to test
HF_DEPS = {
    "transformers": ["4.30.0", "4.31.0", "4.32.0", "4.33.0", "4.34.0"],
    "diffusers": ["0.23.0", "0.24.0", "0.25.0"],
    "accelerate": ["0.20.0", "0.21.0", "0.22.0", "0.23.0", "0.24.0", "0.25.0"],
    "tokenizers": ["0.13.0", "0.13.3", "0.14.0", "0.14.1"],
    "huggingface_hub": ["0.16.4", "0.17.0", "0.17.3"],
}

def test_compatibility(combination):
    """Test if a set of packages can be installed together"""
    deps = [f"{pkg}=={ver}" for pkg, ver in combination.items()]
    deps_str = " ".join(deps)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        venv_path = os.path.join(tmpdir, "venv")
        try:
            # Create a virtual environment
            subprocess.run(
                f"python -m venv {venv_path}", 
                shell=True, check=True, capture_output=True
            )
            
            # Try to install the dependencies
            result = subprocess.run(
                f"source {venv_path}/bin/activate && pip install {deps_str}",
                shell=True, capture_output=True
            )
            
            return result.returncode == 0, deps_str
        except Exception as e:
            return False, f"Error: {str(e)}"

def find_compatible_versions():
    """Find a compatible set of package versions"""
    # First, find compatible core dependencies
    core_combos = []
    for combo in product(*[[(k, v) for v in versions] for k, versions in CORE_DEPS.items()]):
        core_dict = dict(combo)
        success, deps_str = test_compatibility(core_dict)
        if success:
            print(f"✅ Compatible core deps: {deps_str}")
            core_combos.append(core_dict)
        else:
            print(f"❌ Incompatible core deps: {deps_str}")
    
    if not core_combos:
        print("Could not find compatible core dependencies!")
        return None
    
    # Use the first compatible core combo to test with HF deps
    core = core_combos[0]
    
    # Test HF dependencies one by one, finding compatible versions for each
    compatible_hf = {}
    for pkg, versions in HF_DEPS.items():
        for version in versions:
            test_combo = {**core, pkg: version}
            success, deps_str = test_compatibility(test_combo)
            if success:
                compatible_hf[pkg] = version
                print(f"✅ Compatible {pkg}=={version}")
                break
            else:
                print(f"❌ Incompatible {pkg}=={version}")
    
    # Final test with all packages together
    final_combo = {**core, **compatible_hf}
    success, deps_str = test_compatibility(final_combo)
    if success:
        print(f"\n✅ COMPATIBLE COMBINATION FOUND:\n{deps_str}")
        return final_combo
    else:
        print(f"\n❌ Final combination incompatible:\n{deps_str}")
        return None

if __name__ == "__main__":
    print("Testing package compatibility...")
    result = find_compatible_versions()
    
    if result:
        # Generate requirements.txt
        with open("requirements.txt", "w") as f:
            for pkg, ver in result.items():
                f.write(f"{pkg}=={ver}\n")
        print("\nGenerated requirements.txt with compatible versions")
    else:
        print("\nCould not find a compatible set of versions")
        sys.exit(1)