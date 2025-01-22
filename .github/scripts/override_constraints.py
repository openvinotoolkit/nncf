# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import sys
from pathlib import Path

CONSTRAINTS_FILE = "constraints.txt"


def main():
    arg = sys.argv[1]
    overrided_requirements = [r.strip() for r in arg.split(",")]

    print("overrided_requirements: ", arg)

    file = Path(CONSTRAINTS_FILE)
    content = file.read_text()

    for new_requirement in overrided_requirements:
        new_requirement = new_requirement.strip()
        package_name = new_requirement.split("==")[0]
        content = re.sub(f"^{package_name}\s*[=><].*", "", content, flags=re.MULTILINE)
        content += f"\n{new_requirement}"

    print("New constraints:")
    print(content)

    file.write_text(content)


if __name__ == "__main__":
    main()
