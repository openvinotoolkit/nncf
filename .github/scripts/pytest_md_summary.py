# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import xml.etree.ElementTree as ET


def parse_xml_report(xml_file) -> None:
    """
    Parse the XML report generated by pytest.

    :param xml_file: Path to the XML report file
    :return: None
    """
    try:
        tree = ET.parse(xml_file)
    except FileNotFoundError:
        sys.exit(1)

    root = tree.getroot()

    # Build the summary table in Markdown format
    table_lines = []
    table_lines.append("| Test Name | Status | Time | Message |")
    table_lines.append("|:----------|:------:|-----:|:--------|")

    # Iterate over test cases
    for testcase in root.findall(".//testcase"):
        test_name = testcase.get("name")
        time_duration = float(testcase.get("time", "0"))
        message = ""
        if testcase.find("failure") is not None:
            status = "$${\color{red}Failed}$$"
            message = testcase.find("failure").get("message", "")
        elif testcase.find("error") is not None:
            status = "$${\color{red}Error}$$"
        elif testcase.find("skipped") is not None:
            status = "$${\color{orange}Skipped}$$"
            message = testcase.find("skipped").get("message", "")
        else:
            status = "$${\color{green}Ok}$$"

        # Append each row to the table
        if message:
            message = message.splitlines()[0][:60]
        table_lines.append(f"| {test_name} | {status} | {time_duration:.0f} | {message} |")

    if len(table_lines) > 2:
        # Print the summary table only if there are test cases
        print("\n".join(table_lines))


if __name__ == "__main__":
    """
    This script generates a summary table in Markdown format from an XML report generated by pytest.

    Usage in GitHub workflow:
        - name: Test Summary
        if: ${{ !cancelled() }}
        run: |
            python .github/scripts/generate_examples_summary.py pytest-results.xml >> $GITHUB_STEP_SUMMARY
    """
    try:
        parse_xml_report(sys.argv[1])
    except Exception as e:
        print(f"Error: {e}")