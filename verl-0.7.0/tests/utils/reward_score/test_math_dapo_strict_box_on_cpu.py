# Copyright 2025 Individual Contributor: NTHR example contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl.utils.reward_score import default_compute_score


def test_default_math_dapo_score_forwards_strict_box_verify():
    result = default_compute_score(
        data_source="math_dapo",
        solution_str=r"The final answer is \boxed{34}.",
        ground_truth="34",
        strict_box_verify=True,
    )

    assert result["score"] == 1.0
    assert result["acc"] is True
    assert result["pred"] == "34"


def test_default_math_dapo_strict_box_rejects_unboxed_answer():
    result = default_compute_score(
        data_source="math_dapo",
        solution_str="Answer: 34",
        ground_truth="34",
        strict_box_verify=True,
    )

    assert result["score"] == -1.0
    assert result["acc"] is False
