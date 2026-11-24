import pytest

from perda.analyzer.plot_inputs import (
    axis_titles_per_subplot,
    group_variables_by_subplot,
    lookup_subplot_data_instances,
)


@pytest.mark.parametrize(
    "variables, expected",
    [
        pytest.param("a", ["a"], id="single"),
        pytest.param(["a", "b"], [["a", "b"]], id="flat_list_one_subplot"),
        pytest.param([["a", "b"], ["c"]], [["a", "b"], ["c"]], id="nested"),
        pytest.param([["a"], "c"], [["a"], "c"], id="mixed_nesting"),
        pytest.param([None, "x"], [None, "x"], id="none_marks_subplots"),
        pytest.param([[None], ["x"]], [[None], ["x"]], id="list_of_none_entry"),
    ],
)
def test_group_variables_by_subplot(variables, expected):
    assert group_variables_by_subplot(variables) == expected


@pytest.mark.parametrize(
    "subplot_variables, expected_names",
    [
        pytest.param(None, [], id="none"),
        pytest.param([None], [], id="list_of_none"),
        pytest.param("cpp.var_a", ["cpp.var_a"], id="single_name"),
        pytest.param(2, ["cpp.var_b"], id="single_id"),
        pytest.param("cpp.*", ["cpp.var_a", "cpp.var_b"], id="pattern"),
        pytest.param(
            ["cpp.var_b", None, ["cpp.var_a"]], ["cpp.var_b", "cpp.var_a"], id="nested"
        ),
    ],
)
def test_lookup_subplot_data_instances(srd_basic, subplot_variables, expected_names):
    result = lookup_subplot_data_instances(srd_basic, subplot_variables, None, None)
    assert [di.cpp_name for di in result] == expected_names


def test_lookup_subplot_data_instances_trims(srd_basic):
    result = lookup_subplot_data_instances(srd_basic, "cpp.*", 1, None)
    assert [list(di.timestamp_np) for di in result] == [[1, 2], [1, 2]]


@pytest.mark.parametrize(
    "titles, expected",
    [
        pytest.param(None, None, id="none"),
        pytest.param("Speed", ["Speed", "Speed"], id="broadcast_string"),
        pytest.param(("Speed", None), ["Speed", None], id="per_subplot"),
    ],
)
def test_axis_titles_per_subplot(titles, expected):
    assert axis_titles_per_subplot(titles, 2) == expected
