# Copyright Quantinuum
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

import numpy as np
import pytest

from pytket import Circuit, Qubit
from pytket.extensions.qujax import tk_to_qujax


@pytest.mark.parametrize("layout", ["registers", "sparse", "multidimensional"])
@pytest.mark.parametrize("simulator", ["statetensor", "densitytensor", "unitarytensor"])
def test_custom_qubit_identifiers(layout: str, simulator: str) -> None:
    layouts = {
        "registers": [Qubit("b", 0), Qubit("a", 0)],
        "sparse": [Qubit(7), Qubit(3)],
        "multidimensional": [Qubit("grid", [0, 1]), Qubit("grid", [0, 0])],
    }
    first, second = layouts[layout]
    circuit = Circuit()
    circuit.add_qubit(first)
    circuit.add_qubit(second)
    circuit.H(first).CX(first, second).X(second)

    state = circuit.get_statevector()
    expected: np.ndarray
    if simulator == "statetensor":
        expected = state
    elif simulator == "densitytensor":
        expected = np.outer(state, state.conj())
    else:
        expected = circuit.get_unitary()
    result = np.asarray(tk_to_qujax(circuit, simulator=simulator)())
    np.testing.assert_allclose(
        result.reshape(expected.shape), expected, rtol=1e-6, atol=1e-7
    )
