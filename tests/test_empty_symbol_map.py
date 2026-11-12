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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sympy import Symbol

from pytket import Circuit
from pytket.circuit import Unitary1qBox
from pytket.extensions.qujax import tk_to_qujax


def _expected(circuit: Circuit, simulator: str) -> np.ndarray:
    state = circuit.get_statevector()
    if simulator == "statetensor":
        return state
    if simulator == "densitytensor":
        return np.outer(state, state.conj())
    return circuit.get_unitary()


@pytest.mark.parametrize("boxed", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("simulator", ["statetensor", "densitytensor", "unitarytensor"])
def test_empty_symbol_map_embeds_fixed_gates(
    boxed: bool, compiled: bool, simulator: str
) -> None:
    circuit = Circuit(1).H(0)
    if boxed:
        box = Unitary1qBox(Circuit(1).Rx(0.3, 0).get_unitary())
        circuit.add_unitary1qbox(box, 0)
    else:
        circuit.Rx(0.3, 0)
    circuit.Rz(0.7, 0)
    simulate = tk_to_qujax(circuit, {}, simulator=simulator)
    if compiled:
        simulate = jax.jit(simulate)
    expected = _expected(circuit, simulator)
    actual = np.asarray(simulate())
    np.testing.assert_allclose(
        actual.reshape(expected.shape), expected, rtol=1e-6, atol=1e-7
    )


@pytest.mark.parametrize("simulator", ["statetensor", "densitytensor", "unitarytensor"])
def test_empty_symbol_map_rejects_unmapped_symbols(simulator: str) -> None:
    circuit = Circuit(1).Rx(Symbol("theta"), 0)
    with pytest.raises(AssertionError, match="Circuit keys"):
        tk_to_qujax(circuit, {}, simulator=simulator)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("simulator", ["statetensor", "densitytensor", "unitarytensor"])
def test_none_symbol_map_retains_runtime_parameters(
    compiled: bool, simulator: str
) -> None:
    simulate = tk_to_qujax(Circuit(1).Rx(0.3, 0), None, simulator=simulator)
    if compiled:
        simulate = jax.jit(simulate)
    expected = _expected(Circuit(1).Rx(0.8, 0), simulator)
    actual = np.asarray(simulate(jnp.array([0.8])))
    np.testing.assert_allclose(
        actual.reshape(expected.shape), expected, rtol=1e-6, atol=1e-7
    )
