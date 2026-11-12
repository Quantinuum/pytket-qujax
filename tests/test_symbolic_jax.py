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
from sympy import Symbol, cos, exp, sin, sqrt

from pytket import Circuit
from pytket.extensions.qujax import tk_to_qujax


@pytest.mark.parametrize("expression", ["sin", "cos", "exp", "sqrt"])
@pytest.mark.parametrize("simulator", ["statetensor", "densitytensor", "unitarytensor"])
@pytest.mark.parametrize("execution", ["eager", "jit", "grad"])
def test_nonlinear_symbolic_parameters(
    expression: str, simulator: str, execution: str
) -> None:
    theta = Symbol("theta")
    angle = {
        "sin": sin(theta),
        "cos": cos(theta),
        "exp": exp(theta),
        "sqrt": sqrt(theta + 1),
    }[expression]
    value = 0.3
    circuit = Circuit(1).Rx(angle, 0)
    simulate = tk_to_qujax(circuit, {theta: 0}, simulator=simulator)

    if execution == "grad":

        def objective(parameter: jnp.ndarray) -> jnp.ndarray:
            return jnp.real(simulate(parameter).reshape(-1)[0])

        angle_value = float(angle.subs(theta, value))
        angle_derivative = float(angle.diff(theta).subs(theta, value))
        if simulator == "densitytensor":
            expected_gradient = -np.pi * np.sin(np.pi * angle_value) / 2
        else:
            expected_gradient = -np.pi * np.sin(np.pi * angle_value / 2) / 2
        expected_gradient *= angle_derivative
        actual_gradient = jax.jit(jax.grad(objective))(jnp.array([value]))
        np.testing.assert_allclose(
            actual_gradient, [expected_gradient], rtol=1e-5, atol=1e-6
        )
    else:
        circuit.symbol_substitution({theta: value})
        state = circuit.get_statevector()
        expected: np.ndarray
        if simulator == "statetensor":
            expected = state
        elif simulator == "densitytensor":
            expected = np.outer(state, state.conj())
        else:
            expected = circuit.get_unitary()
        if execution == "jit":
            simulate = jax.jit(simulate)
        actual = np.asarray(simulate(jnp.array([value])))
        np.testing.assert_allclose(
            actual.reshape(expected.shape), expected, rtol=1e-6, atol=1e-7
        )
