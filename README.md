This project implements the method described by Hall et al. to compute radial wave functions for helium using a Hartree–Fock approach.

The code supports general electronic states, but currently assumes orthogonality between wave functions. As a result, it works correctly for cases where this assumption holds, such as the helium ground state (1s²), but introduces inaccuracies for states where orthogonality is not properly enforced.

### Current Capabilities

* Correctly computes the radial wave function and energy for the helium ground state (1s²).
* Produces the correct qualitative form of the wave function for the 1s2s triplet state.

### Known Limitations

* The implementation does not enforce orthogonality between orbitals.
* This leads to incorrect energy values for the 1s2s triplet state, despite the wave function shape being reasonable.

### Files

* `helium_atom_gs.py`
  Runs the Hartree(-Fock) algorithm for the helium ground state (1s²). This implementation is stable and produces correct results.

* `helium_atom.py`
  Runs the Hartree–Fock algorithm for the 1s2s triplet state. While the wave function form is captured, the computed energy is inaccurate due to missing orthogonality constraints.

### Notes

Future improvements should include explicit orthogonalization of orbitals (e.g., via Gram–Schmidt or similar methods) to ensure accurate energy calculations for excited states.
