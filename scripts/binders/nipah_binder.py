import random
import fire
import bagel as bg
import os
import logging
import pathlib as pl

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def main(
    use_modal: bool = False,
    binder_sequence: str = None,
    optimization_params: dict = None,
    output_dir: str = 'data/nipah_binder'
):
    # PART 1: Define the target protein
    # Carbonic anhydrase 4, CAH4_HUMAN, Gene: CA4
    # PDB ID: 1ZNC; chain A; UniProt ID: P22748
    target_sequence = "ICLQKTSNQILKPKLISYTLGQSGTCITDPLLAMDEGYFAYSHLERIGSCSRGVSKQRIIGVGEVLDRGDEVPSLFMTNVWTPPNPNTVYHCSAVYNNEFYYVLCAVSTVGDPILNSTYWSGSLMMTRLAVKPKSNGGGYNQHQLALRSIEKGRYDKVMPYGPSGIKQGDTLYFPAVGFLVRTEFKYNDSNCPITKCQYSKPENCRLSMGIRPNSHYILRSGLLKYNLSDGENPKVVFIEISDQRLSIGSPSKIYDSLGQPVFYQASFSWDTMIKFGDVLTVNPLVVNWRNNTVISRPGQSQCPRFNTCPEICWEGVYNDAFLIDRINWISAGVFLDSNQTAENPVFTVFKDNEILYRAQLASEDTNAQKTITNCFLLKNKIWCISLVEIYDTGDNVIRPKLFAVKIPEQCTH"

    # Define the mutability of the residues, all immutable in this case since this is the target sequence
    mutability = [False for _ in range(len(target_sequence))]

    # Define the chain
    residues_target = [
        bg.Residue(name=aa, chain_ID='NIPAH', index=i, mutable=mut)
        for i, (aa, mut) in enumerate(zip(target_sequence, mutability))
    ]

    # Now define residues in the hotspot where you want to bind.
    residue_ids = [
        [24, 25, 26],
        [47, 48, 49, 50, 51, 52],
        [113, 114, 115],
        [197, 198, 199],
        [209, 210, 211, 212],
        [212, 213, 214],
        [266, 267, 268],
        [295, 296, 297, 298, 299, 300, 301, 302],
        [312, 313, 314, 315, 316, 317],
        [337, 338, 339, 340, 341, 342, 343],
        [363, 364, 365],
        [365, 366, 367, 368, 369],
        [387, 388, 389],
        [389, 390, 391],
        [391, 392, 393, 394],
        [396, 397, 398]
        ]
    residue_ids = [item for sublist in residue_ids for item in sublist]

    residues_hotspot = [residues_target[i] for i in residue_ids]
    target_chain = bg.Chain(residues=residues_target)

    # PART 2: Define the binder
    binder_length = 30
    if binder_sequence is None:
        # Start with a random sequence of amino acids selecting randomly from the 30 amino acids
        binder_sequence = ''.join([random.choice(list(bg.constants.aa_dict.keys())) for _ in range(binder_length)])
    else:
        # or restart from a previously designed sequence
        assert len(binder_sequence) == binder_length, 'Binder sequence must be of length 30'

    # Define the mutability of the residues, all mutable in this case since this is the design sequence
    mutability = [True for _ in range(len(binder_sequence))]
    # Define the chain
    residues_binder = [
        bg.Residue(name=aa, chain_ID='BIND', index=i, mutable=mut)
        for i, (aa, mut) in enumerate(zip(binder_sequence, mutability))
    ]
    binder_chain = bg.Chain(residues=residues_binder)

    # PART 3: Define the Oracles and EnergyTerms

    # Define the ESMFold Oracle
    config = {
        'output_pdb': False,
        'output_cif': False,
        'glycine_linker': 50 * "G",
        'position_ids_skip': 512,
    }

    esmfold = bg.oracles.ESMFold(
        use_modal=use_modal, config=config
    )

    # Define the energy terms to be applied to the chain
    energy_terms = [
        bg.energies.PTMEnergy(
            oracle=esmfold,
            weight=1.0,
        ),
        bg.energies.OverallPLDDTEnergy(
            oracle=esmfold,
            weight=1.0,
        ),
        bg.energies.PLDDTEnergy(
            oracle=esmfold,
            residues=residues_binder,
            weight=2.0
        ),
        bg.energies.HydrophobicEnergy(
            oracle=esmfold,
            weight=1.0
            ),
        bg.energies.PAEEnergy(
            oracle=esmfold,
            residues=[residues_hotspot, residues_binder],
            weight=6.0,
        ),
        bg.energies.SeparationEnergy(
            oracle=esmfold,
            residues=[residues_hotspot, residues_binder],
            weight=1.0,
        ),
    ]

    # PART 4: Define the State
    state = bg.State(
        chains=[binder_chain, target_chain],
        energy_terms=energy_terms,
        name='state',
    )

    # PART 5: Define the System
    initial_system = bg.System(
        states=[state],
        name='nipah-binder'
    )

    # PART 6: Define the minimizer and run the optimization
    mutator = bg.mutation.Canonical()


    # Use optimization parameters if provided, otherwise use defaults
    if optimization_params is None:
        optimization_params = {
            'high_temperature': 1.0,
            'low_temperature': 0.1,
            'n_steps_high': 100,
            'n_steps_low': 400,
            'n_cycles': 100,
        }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    minimizer = bg.minimizer.SimulatedTempering(
        mutator=mutator,
        high_temperature=optimization_params['high_temperature'],
        low_temperature=optimization_params['low_temperature'],
        n_steps_high=optimization_params['n_steps_high'],
        n_steps_low=optimization_params['n_steps_low'],
        n_cycles=optimization_params['n_cycles'],
        preserve_best_system_every_n_steps=optimization_params['n_steps_high'] + optimization_params['n_steps_low'],
        log_frequency=1,
        log_path=pl.Path(os.path.join(current_dir, output_dir)),
    )

    # Return the best system
    best_system = minimizer.minimize_system(system=initial_system)
    return best_system


if __name__ == '__main__':
    fire.Fire(main)
