import io

from ase import Atoms
from ase.build import add_adsorbate
import ase.io
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.data.oc.utils import DetectTrajAnomaly

from .models import FAIRChemRelaxConfig


def evaluate_adsorbent_on_adsorbate(cfg: FAIRChemRelaxConfig, adsorbate: Atoms, adsorbent: Atoms):
    evaluator = AdsorbentEvaluator()
    results = evaluator.evaluate(cfg, adsorbate, adsorbent)
    return results


class AdsorbentEvaluator:
    def evaluate(self, cfg: FAIRChemRelaxConfig, adsorbate: Atoms, adsorbent: Atoms):
        self.cfg = cfg

        if cfg.checkpoint_path:
            checkpoint_path = cfg.checkpoint_path
        else:
            checkpoint_path = model_name_to_local_file(
                cfg.model_name, local_cache=cfg.model_local_cache
            )

        calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=cfg.cpu)

        adsorbent = adsorbent.copy()
        adsorbate = adsorbate.copy()

        # relax adsorbent
        adsorbent.set_calculator(calc)
        adsorbent.center(vacuum=10.0, axis=2)
        self.relax(adsorbent)
        adsorbent.center(vacuum=10.0, axis=2)

        # place and relax adsorbate
        def eval_relaxation(atoms):
            traj_io = io.BytesIO()
            traj_out = Trajectory(traj_io, mode='w')
            self.relax(atoms, traj_out)
            traj_io.seek(0)
            traj = ase.io.read(traj_io, ":")
            detector = DetectTrajAnomaly(traj[0], traj[-1], traj[0].get_tags())

            results = dict(
                adsorbate_dissociated=detector.is_adsorbate_dissociated(),
                adsorbate_desorbed=detector.is_adsorbate_desorbed(),
                surface_changed=detector.has_surface_changed(),
                adsorbate_intercalated=detector.is_adsorbate_intercalated(),
                relaxed_energy=atoms.get_potential_energy(),
            )
            return results

        add_adsorbate(adsorbent, adsorbate, 1.2)
        results = eval_relaxation(adsorbent)
        return results

    def relax(self, atoms, trajectory=None):
        opt = BFGS(atoms, trajectory=trajectory)
        opt.run(fmax=self.cfg.opt_fmax, steps=self.cfg.opt_steps)
        return atoms
