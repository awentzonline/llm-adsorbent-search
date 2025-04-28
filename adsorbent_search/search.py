import asyncio
from dataclasses import dataclass
from typing import Any, Dict

from ase import Atoms
import click
import logfire
from pydantic_ai import Agent, RunContext

from .fairchem_relax_tool import evaluate_adsorbent_on_adsorbate
from .models import Adsorbent, FAIRChemRelaxConfig


ADSORBENT_DATABASE: Dict[str, Adsorbent] = {}


@dataclass
class AdsorbentSearchDeps:
    fairchem_cfg: FAIRChemRelaxConfig


logfire.configure()
# DEFAULT_LLM = 'anthropic:claude-3-7-sonnet-latest'
# DEFAULT_LLM = 'openai:gpt-4o'
DEFAULT_LLM = 'google-gla:gemini-2.0-flash'
propose_adsorbates_agent = Agent(
    DEFAULT_LLM,
    result_type=str,
    deps=AdsorbentSearchDeps,
)
Agent.instrument_all()


@propose_adsorbates_agent.tool
def add_adsorbent(ctx: RunContext, adsorbent: Adsorbent) -> str:
    """
    Adds a new adsorbent definiton to the material database under `id_name`.
    The adsorbent is defined by a function which returns ase.Atoms

    :param id_name: string used to refer to this material
    :param code: string containing a python function named `create_adsorbent()` which returns `ase.Atoms`:
        ```
        def create_adsorbent() -> ase.Atoms:
            ...
        ```
    :return: string with a message indicating success or an error
    """
    try:
        atoms = adsorbent.get_atoms()
        if not isinstance(atoms, Atoms):
            raise ValueError('Function did not return an `Atoms` instance')
    except Exception as e:
        return 'Error: ' + str(e)

    ADSORBENT_DATABASE[adsorbent.name] = adsorbent
    return 'Success'


@propose_adsorbates_agent.tool
def test_molecule_on_adsorbent(
    ctx: RunContext, adsorbate: str, adsorbent_name: str
) -> Dict[str, Any]:
    """
    Simulates the interaction of a given adsorbate molecule on the adsorbent.

    :param molecule: string that defines the adsorbate e.g. 'CO2'
    :param adsorbent_name: string referencing an entry in the material database
    :return: dictionary of evaluation metrics
    """
    if adsorbent_name not in ADSORBENT_DATABASE:
        return f'Adsorbent not found: {adsorbent_name}'

    ads_conf = ADSORBENT_DATABASE[adsorbent_name]
    adsorbent_atoms = ads_conf.get_atoms()
    adsorbate_atoms = Atoms(adsorbate)

    try:
        results = evaluate_adsorbent_on_adsorbate(
            ctx.deps.fairchem_cfg, adsorbate_atoms, adsorbent_atoms
        )
    except Exception as e:
        return dict(error=str(e))

    return results


@propose_adsorbates_agent.system_prompt
def adsorbent_search_prompt():
    return (
        "You're a world-class material engineer, machine learning researcher, and physicist.\n"
        "You are searching for an adsorbent material which works well for the user's goal.\n"
        "Design materials by writing a python function that returns an ASE Atoms instance.\n"
        "Test the usefulness of these materials against molecules of your choosing.\n"
        "Use the results of your experiments to generate ever better materials.\n"
        "Stop when you think you've produced the best possible material or after 10 materials.\n"
        "For your final output, return the name of the best material and a summary of your discoveries.\n"
        "The materials should be as inexpensive and easy to synthesize as possible.\n"
        "Do not ask if the user if they want to search for a material--of course they do.\n"
    )


async def run_agent(goal, cfg):
    deps = AdsorbentSearchDeps(fairchem_cfg=cfg)
    result = await propose_adsorbates_agent.run(goal, deps=deps)
    print(result.all_messages())
    print(result.output)


@click.command()
@click.argument('goal')
@click.option('--model_name', default='EquiformerV2-31M-S2EF-OC20-All+MD')
@click.option('--model_local_cache', default='/tmp/fairchem_cache/')
@click.option('--cpu', is_flag=True)
def main(goal, model_name, model_local_cache, cpu):
    cfg = FAIRChemRelaxConfig(
        model_name=model_name,
        model_local_cache=model_local_cache,
        cpu=cpu
    )
    asyncio.run(run_agent(goal, cfg))


if __name__ == '__main__':
    main()
