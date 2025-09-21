from botorch.models import SingleTaskGP
from botorch.acquisition import qLogExpectedImprovement, qMaxValueEntropy
from gpytorch.kernels import MaternKernel

from ax.generators.torch.botorch_modular.surrogate import SurrogateSpec, ModelConfig
from ax.adapter.registry import Generators
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.center_generation_node import CenterGenerationNode
from ax.generation_strategy.transition_criterion import MinTrials
from ax.adapter.registry import Generators



def construct_simple_surrogate(gp: callable, kernel=None):
    if kernel is None:
        return SurrogateSpec(
            model_configs=[
                ModelConfig(
                    botorch_model_class=SingleTaskGP,
                ),
            ]
        )
    else:
        raise NotImplementedError("Custom kernel not implemented yet.")


def construct_gen_spec(
    surrogate_spec: SurrogateSpec, acqf_class: callable, acqf_options: dict = None
):
    generator_spec = GeneratorSpec(
        generator_enum=Generators.BOTORCH_MODULAR,
        model_kwargs={
            "surrogate_spec": surrogate_spec,
            "botorch_acqf_class": qLogExpectedImprovement,
            # Can be used for additional inputs that are not constructed
            # by default in Ax. We will demonstrate below.
            "acquisition_options": {},
        },
        # We can specify various options for the optimizer here.
        model_gen_kwargs={
            "model_gen_options": {
                "optimizer_kwargs": {
                    "num_restarts": 20,
                    "sequential": False,
                    "options": {
                        "batch_limit": 5,
                        "maxiter": 200,
                    },
                },
            },
        },
    )
    return generator_spec



def construct_generation_strategy(
    generator_spec: GeneratorSpec, node_name: str, transition_trials: int = 5
) -> GenerationStrategy:
    """Constructs a Center + Sobol + Modular BoTorch `GenerationStrategy`
    using the provided `generator_spec` for the Modular BoTorch node.
    """
    botorch_node = GenerationNode(
        node_name=node_name,
        generator_specs=[generator_spec],
    )

    # Sobol for initial space exploration
    sobol_node = GenerationNode(
        node_name="Sobol",
        generator_specs=[
            GeneratorSpec(
                generator_enum=Generators.SOBOL,
            ),
        ],
        transition_criteria=[
            # Transition to BoTorch node once there are `transition_trials` trials on the experiment.
            MinTrials(
                threshold=transition_trials,
                transition_to=botorch_node.node_name,
                use_all_trials_in_exp=True,
            )
        ],
    )
    # Center node is a customized node that uses a simplified logic and has a
    # built-in transition criteria that transitions after generating once.
    center_node = CenterGenerationNode(next_node_name=sobol_node.node_name)
    return GenerationStrategy(
        name=f"Center+Sobol+{node_name}", nodes=[center_node, sobol_node, botorch_node]
    )



def get_full_strategy(gp: callable, acqf_class: callable, kernel=None, transition_trials: int = 5):
    surrogate_spec = construct_simple_surrogate(gp=gp, kernel=kernel)
    generator_spec = construct_gen_spec(
        surrogate_spec=surrogate_spec,
        acqf_class=acqf_class,
    )
    generation_strategy = construct_generation_strategy(
        generator_spec=generator_spec,
        node_name=f"{gp.__name__}+{acqf_class.__name__}",
        transition_trials=transition_trials
    )


    return generation_strategy

if __name__ == "__main__":
    # Example usage
    gs = get_full_strategy(gp=SingleTaskGP, acqf_class=qLogExpectedImprovement)
    print(gs)