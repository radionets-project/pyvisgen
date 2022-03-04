import click
from pyvisgen.simulation.data_set import simulate_data_set


@click.command()
@click.argument("configuration_path", type=click.Path(exists=True, dir_okay=False))
# @click.option(
#     "--mode",
#     type=click.Choice(
#         [
#             "simulate",
#             "overview",
#         ],
#         case_sensitive=False,
#     ),
#     default="simulate",
# )
def main(configuration_path):
    simulate_data_set(configuration_path)


if __name__ == "__main__":
    main()
