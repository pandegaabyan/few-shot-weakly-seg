import click

from config.config_maker import gen_id


@click.command()
@click.argument(
    "size",
    type=int,
    default=3,
)
def main(size: int):
    print(gen_id(size))


if __name__ == "__main__":
    main()
