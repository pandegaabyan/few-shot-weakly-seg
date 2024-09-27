from torchmeta import modules


class DummyModel(modules.MetaModule):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Simple encoder block: keeps input size with padding
        self.encoder = modules.MetaConv2d(in_channels, 4, kernel_size=3, padding=1)

        # Simple decoder block: also keeps the input size with padding
        self.decoder = modules.MetaConv2d(4, out_channels, kernel_size=3, padding=1)

    def forward(self, x, params=None):
        # Encode and Decode
        x = self.encoder(x, self.get_subdict(params, "encoder"))
        x = self.decoder(x, self.get_subdict(params, "decoder"))
        return x
