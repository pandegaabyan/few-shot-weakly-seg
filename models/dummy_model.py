from torchmeta import modules


class DummyModel(modules.MetaModule):
    def __init__(self, in_channels=3, out_channels=1, **kwargs):
        super().__init__()

        self.encoder = modules.MetaConv2d(in_channels, 2, kernel_size=1)

        self.decoder = modules.MetaConv2d(2, out_channels, kernel_size=1)

    def forward(self, x, params=None):
        # Encode and Decode
        x = self.encoder(x, self.get_subdict(params, "encoder"))
        x = self.decoder(x, self.get_subdict(params, "decoder"))
        return x
