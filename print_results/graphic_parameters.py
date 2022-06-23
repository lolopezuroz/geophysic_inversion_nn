class Graphic_parameters:

    def __init__(self,
                 name="",
                 title="",
                 cmap="binary_r",
                 vmin=None,
                 vmax=None,
                 extension="png"):

        self.name = name
        self.title = title
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.extension = extension

parameters = {
    "ice_velocity": Graphic_parameters(
            name="ice_velocity",
            title="Vitesse d'écoulement (m/an)",
            cmap="magma_r",
            vmin=0.,
            vmax=100.
        ),
    "slope": Graphic_parameters(
            name="slope",
            title="Pentes °",
            cmap="Reds",
            vmin=0.,
            vmax=90.
        ),
    "ice_occupation": Graphic_parameters(
            name="ice_occupation",
            title="Glacier classification",
            cmap="bwr",
            vmin=0.,
            vmax=1.
        ),
    "ice_thickness": Graphic_parameters(
            name="ice_thickness",
            title="Épaisseur de glace (m)",
            cmap="Blues",
            vmin=0.,
            vmax=500.
        )
}