class Graphic_parameters:

    def __init__(self,
                 name="",
                 title="",
                 cmap="binary_r",
                 vmin=None,
                 vmax=None,
                 extension=".png",
                 save_location="."):

        self.array = []
        self.name = name
        self.title = title
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.extension = extension
        self.save_location = save_location

    def set_array(self,array):
        self.array = array

    def set_save_location(self,save_location):
        self.save_location = save_location

inputs_parameters = {
"ice_velocity":Graphic_parameters(
        name="ice_velocity",
        title="Vitesse d'écoulement (m/an)",
        cmap="magma",
        vmin=0.,
        vmax=100.
),
"slope":Graphic_parameters(
        name="slope",
        title="Pentes °",
        cmap="Reds",
        vmin=0.,
        vmax=90.
    )
}

groundtruths_parameters = {
"ice_occupation":Graphic_parameters(
        name="ice_occupation",
        title="Glacier classification",
        cmap="bwr",
        vmin=0.,
        vmax=1.
    ),
"ice_thickness":Graphic_parameters(
        name="ice_thickness",
        title="Épaisseur de glace (m)",
        cmap="Blues",
        vmin=0.,
        vmax=500.
    )
}