import stdpopsim
from pytest import fixture


@fixture(
    params=[
        ("HomSap", "OutOfAfrica_3G09"),
        ("HomSap", "Africa_1T12"),
        ("HomSap", "PapuansOutOfAfrica_10J19"),
        ("DroMel", "African3Epoch_1S16"),
    ]
)
def demo(request):
    species_id, model_id = request.param
    species = stdpopsim.get_species(species_id)
    return species.get_demographic_model(model_id).model.to_demes()
