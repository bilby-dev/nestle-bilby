import bilby
import pytest
from nestle_bilby import Nestle


@pytest.fixture()
def SamplerClass():
    return Nestle


@pytest.fixture()
def create_sampler(SamplerClass, bilby_gaussian_likelihood_and_priors, tmp_path):
    likelihood, priors = bilby_gaussian_likelihood_and_priors

    def create_fn(**kwargs):
        return SamplerClass(
            likelihood,
            priors,
            outdir=tmp_path / "outdir",
            label="test",
            use_ratio=False,
            **kwargs,
        )

    return create_fn


@pytest.fixture
def sampler(create_sampler):
    return create_sampler()


def test_default_kwargs(sampler):
    expected = dict(
        method="multi",
        npoints=500,
        update_interval=None,
        npdim=None,
        maxiter=None,
        maxcall=None,
        dlogz=None,
        decline_factor=None,
        rstate=None,
        callback=None,
        steps=20,
        enlarge=1.2,
    )
    sampler.kwargs["callback"] = None  # ignore callback for testing
    assert sampler.kwargs == expected


@pytest.mark.parametrize(
    "equiv",
    bilby.core.sampler.base_sampler.NestedSampler.npoints_equiv_kwargs,
)
def test_translate_kwargs(create_sampler, equiv):
    expected = dict(
        method="multi",
        npoints=123,
        update_interval=None,
        npdim=None,
        maxiter=None,
        maxcall=None,
        dlogz=None,
        decline_factor=None,
        rstate=None,
        callback=None,
        steps=20,
        enlarge=1.2,
    )

    sampler = create_sampler(**{equiv: 123})
    sampler.kwargs["callback"] = None  # ignore callback for testing
    assert sampler.kwargs == expected


def test_verbose_kwargs(create_sampler):
    sampler = create_sampler(verbose=True)
    assert "verbose" not in sampler.kwargs
    assert sampler.kwargs["callback"] is not None
