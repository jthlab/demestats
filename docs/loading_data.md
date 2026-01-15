# Loading Data for momi3

The only required user specified data for any function in ``momi3`` is the **number of samples** in each population and the **observed frequency spectrum**. 

## ``Tskit`` Tree Sequence Data


One can simulate tree sequence data with ``tskit`` package.

```python

  demo = msp.Demography()
  demo.add_population(initial_size=5000, name="anc")
  demo.add_population(initial_size=5000, name="P0")
  demo.add_population(initial_size=5000, name="P1")
  demo.set_symmetric_migration_rate(populations=("P0", "P1"), rate=0.0001)
  demo.add_population_split(time=1000, derived=["P0", "P1"], ancestral="anc")
  
  sample_size = 10 # we simulate 10 diploids
  samples = {"P0": sample_size, "P1": sample_size}
  ts = msp.sim_mutations(
      msp.sim_ancestry(
          samples=samples, demography=demo,
          recombination_rate=1e-8, sequence_length=1e8, random_seed=12
      ),
      rate=1e-8, random_seed=13
  )
  
  afs_samples = {"P0": sample_size * 2, "P1": sample_size * 2}
  afs = ts.allele_frequency_spectrum(
      sample_sets=[ts.samples([1]), ts.samples([2])],
      span_normalise=False,
      polarised=True
  )
```

For any ``tskit`` tree sequence object, one can call on ``allele_frequency_spectrum`` to retrieve the observed frequency spectrum ``afs``. In ``momi3``, the number of samples (``afs_samples``) and observed frequency spectrum (``afs``) is all the data necessary to run all of its core functions. 

## VCZ files

## Convert VCF to VCZ Files
