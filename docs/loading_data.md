# Loading Data for momi3

The only required user specified data for any function in ``momi3`` is the **number of haploid samples** in each population and the **observed frequency spectrum**. 

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
  
  afs_samples = {"P0": sample_size * 2, "P1": sample_size * 2} # msprime simulates diploids so we need the multiplier of 2
  afs = ts.allele_frequency_spectrum(
      sample_sets=[ts.samples([1]), ts.samples([2])],
      span_normalise=False,
      polarised=True
  )
```

For any ``tskit`` tree sequence object, one can call on ``allele_frequency_spectrum`` to retrieve the observed frequency spectrum ``afs``. In ``momi3``, the number of samples (``afs_samples``) and observed frequency spectrum (``afs``) is all the data necessary to run all of its core functions. 

## VCZ and VCF files

Suppose we use the previous example and construct a VCF file to use.

```python
vcf_path = "./example.vcf"
with open(vcf_path, "w") as vcf_file:
    ts.write_vcf(vcf_file)
```

To obtain the site frequency spectrum. One can call on ``joint_sfs_from_haploids`` using a genotype matrix.
To obtain that genotype matrix easily, we can convert the VCF file to VCZ. 

```python
from demestats.fit.util import joint_sfs_from_haploids
import sgkit as sg
import bio2zarr.vcf as v2z

# If you already have a vcz file or a vcz dataset you may skip this
vcz_path = "./example.vcz"
v2z.convert([vcf_path], vcz_path)

# you have to specify which haplotypes you wish to use for construct the SFS
pops = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],   # population 1
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],   # population 2
]

path_or_ds = vcz_path 
# you can enter the vcz dataset directly or a path to it
ds = sg.load_dataset(path_or_ds) if isinstance(path_or_ds, str) else path_or_ds
gt = ds["call_genotype"].compute().values.astype(np.int16)
n_variants = gt.shape[0]
genotype = gt.reshape(n_variants, -1)
jsfs = joint_sfs_from_haploids(genotype, pops)
```

