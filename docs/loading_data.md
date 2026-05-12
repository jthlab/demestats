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
  
  # msprime simulates diploids so we need the multiplier of 2 to count the number of haploids
  afs_samples = {"P0": sample_size * 2, "P1": sample_size * 2} 
  afs = ts.allele_frequency_spectrum(
      sample_sets=[ts.samples([1]), ts.samples([2])],
      span_normalise=False,
      polarised=True
  )
```

For any ``tskit`` tree sequence object, one can call on ``allele_frequency_spectrum`` to retrieve the observed frequency spectrum ``afs``. In ``momi3``, the number of haploids (``afs_samples``) and observed frequency spectrum (``afs``) are all the data necessary to run all of its core functions. Please note that setting ``polarised=False`` results in tskit returning a folded SFS that is in "lower triangular" form. This form is not compatabile with ``momi3``. To create a folded SFS compatable with ``momi3``, please see section below and use the function ``joint_sfs_from_vcz`` with ``fold=True``. 

## VCZ and VCF files

Suppose we use the previous example and construct a VCF file to use.

```python
vcf_path = "./example.vcf"
with open(vcf_path, "w") as vcf_file:
    ts.write_vcf(vcf_file)
```

To obtain the site frequency spectrum. One can call on ``joint_sfs_from_vcz`` using a VCZ file path or dataset. Note that
all non-segregating sites will be ignored when constructing the site frequency spectrum. 

```python
from demestats.fit.util import joint_sfs_from_vcz
import sgkit as sg
import bio2zarr.vcf as v2z

# If you already have a vcz file or a vcz dataset you may skip this
vcz_path = "./example.vcz"
v2z.convert([vcf_path], vcz_path)

# you have to specify the index of which haplotypes you wish to use for construct the SFS
# you are picking out the columns/samples of vcf/vcz file you wish to include
pops = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],   # population 1
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],   # population 2
]

# you can either provide the path to the vcz or pass in a vcz dataset directly
# e.g. ds = sg.load_dataset(vcz_path) and pass in ds instead of vcz_path
unfolded_jsfs = joint_sfs_from_vcz(vcz_path, pops, ploidy = 2, fold=False) 
# to get folded sfs, set fold=True
# ploidy = 2 because we are working with diploids
```

## PLINK/BED files
One can use ``bio2zarr`` package to convert .bed, .bim, and .fam files to VCZ files.

```python
import bio2zarr.plink as p2z

root = p2z.convert(plink_file_path, vcz_output_path)
```

Suggestions for preprocessing your data:
- For multiple VCZ files, the joint site frequency spectrum (JSFS) can be computed independently for each file and subsequently summed, as long as the dimensionality is consistent across all files.
- If there is missing data, we suggest to remove the corresponding rows. But one can also attempt to impute the values.
- We recommend the JSFS to include sites that are fully represented in the sample, but if you have a dataset with missing data, you can also project or downsample the JSFS using the function ``downsample``. 

If there is a high degree of missing data, then one can downsample the JSFS by specifying a desired size.

```python
from demestats.jsfs import JSFS

# For population "P1" we simulated with 20 haploids but we downsample to 13
jsfs_obj = JSFS.from_dense(unfolded_jsfs, pops=["P0", "P1"]) # pops is the list of population names
new_jsfs_obj = jsfs_obj.downsample({"P0":20, "P1":13}) # you must provide the dictionary you wish to down sample to
unfolded_jsfs_downsample = new_jsfs_obj.todense() # convert back to a dense jsfs
```