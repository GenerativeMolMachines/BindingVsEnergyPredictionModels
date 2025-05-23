# BindingVsEnergyPredictionModels
Models for predicting binding, mfe and affinity between different structures and complexes

This module was created to enrich the knowledge graph and create new connections based on machine learning models.

Each unit has been taken from open source works, and **we do not claim to have appropriated the developments**. The description for each block will include references to the sources.

## [antibody_antigen](https://github.com/GenerativeMolMachines/BindingVsEnergyPredictionModels/tree/main/antibody_antigen)
Predicts the binding fact (True/False) for the antibody -> antigen complex.
 - [Source article](https://pubmed.ncbi.nlm.nih.gov/39363630/)
 - [Sourse code](https://github.com/TAI-Medical-Lab/MVSF-AB)

## [aptamer_protein](https://github.com/GenerativeMolMachines/BindingVsEnergyPredictionModels/tree/main/aptamer_protein)
Predicts affinity for the aptamer -> protein complex interaction.

PyBioMed was used to calculate descriptors.
XGBoost was used for prediction.

 - [Source article](https://pubs.acs.org/doi/10.1021/acs.jcim.3c00713)
 - [Sourse code](https://github.com/Meaw0415/APIPred)

## [aptamer_small_molecule](https://github.com/GenerativeMolMachines/BindingVsEnergyPredictionModels/tree/main/aptamer_small_molecule)
Predicts the binding fact (True/False) for the aptamer -> small molecule complex.
 - [Source article](https://academic.oup.com/bib/article/25/2/bbae002/7584787?login=false)
 - [Sourse code](https://github.com/Sowmya-R-Krishnan/RSAPred/)

## [nanobody_antigen](https://github.com/GenerativeMolMachines/BindingVsEnergyPredictionModels/tree/main/nanobody_antigen)
Predicts the binding fact (True/False) for the nanobody -> antigen complex.
 - [Source article](https://link.springer.com/chapter/10.1007/978-981-99-7074-2_18)
 - [Sourse code](https://github.com/sarwanpasha/Nanobody_Antigen)

## [protein_protein](https://github.com/GenerativeMolMachines/BindingVsEnergyPredictionModels/tree/main/protein_protein)
Predicts the binding fact (True/False) for the protein -> protein complex.

The model used is a Transformer-based uncertainty-aware model (TUnA) for PPI prediction. TUnA uses ESM-2 embeddings with Transformer encoders and incorporates a Spectral-normalized Neural Gaussian Process.

 - [Source article](https://academic.oup.com/bib/article/25/5/bbae359/7720609)
 - [Sourse code](https://github.com/Wang-lab-UCSD/TUnA/tree/main)

## [rna_rna](https://github.com/GenerativeMolMachines/BindingVsEnergyPredictionModels/tree/main/rna_rna)
Predicts minimum free energy (MFE) for the rna -> rna complex interaction.
 - [Source article](https://www.tbi.univie.ac.at/RNA/?_gl=1*1dj5tnj*_gcl_au*ODAxMTczOTU5LjE3NDE0NDk5MjY.)
 - [Sourse code](https://github.com/ViennaRNA/ViennaRNA?tab=readme-ov-file)

## [similarity](https://github.com/GenerativeMolMachines/BindingVsEnergyPredictionModels/tree/main/similarity)
Calculates small molecule similarity by Tanimoto algorithm and protein/DNA/RNA sequence similarity as a percentage of match by pairwise alignment of two sequences.

## Deployment
First go for set_up_env.py where it exists.
```
python3 set_up_env.py
```

If it creates .env file, use this command to start docker:
```
docker compose -f docker-compose.yaml --env-file .env up -d --build
```
Else:
```
docker compose -f docker-compose.yaml up -d --build
```
Then build it:
```
docker build . -t [container name]:latest
```
