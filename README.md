# BioBatchNet

## Installation

### Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/Manchester-HealthAI/BioBatchNet](https://github.com/Manchester-HealthAI/BioBatchNet
```

### Set Up the Environment

Create a virtual environment and install dependencies using `requirements.yaml`:

#### Using Conda:

```bash
conda env create -f requirements.yaml
conda activate BioBatchNet
```

## BioBatchNet Usage

```bash
cd BioBatchNet
```

For the IMC dataset, create a folder named `IMC_data` inside the `Data` directory and place the dataset inside:

```bash
mkdir -p Data/IMC_data
mv <your-imc-dataset> Data/IMC_data/
```

For scRNA-seq data, create a folder named `gene_data` inside the `Data` directory and place the dataset inside:

```bash
mkdir -p Data/gene_data
mv <your-scrna-dataset> Data/gene_data/
```

Modify the dataset, network layers, and other parameters in `config_imc.yaml`. 
Run the following command to remove batch effect for IMC data:

```bash
python IMC.py -c config_imc.yaml
```

For scRNA-seq data, use `config_gene.yaml` similarly. Run the following command to remove batch effect:

```bash
python Gene.py -c config_gene.yaml
```

## CPC Usage

Using the same environments with BioBatchNet, and all the results can be found in the following directory:

```bash
cd CPC/IMC_experiment
```

## To Do List

- [ ] Data download link
- [ ] Checkpoint
- [ ] Benchmark method results

## License

This project is licensed under the MIT License. See the LICENSE file for details.

